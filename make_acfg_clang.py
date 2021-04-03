'''
make_acfg.py

Performs CFG extraction from binaries in parallel using
the multiprocessing module. If successful, each process will:
1) Get the adjacency matrix of the CFG (not per function, not graphlets)
2) For each basic block (node), compute syntactic features
3) Encode into the DGCNN format

Before using this program, install angr's dependencies with:
$ sudo apt install python3-dev libffi-dev build-essential virtualenvwrapper
Then install with:
$  pip3 install angr

nosajmik, April 2021
'''

import sys
import os
import angr
import gc
import multiprocessing as mp


def get_disas_as_list(bb):
    '''
    Gets the disassembly of a basic block, bb, as a tuple of
    (entry point, opcode, args).
    After digging angr's source code, found a way to get address, opcode,
    and args for each instruction. This was not documented!
    '''
    result = []
    for insn in bb.capstone.insns:
        result.append((insn.address, insn.mnemonic, insn.op_str))
    return result


def get_branch_type(insns):
    '''
    insns = result of calling get_disas_as_list on a basic block.
    For branch type, let it be an integer, but one-hot encoded within
    an 8-dimensional vector.
    0 indicates simple and unconditional branch (B, BX, BR, RET).
    1 indicates unconditional branch with link (used frequently for function
      calls, since ARM has no dedicated CALL instruction unlike x86) (BL, BLX, BLR).
    2 indicates compare- or test-and-branch w/o condition flag change
      (CBZ, CBNZ, TBZ, TBNZ).
    3 indicates conditional branch with no link (B/BX + {condition}).
    4 indicates conditional branch with link (BL/BLX + {condition}).
    5 indicates POP.
    6 indicates POP + {condition}.
    7 indicates all others (miscellaneous).
    '''
    conds = {"eq", "ne", "cs", "hs", "cc", "lo", "mi", "pl", "vs", "vc",
             "hi", "ls", "ge", "lt", "gt", "le"}
    cb_no_link = {"b" + c for c in conds}.union({"bx" + c for c in conds})
    cb_link = {"bl" + c for c in conds}.union({"blx" + c for c in conds})
    br_opcode = insns[-1][1]

    if br_opcode in {"cbz", "cbnz", "tbz", "tbnz"}:
        return 2
    elif br_opcode.startswith("pop"):
        if br_opcode == "pop":
            return 5
        else:
            return 6
    elif br_opcode.startswith("b"):
        if br_opcode in {"b", "bx", "br"}:
            return 0
        elif br_opcode in {"bl", "blx", "blr"}:
            return 1
        elif br_opcode in cb_no_link:
            return 3
        elif br_opcode in cb_link:
            return 4
        else:
            return 7
    elif br_opcode == "ret":
        return 0
    else:
        return 7


def get_num_math_insns(insns):
    arithmetics = {"adc", "adcs", "add", "adds", "adr", "adrp", "cmn", "cmp",
                   "madd", "mneg", "msub", "mul", "neg", "negs", "ngc",
                   "ngcs", "sbc", "sbcs", "sdiv", "smaddl", "smnegl", "smsubl",
                   "smulh", "smull", "sub", "subs", "udiv", "umaddl", "umnegl",
                   "umsubl", "umulh", "umull"}
    count = 0
    for insn in insns:
        if insn[1] in arithmetics:
            count += 1
    return count


def get_num_log_insns(insns):
    logicals = {"and", "ands", "asr", "bic", "bics", "eon", "eor", "lsl", "lsr",
                "mov", "movk", "movn", "movz", "mvn", "orn", "orr", "ror", "tst"}
    count = 0
    for insn in insns:
        if insn[1] in logicals:
            count += 1
    return count


def get_num_immediates(insns):
    '''
    Although # is optional for immediates in ARMv8, angr will prepend it
    to any immediate values (constants).
    '''
    count = 0
    for insn in insns:
        count += insn[2].count("#")
    return count


def chunks(lst, n):
    '''
    Yield successive n-sized chunks from lst
    Shamelessly stolen from https://stackoverflow.com/questions
    /312443/how-do-you-split-a-list-into-evenly-sized-chunks
    '''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def worker(basenames, outfile_suffix, num):
    # If aborted in the middle, delete the files!
    with open(str(num) + "-" + outfile_suffix, 'a') as file:
        for bname in basenames:
            if "clang" not in bname:
                continue

            path = os.path.join(sys.argv[1], bname)
            try:
                p = angr.Project(path, load_options={"auto_load_libs": False})
                cfg = p.analyses.CFGFast()
                # normalize() ensures that basic blocks do not overlap
                cfg.normalize()
                
                # Map each block to a number (block ID)
                # block ID starts at 0, and is used as an index
                # for degrees, insn_list, and feat_list
                block_map = {}
                for n in cfg.graph:
                    if n.block is not None:
                        block_map[len(block_map)] = n.block
                        
                degrees = []
                for n in cfg.graph:
                    if n.block is not None:
                        degrees.append(cfg.graph.degree(n))
                assert(len(block_map) == len(degrees))
                
                insn_list = []
                for i in range(len(block_map)):
                    insn_list.append(get_disas_as_list(block_map[i]))

                feat_list = []
                for i in range(len(block_map)):
                    insns = insn_list[i]
                    feat = [block_map[i].instructions, get_branch_type(insns)] +\
                            [get_num_math_insns(insns), get_num_log_insns(insns),
                            get_num_immediates(insns)] + [degrees[i]]
                    feat_list.append(feat)

                # Need a reverse block map for adjacency matrix
                reverse_block_map = {}
                for n in cfg.graph:
                    if n.block is not None:
                        reverse_block_map[n.block] = len(reverse_block_map)
                assert(len(block_map) == len(reverse_block_map))

                adj_lists = [ [] for i in range(len(block_map)) ]
                for n, nbrdict in cfg.graph.adjacency():
                    if n.block is not None:
                        adj_list_idx = reverse_block_map[n.block]
                        for nbr, _ in nbrdict.items():
                            if nbr.block is not None:
                                id_of_nbr = reverse_block_map[nbr.block]
                                adj_lists[adj_list_idx].append(id_of_nbr)
                
                if "clang-9" in bname:
                    label = 1
                elif "clang-5.0" in bname:
                    label = -1
                else:
                    raise RuntimeError("clang version should be 5.0 or 9")
                
                # from data README.md: n l
                header = f"{len(feat_list)} {label}\n"
                print(header, end='')
                file.write(header)
                for i in range(len(feat_list)):
                    # start with t (tag is 0) m, then m numbers for adjacent
                    # node indices, then d feature dimensions
                    adjs = ' '.join([str(a) for a in adj_lists[i]])
                    feats = ' '.join([str(a) for a in feat_list[i]])
                    line = f"0 {len(adj_lists[i])} {adjs} {feats}\n"
                    print(line, end='')
                    file.write(line)

                del p
                gc.collect()

            except Exception as e:
                print(e)
                print(f"Skipping {bname} due to angr parsing error")
                del p
                gc.collect()
                continue


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} directory-of-unique-binaries outfile-suffix")
        exit(1)

    assert(os.path.isdir(sys.argv[1]))
    basenames = os.listdir(sys.argv[1])

    NUM_CORES = 8
    split_basenames = list(chunks(basenames, len(basenames) // NUM_CORES + 1))
    assert(len(split_basenames) == NUM_CORES)

    processes = [mp.Process(target=worker,
                            args=(split_basenames[i], sys.argv[2], i))
                            for i in range(NUM_CORES)]

    # start each worker, and block until all of them finish
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # need to aggregate each worker's outfile
    num_graphs = 0
    for i in range(NUM_CORES):
        with open(str(i) + "-" + sys.argv[2], 'r') as file:
            lines = file.readlines()
            linenum = 0
            while(linenum < len(lines)):
                num_nodes = int(lines[linenum].strip().split(' ')[0])
                num_graphs += 1
                linenum += (num_nodes + 1)
    
    # write num_graphs to a dummy file,
    # and concatenate all the outfiles
    with open("dummyfile", 'w') as file:
        file.write(str(num_graphs) + "\n")
    os.system(f"cat dummyfile *-{sys.argv[2]} > {sys.argv[2]}")
    os.system(f"rm dummyfile *-{sys.argv[2]}")


if __name__ == "__main__":
    main()
