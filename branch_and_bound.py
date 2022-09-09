#import RNA
import random
import math
import numpy as np
import pandas as pd
import heapq
import copy
import time
import argparse
import multiprocessing as mp
import sys
sys.path.append('./cpp')
import four_russians
import networkx as nx

NUM_THREAD = 8
MAX_BOND = 0

def getAminos(sequence):
    aminos = ''
    for h in range(len(sequence) // 3):
        amino = TableDNA[sequence[h * 3: h * 3 + 3]]
        aminos += amino
    return aminos


def getRandomSyn(aminos):
    rand_seq = ''
    for amino in aminos:
        rand_seq += random.choice(InvTableDNA[amino])
    return rand_seq


def getSynonymNumber(aminos):
    count = 1
    for a in aminos:
        count *= len(InvTableDNA[a])
    return count


def make_binary(cds: str, BinaryAminoTable) -> list:
    """
    Scorro gli amminoacidi della CDS e prendo da BinaryAminoTable le rispettive liste
    :param cds: CDS in input
    :return: lista contenente gli array della BinaryAminoTable correlati agli amminoacidi della CDS
    """
    sequences = [cds[i * 3:i * 3 + 3] for i in range(len(cds) // 3)]
    binary_bases = []
    for idx, codon in enumerate(sequences):
        for key in find_amino_from_codon(codon, TableDNA).keys():
            for cod in BinaryAminoTable[key]:
                binary_bases.append(cod)

    return binary_bases


InvTableDNA = {
    'A': ['gcu', 'gcc', 'gca', 'gcg'],
    'L': ['uua', 'uug', 'cuu', 'cuc', 'cua', 'cug'],
    'R': ['cgu', 'cgc', 'cga', 'cgg', 'aga', 'agg'],
    'K': ['aaa', 'aag'],
    'N': ['aau', 'aac'],
    'M': ['aug'],
    'D': ['gau', 'gac'],
    'F': ['uuu', 'uuc'],
    'C': ['ugc', 'ugu'],
    'P': ['ccu', 'ccc', 'cca', 'ccg'],
    'Q': ['caa', 'cag'],
    'S': ['ucu', 'ucc', 'uca', 'ucg', 'agu', 'agc'],
    'E': ['gaa', 'gag'],
    'T': ['acu', 'acc', 'aca', 'acg'],
    'G': ['ggu', 'ggc', 'gga', 'ggg'],
    'W': ['ugg'],
    'H': ['cau', 'cac'],
    'Y': ['uau', 'uac'],
    'I': ['auu', 'auc', 'aua'],
    'V': ['guu', 'guc', 'gua', 'gug'],
    '#': ['uaa', 'uga', 'uag']
}
Branches = {
    'L': [[1, 1, 10], [4, 1, 15]],
    'R': [[8, 2, 10], [4, 2, 15]],
    'S': [[1, 4, 15], [8, 2, 5]],
    '#': [[1, 8, 10], [1, 2, 8]]
}
#
# Reverse InvTableDNA
#

TableDNA = {i: k for k, v in InvTableDNA.items() for i in v}


StandardBinaryAminoTable = {
    'A': [2, 4, 15],
    'L': [5, 1, 15],
    'R': [12, 2, 15],
    'K': [8, 8, 10],
    'N': [8, 8, 5],
    'M': [8, 1, 2],
    'D': [2, 8, 5],
    'F': [1, 1, 5],
    'C': [1, 2, 5],
    'P': [4, 4, 15],
    'Q': [4, 8, 10],
    'S': [9, 6, 15],
    'E': [2, 8, 10],
    'T': [8, 4, 15],
    'G': [2, 2, 15],
    'W': [1, 2, 2],
    'H': [4, 8, 5],
    'Y': [1, 8, 5],
    'I': [8, 1, 13],
    'V': [2, 1, 15],
    '#': [1, 10, 10]
}
AllBinaryAminoTable = {
    'A': [2, 4, 15],
    'L': [4, 1, 15],
    'R': [4, 2, 15],
    'K': [8, 8, 10],
    'N': [8, 8, 5],
    'M': [8, 1, 2],
    'D': [2, 8, 5],
    'F': [1, 1, 5],
    'C': [1, 2, 5],
    'P': [4, 4, 15],
    'Q': [4, 8, 10],
    'S': [1, 4, 15],
    'E': [2, 8, 10],
    'T': [8, 4, 15],
    'G': [2, 2, 15],
    'W': [1, 2, 2],
    'H': [4, 8, 5],
    'Y': [1, 8, 5],
    'I': [8, 1, 13],
    'V': [2, 1, 15],
    '#': [1, 8, 10]
}

DecodingTable = {
    1: ['u'],
    2: ['g'],
    3: ['g', 'u'],
    4: ['c'],
    5: ['c', 'u'],
    6: ['c', 'g'],
    7: ['c', 'g', 'u'],
    8: ['a'],
    9: ['a', 'u'],
    10: ['a', 'g'],
    11: ['a', 'g', 'u'],
    12: ['a', 'c'],
    13: ['a', 'c', 'u'],
    14: ['a', 'c', 'g'],
    15: ['a', 'c', 'g', 'u']
}


def find_amino_from_codon(codon: str,
                          table: dict) -> dict:
    """
    Find all possible aminos that can be codified by a (partly empty) codon
    :param: a 3-chars string representing a codon. May contain '*' as don't care
    either at the beginning or at the end of the sequence
    :return: a dictionary with all possible amino acids that the sequence codifies
    """

    aminos = {}

    if codon.isalpha():
        if table[codon] in aminos:
            aminos[table[codon]].append(codon)
        else:
            aminos[table[codon]] = [codon]

    if codon[0] == '*':
        partial_codon = codon.lstrip('*')
        offset = len(codon) - len(partial_codon)
        for i in table:
            if partial_codon == i[offset:]:
                if table[i] in aminos:
                    aminos[table[i]].append(i)
                else:
                    aminos[table[i]] = [i]

    if codon[2] == '*':
        partial_codon = codon.rstrip('*')
        offset = len(codon) - len(partial_codon)

        for i in table:
            if partial_codon == i[:3 - offset]:
                if table[i] in aminos:
                    aminos[table[i]].append(i)
                else:
                    aminos[table[i]] = [i]

    return aminos


def reverse_Bits(n, no_of_bits):
    result = 0
    for _ in range(no_of_bits):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result


def check_possible_pairing_no_writing(base1, base2):
    reversed = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    return 1 if np.bitwise_and(base1, reversed[base2]) > 0 else 0


def basic_dp(sequence):
    len_seq = len(sequence)
    M = np.zeros((len_seq, len_seq))

    for d in range(4, len_seq):  # [4, len_seq)
        for i in range(len_seq-d):  # [0, len_seq-d)
            j = i+d
            temp = 0
            for k in range(i, j):
                temp = max(temp, M[i][k-1]+M[k+1][j-1] + (
                    check_possible_pairing_no_writing(sequence[k], sequence[j]) if j-k > 3 else 0))
            M[i][j] = max(M[i][j-1], temp)
    return M


def traceback(i, j, structure, DP, sequence):
    # in this case we've gone through the whole sequence. Nothing to do.
    if j <= i:
        return
    # if j is unpaired, there will be no change in score when we take it out, so we just recurse to the next index
    elif DP[i][j] == DP[i][j-1]:
        traceback(i, j-1, structure, DP, sequence)
    # consider cases where j forms a pair.
    else:
        # try pairing j with a matching index k to its left.
        for k in range(i, j-3):
            if check_possible_pairing_no_writing(sequence[k], sequence[j]) == 0:
                continue
            # if the score at i,j is the result of adding 1 from pairing (j,k) and whatever score
            # comes from the substructure to its left (i, k-1) and to its right (k+1, j-1)
            if k-1 < 0:
                if DP[i][j] == DP[k+1][j-1] + 1:
                    structure.append((k, j))
                    traceback(k+1, j-1, structure, DP, sequence)
                    break
            elif DP[i][j] == DP[i][k-1] + DP[k+1][j-1] + 1:
                # add the pair (j,k) to our list of pairs
                structure.append((k, j))
                # move the recursion to the two substructures formed by this pairing
                traceback(i, k-1, structure, DP, sequence)
                traceback(k+1, j-1, structure, DP, sequence)
                break


def reconstruct_sequence(bonds, binary_sequence):
    global MAX_BOND
    MAX_BOND = max(MAX_BOND, len(bonds))
    sequence = ['' for _ in range(len(binary_sequence))]
    for idx, bin in enumerate(binary_sequence):
        if idx % 3 != 2:
            #print(DecodingTable[bin], idx)
            if len(DecodingTable[bin]) == 1:
                sequence[idx] = DecodingTable[bin][0]
    for bond in bonds:
        first = DecodingTable[binary_sequence[bond[0]]]
        second = DecodingTable[binary_sequence[bond[1]]]
        if 'c' in first and 'g' in second:
            sequence[bond[0]] = 'c'
            sequence[bond[1]] = 'g'
            continue
        if 'g' in first and 'c' in second:
            sequence[bond[0]] = 'g'
            sequence[bond[1]] = 'c'
            continue
        if 'a' in first and 'u' in second:
            sequence[bond[0]] = 'a'
            sequence[bond[1]] = 'u'
            continue
        if 'u' in first and 'a' in second:
            sequence[bond[0]] = 'u'
            sequence[bond[1]] = 'a'
            continue
    for idx, base in enumerate(sequence):
        if base == '':
            if 'g' in DecodingTable[binary_sequence[idx]]:
                sequence[idx] = 'g'
                continue
            if 'c' in DecodingTable[binary_sequence[idx]]:
                sequence[idx] = 'c'
                continue
            if 'u' in DecodingTable[binary_sequence[idx]]:
                sequence[idx] = 'u'
                continue
            assert('a' in DecodingTable[binary_sequence[idx]])
            sequence[idx] = 'a'
    return ''.join(sequence)

def read_fasta(fasta_file: str) -> pd.DataFrame:
    """
    Read fasta file using pandas
    :param: filename
    :return: pandas DataFrame with target proteins
    """
    df = pd.read_csv(fasta_file, header=None, comment='>', engine='python',
                     names=['CDS'])

    df['CDS'] = df['CDS'].apply(lambda x: x[:len(x) - len(x) % 3])

    df['Length'] = df['CDS'].str.len()
    print(fasta_file, ('has %d protein(s)' % df.shape[0]))

    return df


def find_bad_amino_positions(aminos):
    positions = []
    for idx, amino in enumerate(aminos):
        if amino in ['L', 'R', 'S', '#']:
            positions.append(idx)
    return positions


def retrieve_opt_seq(binary_sequence, matrix):
    pairs = []
    traceback(0, len(binary_sequence)-1, pairs, matrix, binary_sequence)
    opt_seq = reconstruct_sequence(pairs, binary_sequence)
    return opt_seq


class Node:
    def __init__(self, id, s, ub, lb, cbp, level):
        self.id = id
        self.seq = s
        self.ub = ub
        self.lb = lb
        self.cbp = cbp
        self.level = level

    def __lt__(self, other):
        if self.ub == other.ub:
            return self.level > other.level
        return self.ub > other.ub


def python_frrna(seq, seq_len, q):
    return four_russians.four_russians(seq, seq_len, q)[0]


def get_rna_seq(bin_sequence):
    M = basic_dp(bin_sequence)
    pairs = []
    traceback(0, len(bin_sequence)-1, pairs, M, bin_sequence)
    opt_seq = reconstruct_sequence(pairs, bin_sequence)
    return opt_seq


def iterative_dp(sequence):

    graph = nx.Graph()
    input_aminos = getAminos(sequence)

    bad_amino_pos = find_bad_amino_positions(input_aminos)
    binary_sequence = make_binary(sequence, StandardBinaryAminoTable)

    lb_binary_sequence = make_binary(sequence, AllBinaryAminoTable)
    q = int(math.log(len(sequence), 2)) # log base 2 of CDS length
    arg1 = [binary_sequence, len(binary_sequence), q]
    arg2 = [lb_binary_sequence, len(lb_binary_sequence), q]

    bin_seqs = [make_binary(sequence, StandardBinaryAminoTable) for _ in range(NUM_THREAD)]

    minim = 0
    args = []
    best_matrix = None
    best_bin_seq = ''
    for irng in range(1):
        args = []
        print("Starting iter:", irng, "with minim = ", minim)
        for ind in range(NUM_THREAD):
            for pos in bad_amino_pos:
                sel = random.randint(0, 1)
                for idx, binary in enumerate(Branches[input_aminos[pos]][sel]):
                    assert (sequence[pos * 3:pos * 3 + 3] in InvTableDNA[input_aminos[pos]])

                    bin_seqs[ind][pos * 3 + idx] = binary
            args.append([bin_seqs[ind], len(bin_seqs[ind]), q])
        par_time = time.time()
        p = mp.Pool(NUM_THREAD)
        #print ('-'*40)
        matrices = p.starmap(python_frrna, args)
        p.close()
        p.terminate()
        print('Parallel computation in', time.time() - par_time)
        par_time = time.time()
        python_frrna(binary_sequence, len(binary_sequence), q)
        print('Sequential computation in', time.time() - par_time)

        for index, ms in enumerate(matrices):
            if minim < ms[0][-1]:
                print("New max: from", minim, " to ", ms[0][-1])
                minim = ms[0][-1]
                best_matrix = ms
                best_bin_seq = bin_seqs[index]
    print("Minimum:", minim)
    bandbtime = time.time()
    p = mp.Pool(NUM_THREAD)
    root_matrix, lb_root_matrix = p.starmap(python_frrna, [arg1, arg2])
    p.terminate()
    p.close()

    lb_binary_retrieve = lb_binary_sequence
    lb_root_seq = retrieve_opt_seq(lb_binary_sequence, lb_root_matrix)
    root_opt_seq = retrieve_opt_seq(binary_sequence, root_matrix)
    opt_seq = lb_root_seq
    binary_opt_seq = lb_binary_retrieve
    root_ub = root_matrix[0][-1]
    root_lb = lb_root_matrix[0][-1]
    if root_lb < minim:
        root_lb = minim
        lb_root_seq = retrieve_opt_seq(best_bin_seq, best_matrix)
        lb_binary_retrieve = best_bin_seq
    if root_lb == root_ub:
        return lb_root_seq, root_ub, 0, 0, 0, 0, 0, 0, lb_binary_retrieve
    visited_config = set()
    queue = []
    graph.add_node(0, COLOR="green")
    for pos, (a1, a2) in enumerate(zip(input_aminos, getAminos(root_opt_seq))):
        if a1 != a2:
            assert(pos in bad_amino_pos)
            cbp = [p for p in bad_amino_pos if p != pos]

            branch_sequences = [
                copy.copy(binary_sequence), copy.copy(binary_sequence)]
            for i, binary_branches in enumerate(Branches[input_aminos[pos]]):
                for idx, binary in enumerate(binary_branches):
                    branch_sequences[i][pos*3+idx] = binary
            for branch in branch_sequences:
                queue.append(Node(graph.number_of_nodes(), branch, root_ub, root_lb, cbp, 1))
                child_id = graph.number_of_nodes()
                graph.add_node(graph.number_of_nodes(), UB=root_ub, LB=root_lb, COLOR="green")
                graph.add_edge(0, child_id)
            break
    if len(queue) == 0:
        return root_opt_seq, root_ub, 0, 0, 0, 0, 0, 0, lb_binary_retrieve
    heapq.heapify(queue)
    best_ub = 0
    best_lb = root_lb
    node_extraction_count = 0
    prunings_1 = 0
    prunings_2 = 0
    last_branched_node = 0
    ub_queue = queue[0].ub
    while len(queue) > 0 and time.time() - bandbtime <= 3600:

        node = heapq.heappop(queue)
        node_extraction_count += 1
        ub_queue = node.ub
        if int(node.ub) <= int(best_lb):  # check
            ub = int(best_lb)
            break

        branch_lb_seq = copy.copy(node.seq)
        bin_seqs = [branch_lb_seq for _ in range(NUM_THREAD - 1)]
        minim = 0
        args = [[node.seq, len(node.seq), q]]
        best_matrix = None
        best_bin_seq = ''
        for ind in range(NUM_THREAD - 1):
            for pos in node.cbp:
                sel = 0 if random.random() < 0.5 else 1
                for idx, binary in enumerate(Branches[input_aminos[pos]][sel]):
                    assert (sequence[pos * 3:pos * 3 + 3] in InvTableDNA[input_aminos[pos]])
                    bin_seqs[ind][pos * 3 + idx] = binary

            args.append([bin_seqs[ind], len(bin_seqs[ind]), q])

        p = mp.Pool(NUM_THREAD)
        matrices = p.starmap(python_frrna, args)
        p.close()
        p.terminate()
        branch_lb_matrix = None
        for index, ms in enumerate(matrices[1:]):
            if minim < ms[0][-1]:
                minim = ms[0][-1]
                branch_lb_matrix = ms
                branch_lb_seq = bin_seqs[index]
        branch_matrix = matrices[0]

        if branch_lb_matrix[0][-1] > best_lb:
            best_lb = branch_lb_matrix[0][-1]
            opt_seq = retrieve_opt_seq(branch_lb_seq, branch_lb_matrix)
            binary_opt_seq = branch_lb_seq
        del branch_lb_seq

        ub = branch_matrix[0][-1]

        if ub <= best_lb:
            prunings_1 += 1
            continue
        
        branch_opt_seq = retrieve_opt_seq(node.seq, branch_matrix)
        current_aminos = getAminos(branch_opt_seq)

        bad_amino_flag = True
        for pos, (a1, a2) in enumerate(zip(input_aminos, current_aminos)):
            if a1 != a2:
                bad_amino_flag = False
                assert(pos in node.cbp)
                cbp = [p for p in node.cbp if p != pos]
                branch_sequences = [copy.copy(node.seq), copy.copy(node.seq)]
                for i, binary_branches in enumerate(Branches[input_aminos[pos]]):
                    for idx, binary in enumerate(binary_branches):
                        branch_sequences[i][pos*3+idx] = binary
                assert(branch_sequences[0] != branch_sequences[1])
                assert(len(branch_sequences) == 2)
                for branch in branch_sequences:
                    child_id = graph.number_of_nodes()
                    heapq.heappush(queue, Node(child_id,
                        branch, ub, best_lb, cbp, node.level + 1))
                    graph.add_node(child_id, UB=ub, LB=best_lb, COLOR="green")
                    graph.add_edge(node.id, child_id)

                break
        del branch_matrix
        if bad_amino_flag:  # check
            if best_lb <= ub:
                best_lb = ub
                opt_seq = branch_opt_seq
                best_ub = max(best_ub, ub)
                last_branched_node = node.id
                binary_opt_seq = node.seq

            best_ub = max(best_ub, ub)

        del branch_opt_seq
        print("%10d | %10d | %4d | %9.2f | %9.2f | %9.2f | %9.2f| %8d" %
              (node_extraction_count, len(queue), node.level, root_ub, node.ub,
               best_lb, ub, len(bad_amino_pos)))
    graph.nodes[last_branched_node]["COLOR"] = "red"

    nx.write_gml(graph, 'enumeration_tree.gml')
    return opt_seq, best_lb, best_ub, ub_queue, node_extraction_count, prunings_1, prunings_2, time.time()-bandbtime, \
           binary_opt_seq



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Input file',
                        default='branchandbound')
    parser.add_argument('--output', '-o', help='Output file',
                        default='branchandbound')
    args = parser.parse_args()

    dataset = read_fasta(args.input)
    rep = 10
    random.seed(2020)
    for idx, row in dataset.iterrows():
            sequence = row['CDS']
            amino_sequence = getAminos(sequence)
            bad_amino_pos = find_bad_amino_positions(amino_sequence)
            dataset.loc[idx, 'Protein'] = amino_sequence
            dataset.loc[idx, 'BadAminoCount'] = len(bad_amino_pos)
            dataset.loc[idx, 'EnumerationSize'] = int(
                math.pow(2, len(bad_amino_pos)))
            sequence = getRandomSyn(amino_sequence)
            exec_time = time.time()

            opt_seq, sol, lower_ub, ub_queue, nec, prunings_1, prunings_2, bandbtime, boptseq = iterative_dp(sequence)
            end = time.time()
            test = get_rna_seq(boptseq)

            dataset.loc[idx, 'BestLB'] = sol
            dataset.loc[idx, 'LowerUB'] = lower_ub
            dataset.loc[idx, 'QueueUB'] = ub_queue
            dataset.loc[idx, 'Nodes'] = nec
            dataset.loc[idx, 'OPTCDS'] = opt_seq
            dataset.loc[idx, 'Prune1'] = prunings_1
            dataset.loc[idx, 'Prune2'] = prunings_2
            dataset.loc[idx, 'Time'] = end-exec_time
            dataset.loc[idx, 'BranchBoundTime'] = bandbtime
            dataset.loc[idx, 'Reconstructed'] = test
            dataset.loc[idx, 'Correctness'] = getAminos(opt_seq) == amino_sequence
            dataset.loc[idx, 'EnumerationFraction'] = nec / int(math.pow(2, len(bad_amino_pos)))
            idx += 1
            dataset.to_csv(args.output + '.csv', index=False)
