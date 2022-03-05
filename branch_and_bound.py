import RNA
import random
import math
import numpy as np
import pandas as pd
import heapq
import copy
import gurobipy as gp
import time
import argparse
import multiprocessing as mp
import threading as th
import sys
sys.path.append('./cpp')
import nussinov


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


def cai(protein):
    return math.pow(math.e, sum(Costs[protein[i * 3:i * 3 + 3]] for i in range(len(protein) // 3)) / (
        len(protein) // 3))


def vienna_aup(seq):
    fc = RNA.fold_compound(seq)
    (propensity, ensemble_energy) = fc.pf()
    basepair_probs = fc.bpp()
    x = np.array(basepair_probs)
    t = np.triu(x).T + np.tril(x)
    e = t + x
    return np.mean(1 - np.sum(e, axis=0))


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


def count_not_good_aminos(aminos):
    amminoacidi_cattivi = ['L', 'R', 'S', '#']
    count = 0
    for amino in aminos:
        if amino in amminoacidi_cattivi:
            count += 1
    return count


amminoacidi_buoni = ['A', 'K', 'N', 'M', 'D', 'F', 'C',
                     'P', 'Q', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
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

FrequencyNorm = {'TTT': 0.87, 'TTC': 1.0, 'TTA': 0.19, 'TTG': 0.33, 'CTT': 0.33, 'CTC': 0.49, 'CTA': 0.18, 'CTG': 1.0,
                 'ATT': 0.77, 'ATC': 1.0, 'ATA': 0.36, 'ATG': 1.0, 'GTT': 0.39, 'GTC': 0.51, 'GTA': 0.25, 'GTG': 1.0,
                 'TCT': 0.78, 'TCC': 0.91, 'TCA': 0.63, 'TCG': 0.23, 'CCT': 0.89, 'CCC': 1.0, 'CCA': 0.86, 'CCG': 0.35,
                 'ACT': 0.69, 'ACC': 1.0, 'ACA': 0.8, 'ACG': 0.32, 'GCT': 0.67, 'GCC': 1.0, 'GCA': 0.57, 'GCG': 0.27,
                 'TAT': 0.8, 'TAC': 1.0, 'TAA': 0.64, 'TAG': 0.51, 'CAT': 0.72, 'CAC': 1.0, 'CAA': 0.36, 'CAG': 1.0,
                 'AAT': 0.89, 'AAC': 1.0, 'AAA': 0.77, 'AAG': 1.0, 'GAT': 0.87, 'GAC': 1.0, 'GAA': 0.73, 'GAG': 1.0,
                 'TGT': 0.84, 'TGC': 1.0, 'TGA': 1.0, 'TGG': 1.0, 'CGT': 0.37, 'CGC': 0.86, 'CGA': 0.51, 'CGG': 0.94,
                 'AGT': 0.62, 'AGC': 1.0, 'AGA': 1.0, 'AGG': 0.98, 'GGT': 0.48, 'GGC': 1.0, 'GGA': 0.74, 'GGG': 0.74}

Costs = {k: math.log(FrequencyNorm[k]) for k in FrequencyNorm.keys()}
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
StBinaryAminoTable = {
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
    '#': [1, 8, 10]
}
SBinaryAminoTable = {
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
    'S': [1, 4, 15],
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
RBinaryAminoTable = {
    'A': [2, 4, 15],
    'L': [5, 1, 15],
    'R': [4, 2, 15],
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
LBinaryAminoTable = {
    'A': [2, 4, 15],
    'L': [4, 1, 15],
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


def check_possible_pairing(base1, base2, i=0, j=0, pairing_matrix=None):
    # if pairing_matrix is not None and pairing_matrix[i][j] != -1:
    #     return pairing_matrix[i][j]
    pairing_matrix[i][j] = 1 if np.bitwise_and(
        base1, reverse_Bits(base2, 4)) > 0 else 0
    return pairing_matrix[i][j]


def check_possible_pairing_no_writing(base1, base2):
    # if pairing_matrix is not None and pairing_matrix[i][j] != -1:
    #     return pairing_matrix[i][j]
    return 1 if np.bitwise_and(base1, reverse_Bits(base2, 4)) > 0 else 0


def recursive_dp(sequence, i, j, m, check):
    temp = 0
    if j-i <= 3:
        check[i][j] = 1
        return 0
    if check[i][j]:
        return m[i][j]
    for k in range(i, j):
        temp = max(temp, recursive_dp(sequence, i, k-1, m, check)+recursive_dp(sequence, k+1,
                                                                               j-1, m, check) + (check_possible_pairing(sequence[k], sequence[j]) if j-k > 3 else 0))

    m[i][j] = max(recursive_dp(sequence, i, j-1, m, check), temp)
    check[i][j] = 1

    return m[i][j]


def dp(sequence, matrix=None, bad_amino_pos=[], pairing_matrix=None):
    len_seq = len(sequence)
    bad_base_pos = set()
    if matrix is not None:
        M = matrix
    else:
        M = np.zeros((len_seq, len_seq))
    for d in range(4, len_seq):  # [4, len_seq)
        for i in range(len_seq-d):  # [0, len_seq-d)
            j = i+d
            temp = 0
            for k in range(i, j):
                if pairing_matrix[i][j] == -1:
                    temp = max(temp, M[i][k-1]+M[k+1][j-1] + (check_possible_pairing(
                        sequence[k], sequence[j], k, j, pairing_matrix) if j-k > 3 else 0))
                else:
                    temp = max(temp, M[i][k-1]+M[k+1][j-1] +
                               (pairing_matrix[k][j] if j-k > 3 else 0))

            M[i][j] = max(M[i][j-1], temp)
    return M


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


def gusfield_dp(sequence, pairing_matrix=None):
    len_seq = len(sequence)
    M = np.zeros((len_seq, len_seq))

    for j in range(1, len_seq):
        for i in range(j-1):
            if pairing_matrix[i][j] == -1:
                M[i][j] = max(M[i+1][j-1] + (check_possible_pairing(sequence[i],
                                                                    sequence[j], i, j, pairing_matrix) if j-i > 3 else 0), M[i][j-1])
            else:
                M[i][j] = max(M[i+1][j-1] + pairing_matrix[i][j], M[i][j-1])
        for i in range(j-2, -1, -1):
            M[i][j] = max(M[i+1][j], M[i][j])
            for k in range(j-2, i, -1):
                M[i][j] = max(M[i][j], M[i][k-1]+M[k][j])
    return M


def dp_no_writing(sequence, matrix=None, pairing_matrix=None):
    len_seq = len(sequence)
    if matrix is not None:
        M = matrix
    else:
        M = np.zeros((len_seq, len_seq))
    # if matrix is not None:
    #     for p1, p2 in zip(bad_amino_pos, bad_amino_pos[1:]):
    #         M[(p1+1)*3:p2*3, (p1+1)*3:p2*3] = matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3]
    #     M[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3] = matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3]
    #     M[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:] = \
    #         matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:]
    #     for d in range(4, len_seq): # [4, len_seq)
    #         for i in range(len_seq-d): # [0, len_seq-d)
    #             j = i+d
    #             temp = 0
    #             for p in bad_base_pos:
    #                 if p >= i and p <= j:
    #                     for k in range(i, j):
    #                         temp = max(temp, M[i][k-1]+M[k+1][j-1] + (check_possible_pairing(sequence[k], sequence[j], i, j, pairing_matrix) if j-k>3 else 0))
    #                     M[i][j] = max(M[i][j-1], temp)
    #                     break
    # else:
    for d in range(4, len_seq):  # [4, len_seq)
        for i in range(len_seq-d):  # [0, len_seq-d)
            j = i+d
            temp = 0
            for k in range(i, j):
                if pairing_matrix[i][j] == -1:
                    temp = max(temp, M[i][k-1]+M[k+1][j-1] + (
                        check_possible_pairing_no_writing(sequence[k], sequence[j]) if j-k > 3 else 0))
                else:
                    temp = max(temp, M[i][k-1]+M[k+1][j-1] +
                               (pairing_matrix[k][j] if j-k > 3 else 0))

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
        for k in [b for b in range(i, j-3) if check_possible_pairing_no_writing(sequence[b], sequence[j])]:
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
    sequence = ['' for _ in range(len(binary_sequence))]
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
                     names=['Protein'])

    df['Protein'] = df['Protein'].apply(lambda x: x[:len(x) - len(x) % 3])

    df['Length'] = df['Protein'].str.len()
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


def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


class Node:
    def __init__(self, s, ub, lb, cbp, level):
        self.seq = s
        self.ub = ub
        self.lb = lb
        self.cbp = cbp
        self.level = level

    def __lt__(self, other):
        if self.ub == other.ub:
            return self.level < other.level
        return self.ub > other.ub

def python_nussinov(seq, seq_len):
    return nussinov.nussinov(seq, seq_len)

def iterative_dp(sequence):
    input_aminos = getAminos(sequence)
    #pairing_matrix = np.ones((len(sequence), len(sequence))) * -1
    #lb_pairing_matrix = np.ones((len(sequence), len(sequence))) * -1
    bad_amino_pos = find_bad_amino_positions(input_aminos)
    binary_sequence = make_binary(sequence, StandardBinaryAminoTable)

    lb_binary_sequence = make_binary(sequence, AllBinaryAminoTable)
    arg1 = [binary_sequence, len(binary_sequence)]
    arg2 = [lb_binary_sequence, len(lb_binary_sequence)]
    with mp.Pool(2) as p:
        root_matrix, lb_root_matrix = p.starmap(python_nussinov, [arg1, arg2])
    #lb_root_matrix = dp(lb_binary_sequence, pairing_matrix=lb_pairing_matrix)
    #root_matrix = dp(binary_sequence,pairing_matrix=pairing_matrix)

    lb_root_seq = retrieve_opt_seq(lb_binary_sequence, lb_root_matrix)
    root_opt_seq = retrieve_opt_seq(binary_sequence, root_matrix)
    opt_seq = lb_root_seq
    root_ub = root_matrix[0][-1]
    root_lb = lb_root_matrix[0][-1]
    if root_lb == root_ub:
        return lb_root_seq, root_ub, 0, 0, 0
    visited_config = set()
    queue = []
    for pos, (a1, a2) in enumerate(zip(input_aminos, getAminos(root_opt_seq))):
        if a1 != a2:
            assert(pos in bad_amino_pos)
            cbp = [p for p in bad_amino_pos if p != pos]
            visited_config.add(tuple(cbp))
            branch_sequences = [
                copy.copy(binary_sequence), copy.copy(binary_sequence)]
            for i, binary_branches in enumerate(Branches[input_aminos[pos]]):
                for idx, binary in enumerate(binary_branches):
                    branch_sequences[i][pos*3+idx] = binary
            for branch in branch_sequences:
                queue.append(Node(branch, root_ub, root_lb, cbp, 1))
    if len(queue) == 0:
        return root_opt_seq, root_ub, 0, 0, 0
    heapq.heapify(queue)
    best_ub = 0
    best_lb = root_lb
    node_extraction_count = 0
    prunings_1 = 0
    prunings_2 = 0

    # basic_pairing_matrix = copy.copy(pairing_matrix)
    # for a in bad_amino_pos:
    #     for b in range(3):
    #         basic_pairing_matrix[a*3+b,:] = -1
    #         basic_pairing_matrix[:, a*3+b] = -1
    while len(queue) > 0:
        node = heapq.heappop(queue)
        node_extraction_count += 1
        if int(node.ub) <= int(best_lb):  # check
            break
        #branch_matrix = np.zeros((len(sequence), len(sequence)))
        #branch_pairing_matrix = np.ones((len(sequence), len(sequence))) * -1
        # for p1, p2 in zip(bad_amino_pos, bad_amino_pos[1:]):
        #     #branch_matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3] = root_matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3]
        #     branch_pairing_matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3] = pairing_matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3]
        # #branch_matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3] = root_matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3]
        # branch_pairing_matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3] = pairing_matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3]

        # #branch_matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:] = \
        #         #root_matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:]
        # branch_pairing_matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:] = \
        #         pairing_matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:]

        branch_lb_seq = copy.copy(node.seq)
        #branch_lb_matrix = np.zeros((len(sequence), len(sequence)))
        #branch_lb_pairing_matrix = np.ones((len(sequence), len(sequence))) * -1

        # for p1, p2 in zip(bad_amino_pos, bad_amino_pos[1:]):
        #     #branch_lb_matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3] = root_matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3]
        #     branch_lb_pairing_matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3] = lb_pairing_matrix[(p1+1)*3:p2*3, (p1+1)*3:p2*3]
        # #branch_lb_matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3] = root_matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3]
        # branch_lb_pairing_matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3] = lb_pairing_matrix[0:bad_amino_pos[0]*3, 0:bad_amino_pos[0]*3]
        # #branch_lb_matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:] = \
        #         #root_matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:]
        # branch_lb_pairing_matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:] = \
        #         lb_pairing_matrix[(bad_amino_pos[-1]+1)*3:, (bad_amino_pos[-1]+1)*3:]
        for bap in node.cbp:
            for idx, binary in enumerate(Branches[input_aminos[bap]][0]):
                branch_lb_seq[bap*3+idx] = binary

        arg1 = [node.seq, len(node.seq)]
        arg2 = [branch_lb_seq, len(branch_lb_seq)]
        p = mp.Pool()
        branch_matrix, branch_lb_matrix = p.starmap(python_nussinov, [arg1, arg2])
        p.close()
        p.terminate()
        #branch_matrix = dp(node.seq, bad_amino_pos, matrix=branch_matrix, pairing_matrix=branch_pairing_matrix)
        #branch_lb_matrix = dp(branch_lb_seq, matrix=branch_matrix, pairing_matrix=branch_lb_pairing_matrix)
        if branch_lb_matrix[0][-1] > best_lb:
            best_lb = branch_lb_matrix[0][-1]
            opt_seq = retrieve_opt_seq(branch_lb_seq, branch_lb_matrix)
        del branch_lb_seq

        ub = branch_matrix[0][-1]
        print("Node-number:", node_extraction_count, "| Left:", len(queue), "| Node-level:", node.level,
              "| Root-UB:", root_ub, "| Max-UB-queue:", node.ub, "| Best-LB:", best_lb, "| Node-UB:", ub, "| BadAminos:",len(bad_amino_pos))

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
                if tuple(cbp) in visited_config:
                    prunings_2 += 1
                    continue
                visited_config.add(tuple(cbp))
                branch_sequences = [copy.copy(node.seq), copy.copy(node.seq)]
                for i, binary_branches in enumerate(Branches[input_aminos[pos]]):
                    for idx, binary in enumerate(binary_branches):
                        branch_sequences[i][pos*3+idx] = binary
                assert(branch_sequences[0] != branch_sequences[1])
                for branch in branch_sequences:
                    heapq.heappush(queue, Node(
                        branch, ub, best_lb, cbp, node.level + 1))
        del branch_matrix

        if bad_amino_flag:  # check
            if best_lb <= ub:
                best_lb = ub
                opt_seq = branch_opt_seq
                best_ub = max(best_ub, ub)

            best_ub = max(best_ub, ub)

        del branch_opt_seq

    return opt_seq, best_lb, node_extraction_count, prunings_1, prunings_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Input file',
                        default='branchandbound')
    parser.add_argument('--output', '-o', help='Output file',
                        default='branchandbound')
    args = parser.parse_args()
    amminoacidi_cattivi = ['L', 'R', 'S', '#']
    dataset = pd.DataFrame()
    rep = 10
    random.seed(2022)
    for idx1, amino_number in enumerate([67]):
        for idx, idx2 in enumerate(range(10, 21)):
            print(idx)
            amino_sequence = ''
            amino_indexes = random.sample(range(67), idx2)
            for pos in range(amino_number):
                if pos in amino_indexes:
                    amino_sequence += random.choice(amminoacidi_cattivi)
                else:
                    amino_sequence += random.choice(amminoacidi_buoni)
    
            bad_amino_pos = find_bad_amino_positions(amino_sequence)
            dataset.loc[idx, 'Protein'] = amino_sequence
            dataset.loc[idx, 'BadAminoCount'] = len(bad_amino_pos)
            dataset.loc[idx, 'EnumerationSize'] = int(
                math.pow(2, len(bad_amino_pos)))
            dataset.loc[idx, 'SynonymsCount'] = getSynonymNumber(amino_sequence)
            sequence = getRandomSyn(amino_sequence)
            dataset.loc[idx, 'CDS'] = sequence
            dataset.loc[idx, 'Length'] = len(sequence)
       
            exec_time = time.time()
            opt_seq, sol, nec, prunings_1, prunings_2 = iterative_dp(sequence)
            end = time.time()
            dataset.loc[idx, 'OPT'] = sol
            dataset.loc[idx, 'Nodes'] = nec
            dataset.loc[idx, 'OPTCDS'] = opt_seq
            dataset.loc[idx, 'Prune1'] = prunings_1
            dataset.loc[idx, 'Prune2'] = prunings_2
            dataset.loc[idx, 'Time'] = end-exec_time
            dataset.loc[idx, 'AUP'] = vienna_aup(opt_seq)
            dataset.loc[idx, 'AMFE'] = RNA.fold(opt_seq)[1] / len(sequence)
            dataset.loc[idx, 'Correctness'] = getAminos(opt_seq) == amino_sequence
            dataset.loc[idx, 'EnumerationFraction'] = round(
                nec / int(math.pow(2, len(bad_amino_pos))), 1)
            idx += 1
            dataset.to_csv(args.output + '.csv', index=False)
