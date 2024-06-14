from itertools import product
from math import sqrt

codon_usage = {
    'UUU': 22.3, 'UCU': 17.6, 'UAU': 16.7, 'UGU': 11.3,
    'UUC': 22.8, 'UCC': 15.5, 'UAC': 17.2, 'UGC': 11.4,
    'UUA': 7.6,  'UCA': 11.1, 'UAA': 1.1,  'UGA': 0.9,
    'UUG': 15.3, 'UCG': 2.4,  'UAG': 0.4,  'UGG': 13.0,
    'CUU': 16.4, 'CCU': 16.5, 'CAU': 12.5, 'CGU': 8.7,
    'CUC': 15.6, 'CCC': 11.2, 'CAC': 11.7, 'CGC': 7.7,
    'CUA': 10.0, 'CCA': 16.6, 'CAA': 14.4, 'CGA': 5.3,
    'CUG': 28.6, 'CCG': 3.9,  'CAG': 24.8, 'CGG': 5.5,
    'AUU': 22.8, 'ACU': 14.7, 'AAU': 20.4, 'AGU': 10.8,
    'AUC': 22.9, 'ACC': 17.6, 'AAC': 22.6, 'AGC': 14.8,
    'AUA': 10.9, 'ACA': 16.9, 'AAA': 29.1, 'AGA': 12.9,
    'AUG': 26.6, 'ACG': 4.1,  'AAG': 31.9, 'AGG': 9.2,
    'GUU': 15.5, 'GCU': 23.7, 'GAU': 29.1, 'GGU': 21.0,
    'GUC': 14.4, 'GCC': 20.3, 'GAC': 22.1, 'GGC': 15.6,
    'GUA': 10.4, 'GCA': 16.9, 'GAA': 34.1, 'GGA': 25.6,
    'GUG': 22.0, 'GCG': 3.6,  'GAG': 31.9, 'GGG': 9.5,
}

codon_to_aa = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# Convert codon usage frequencies to probabilities by dividing by 100
codon_usage_prob = {codon: freq / 100 for codon, freq in codon_usage.items()}

# Calculate C_N, which is the sum of all codon usage frequencies except for the stop codons
C_N = sum(freq for codon, freq in codon_usage_prob.items() if codon_to_aa[codon] != '*')

# Now calculate the codon frequencies for each amino acid
# Initialize a dictionary to hold the codon frequencies for each amino acid
aa_to_codon_freq = {aa: [] for aa in set(codon_to_aa.values()) if aa != '*'}

# Sum the codon usage for each amino acid
for codon, aa in codon_to_aa.items():
    if aa != '*':  # We do not consider stop codons ('*') in this calculation
        aa_to_codon_freq[aa].append(codon_usage_prob[codon])

# Calculate the sum of codon frequencies for each amino acid
aa_to_codon_freq_sum = {aa: sum(freqs) for aa, freqs in aa_to_codon_freq.items()}

# We will need the individual codon usage probabilities when calculating T_m and T_v later on
aa_to_codon_usage = {aa: {} for aa in aa_to_codon_freq}

for codon, aa in codon_to_aa.items():
    if aa != '*':
        aa_to_codon_usage[aa][codon] = codon_usage_prob[codon]

C_N, aa_to_codon_freq_sum, aa_to_codon_usage  # Output these values to check if they are correct


# Continuing from where we left off to calculate the Tm and Tv values for all possible dipeptides
aa_to_codons = {}
for codon, aa in codon_to_aa.items():
    if aa not in aa_to_codons:
        aa_to_codons[aa] = []
    aa_to_codons[aa].append(codon)
# Now calculate the Ci values for each amino acid
aa_to_Ci = {aa: sum(codon_usage[codon] for codon in codons) for aa, codons in aa_to_codons.items() if aa != '*'}

# Generate all possible dipeptides from 20 amino acids
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]

CN = 61

# Calculate the Tm for each dipeptide
Tm_values = {}
for dipeptide in dipeptides:
    aa1, aa2 = dipeptide[0], dipeptide[1]
    Ci = aa_to_Ci.get(aa1, 0) / CN
    Cj = aa_to_Ci.get(aa2, 0) / CN
    Tm_values[dipeptide] = Ci * Cj

# Calculate Tv for each dipeptide
Tv_values = {dipeptide: Tm * (1 - Tm) for dipeptide, Tm in Tm_values.items()}

# Now we have Tm and Tv values for all 400 dipeptides, we can use them to calculate the DDE vector
# Since we don't have an actual sequence, we will skip the calculation of Nij and DDE vector for now
Tm_values, Tv_values

def calculate_dipeptide_frequencies(sequence):
    """
    Calculate the frequencies of each dipeptide in a given sequence.
    """
    # Generate all possible dipeptides from 20 amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    dipeptide_freqs = dict.fromkeys(dipeptides, 0)

    # Calculate frequencies
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i + 2]
        if dipeptide in dipeptide_freqs:
            dipeptide_freqs[dipeptide] += 1

    # Normalize frequencies
    total_dipeptides = sum(dipeptide_freqs.values())
    for dipeptide in dipeptide_freqs:
        dipeptide_freqs[dipeptide] /= total_dipeptides

    return dipeptide_freqs


# We now define the function to calculate DDE for a given protein sequence using the previously calculated Tm and Tv values
def calculate_DDE(sequence):
    # Calculate dipeptide frequencies in the given sequence
    dipeptide_freqs = calculate_dipeptide_frequencies(sequence)

    # Length of the sequence for dipeptide calculation is one less than the actual sequence
    L = len(sequence) - 1

    # Calculate the DDE for each dipeptide
    DDE_vector = {}
    for dipeptide in dipeptides:
        Nij = dipeptide_freqs[dipeptide] * (L - 1)  # Get the count of dipeptide in the sequence
        Dc = Nij / (L - 1) if L > 1 else 0  # Dc calculation
        Tm = Tm_values.get(dipeptide, 0)  # Theoretical mean for the dipeptide
        Tv = Tv_values.get(dipeptide, 0)  # Theoretical variance for the dipeptide
        DDE = (Dc - Tm) / sqrt(Tv) if Tv > 0 else 0  # DDE calculation
        DDE_vector[dipeptide] = DDE

    # Convert the DDE vector to a list to represent the 400-dimensional vector
    DDE_list = [DDE_vector[dp] for dp in dipeptides]
    return DDE_list




codon_usage = {
    'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2,
    'I': 3, 'K': 2, 'L': 6, 'M': 1, 'N': 2, 'P': 4, 'Q': 2,
    'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
}

# Total number of codons excluding the stop codons
total_codons = 61  # This is the usual number in standard genetic code excluding stop codons

def calculate_dipeptide_frequencies(sequence):
    """
    :param sequence:多肽序列
    :return dipeptide_freqs: 二肽频数
    计算给定序列的二肽频数
    """
    # 生成20种氨基酸所能组成产生的所有二肽
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    dipeptide_freqs = dict.fromkeys(dipeptides, 0)

    # 计算频数
    for i in range(len(sequence) - 1):
        dipeptide = sequence[i:i + 2]
        if dipeptide in dipeptide_freqs:
            dipeptide_freqs[dipeptide] += 1

    # 计算频数
    for dipeptide in dipeptide_freqs:
        dipeptide_freqs[dipeptide] /= (len(sequence) - 1)

    return dipeptide_freqs

def calculate_dde(sequence):
    """
    :param sequence:多肽序列
    :return DDE: DDE描述符
    计算DDE描述符
    """
    # 计算二肽组成
    dipeptide_freqs = calculate_dipeptide_frequencies(sequence)

    # 计算二肽的理论均值 (Tm) 与方差 (Tv)
    Tm = {}
    Tv = {}
    for dipeptide in dipeptide_freqs:
        Ci = codon_usage[dipeptide[0]]
        Cj = codon_usage[dipeptide[1]]
        Tm[dipeptide] = (Ci / total_codons) * (Cj / total_codons)
        Tv[dipeptide] = Tm[dipeptide] * (1 - Tm[dipeptide]) / (len(sequence) - 1)

    # 计算DDE
    DDE = {}
    for dipeptide in dipeptide_freqs:
        Dc = dipeptide_freqs[dipeptide]
        DDE[dipeptide] = (Dc - Tm[dipeptide]) / (Tv[dipeptide] ** 0.5) if Tv[dipeptide] != 0 else 0

    return DDE


# For demonstration, let's calculate the DDE vector for a dummy sequence
dummy_sequence = "ACDEFGHIKLMNPQRSTVWY" * 20  # A dummy sequence for illustration
DDE_vector = calculate_DDE(dummy_sequence)

def calculate_cksapp_features(sequence, k_range=range(6)):
    """
    :param sequence: 多肽序列
    :param k_range: 间隔取值范围
    :return cksaap_features: K-间隔氨基酸对数(CKSAAP)特征
    计算指定多肽序列的K-间隔氨基酸对数(CKSAAP)特征
    """
    # 生成所有可能二肽
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    pairs = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    # 维护一个保存每一个k的每一对二肽的出现频率字典
    cksaap_features = {}
    for k in k_range:
        for pair in pairs:
            cksaap_features[f'{pair}_k{k}'] = 0

    # 计算给定确定k值时每一对二肽的出现频率
    for k in k_range:
        for i in range(len(sequence) - 1 - k):
            pair = sequence[i] + sequence[i + 1 + k]
            if pair in pairs:
                cksaap_features[f'{pair}_k{k}'] += 1

    # 正则化
    for k in k_range:
        total_pairs = sum(cksaap_features[f'{pair}_k{k}'] for pair in pairs)
        if total_pairs > 0:
            for pair in pairs:
                cksaap_features[f'{pair}_k{k}'] /= total_pairs

    return cksaap_features
