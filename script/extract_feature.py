import properties
from dde_cksaap import calculate_dde, calculate_cksapp_features
import numpy as np


# Combine DDE and CKSAAP features for all sequences
def extract_features(sequences):
    features = []
    for sequence in sequences:
        dde = calculate_dde(sequence)
        cksaap = calculate_cksapp_features(sequence)
        # dpc = compute_kmer_composition(sequence, 2)
        # tpc = compute_kmer_composition(sequence, 3)
        # 平均疏水性 (Average Hydrophobicity)
        hydrophobicity = properties.calculate_hydrophobicity(sequence)
        # 平均电荷 (Average Charge)
        charge = properties.calculate_charge(sequence)
        # 平均极性 (Average Polarity)
        polarity = properties.calculate_polarity(sequence)
        # 平均极化率 (Average Polarizability)
        polarizability = properties.calculate_polarizability(sequence)
        # 平均α螺旋倾向性 (Average Alpha-Helix Propensity)
        # alpha_helix_propensity = calculate_alpha_helix_propensity(sequence)
        # 平均β-折叠倾向性 (Average Beta-Sheet Propensity)
        beta_sheet_propensity = properties.calculate_beta_sheet_propensity(sequence)
        new_properties = properties.calculate_amino_acid_properties(sequence)
        # 平均体积
        volume = properties.calculate_volume(sequence)
        feature_vector = new_properties + hydrophobicity + charge + polarity + polarizability + volume + beta_sheet_propensity + list(
            dde.values()) + list(cksaap.values())
        features.append(feature_vector)
    return np.array(features)
