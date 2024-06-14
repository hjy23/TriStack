import pandas as pd

# 载入Excel
properties_file_path = 'properties_of_amino_acids.xlsx'
amino_acid_properties = pd.read_excel(properties_file_path, engine='openpyxl')
# 重命名第一列，以便更容易操作数据
amino_acid_properties = amino_acid_properties.rename(columns={'Unnamed: 0': 'AminoAcid'})
# 修正数据处理步骤，确保所有条目都是字符串类型
amino_acid_properties['AminoAcid'] = amino_acid_properties['AminoAcid'].astype(str).apply(
    lambda x: x.split('(')[0].strip())

# 重新构建氨基酸属性字典，考虑到可能的数据格式问题
aa_props_dict = amino_acid_properties.set_index('AminoAcid').to_dict(orient='index')


def calculate_amino_acid_properties(sequence: str) -> list:
    """
        :param sequence:蛋白质序列，如"ADDGHKKA"
        :return amino_acid_properties:新8种理化特征
        计算给定蛋白质序列的新8种理化特征
    """
    blam = sum(aa_props_dict[aa]['BLAM930101'] for aa in sequence if aa in aa_props_dict)
    biov = sum(aa_props_dict[aa]['BIOV880101'] for aa in sequence if aa in aa_props_dict)
    maxf = sum(aa_props_dict[aa]['MAXF760101(α螺旋 二级结构重复）'] for aa in sequence if aa in aa_props_dict)
    tsaj = sum(aa_props_dict[aa]['TSAJ990101'] for aa in sequence if aa in aa_props_dict)
    nakh = sum(aa_props_dict[aa]['NAKH920108'] for aa in sequence if aa in aa_props_dict)
    cedj = sum(aa_props_dict[aa]['CEDJ970104'] for aa in sequence if aa in aa_props_dict)
    lifs = sum(aa_props_dict[aa]['LIFS790101'] for aa in sequence if aa in aa_props_dict)
    miys = sum(aa_props_dict[aa]['MIYS990104'] for aa in sequence if aa in aa_props_dict)
    blam_mean = sum(aa_props_dict[aa]['BLAM930101'] for aa in sequence if aa in aa_props_dict) / len(sequence)
    biov_mean = sum(aa_props_dict[aa]['BIOV880101'] for aa in sequence if aa in aa_props_dict) / len(sequence)
    maxf_mean = sum(
        aa_props_dict[aa]['MAXF760101(α螺旋 二级结构重复）'] for aa in sequence if aa in aa_props_dict) / len(sequence)
    tsaj_mean = sum(aa_props_dict[aa]['TSAJ990101'] for aa in sequence if aa in aa_props_dict) / len(sequence)
    nakh_mean = sum(aa_props_dict[aa]['NAKH920108'] for aa in sequence if aa in aa_props_dict) / len(sequence)
    cedj_mean = sum(aa_props_dict[aa]['CEDJ970104'] for aa in sequence if aa in aa_props_dict) / len(sequence)
    lifs_mean = sum(aa_props_dict[aa]['LIFS790101'] for aa in sequence if aa in aa_props_dict) / len(sequence)
    miys_mean = sum(aa_props_dict[aa]['MIYS990104'] for aa in sequence if aa in aa_props_dict) / len(sequence)
    return [blam, tsaj, nakh, cedj, biov, maxf, lifs, miys, blam_mean, biov_mean, maxf_mean, cedj_mean, tsaj_mean,
            nakh_mean, lifs_mean, miys_mean]


def calculate_hydrophobicity(sequence: str) -> list:
    """
        :param sequence: 蛋白质序列，如"ADDGHKKA"
        :return: 疏水性特征
        计算给定蛋白质序列的疏水性特征
    """
    total_hydrophobicity = sum(aa_props_dict[aa]['疏水性(hydrophobic)'] for aa in sequence if aa in aa_props_dict)
    return [total_hydrophobicity / len(sequence), total_hydrophobicity]


def calculate_charge(sequence: str) -> list:
    """
        :param sequence: 蛋白质序列，如"ADDGHKKA"
        :return: 电荷特征
        计算给定蛋白质序列的电荷特征
    """
    total_charge = sum(aa_props_dict[aa]['净电荷（net charge)'] for aa in sequence if aa in aa_props_dict)
    return [total_charge / len(sequence), total_charge]


def calculate_polarity(sequence: str) -> list:
    """
        :param sequence: 蛋白质序列，如"ADDGHKKA"
        :return: 极性特征
        计算给定蛋白质序列的电荷特征
    """
    total_polarity = sum(aa_props_dict[aa]['极性      (polarity)'] for aa in sequence if aa in aa_props_dict)
    return [total_polarity / len(sequence), total_polarity]


def calculate_polarizability(sequence: str) -> list:
    """
        :param sequence: 蛋白质序列，如"ADDGHKKA"
        :return: 极性率特征
        计算给定蛋白质序列的极性率特征
    """
    total_polarizability = sum(aa_props_dict[aa]['极化率   (polarizability)'] for aa in sequence if aa in aa_props_dict)
    return [total_polarizability / len(sequence), total_polarizability]


def calculate_alpha_helix_propensity(sequence: str) -> list:
    """
        :param sequence: 蛋白质序列，如"ADDGHKKA"
        :return: α螺旋特征
        计算给定蛋白质序列的α螺旋特征
    """
    total_alpha_helix = sum(
        aa_props_dict[aa]['MAXF760101(α螺旋 二级结构重复）'] for aa in sequence if aa in aa_props_dict)
    return [total_alpha_helix / len(sequence), total_alpha_helix]


def calculate_beta_sheet_propensity(sequence: str) -> list:
    """
        :param sequence: 蛋白质序列，如"ADDGHKKA"
        :return: β-折叠特征
        计算给定蛋白质序列的β-折叠特征
    """
    total_beta_sheet = sum(
        aa_props_dict[aa]['β-折叠（beta-sheet）二级结构另一部分'] for aa in sequence if aa in aa_props_dict)
    return [total_beta_sheet / len(sequence), total_beta_sheet]


def calculate_volume(sequence):
    """
        :param sequence: 蛋白质序列，如"ADDGHKKA"
        :return: 原子数特征
        计算给定蛋白质序列的原子数特征
        * 该特征已经不再使用
    """
    total_volume = sum(aa_props_dict[aa]['原子数'] for aa in sequence if aa in aa_props_dict)
    return [total_volume, total_volume / len(sequence)]
