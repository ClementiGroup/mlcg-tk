from typing import Dict

embedding_map_fivebead = {
    "ALA": 1,
    "CYS": 2,
    "ASP": 3,
    "GLU": 4,
    "PHE": 5,
    "GLY": 6,
    "HIS": 7,
    "ILE": 8,
    "LYS": 9,
    "LEU": 10,
    "NLE": 10,  # Type Norleucine as Leucine
    "MET": 11,
    "ASN": 12,
    "PRO": 13,
    "GLN": 14,
    "ARG": 15,
    "SER": 16,
    "THR": 17,
    "VAL": 18,
    "TRP": 19,
    "TYR": 20,
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
}

dict_H = {
    'ALA': 'H',  # A
    'CYS': 'H',  # C
    'ASP': 'H',  # D
    'GLU': 'H',  # E
    'PHE': 'H',  # F
    'GLY': 'H',  # G
    'HIS': 'H',  # H
    'ILE': 'H',  # I
    'LYS': 'H',  # K
    'LEU': 'H',  # L
    'NLE': 'H',  # L
    'MET': 'H',  # M
    'ASN': 'H',  # N
    'PRO': 'H',  # P
    'GLN': 'H',  # Q
    'ARG': 'H',  # R
    'SER': 'H',  # S
    'THR': 'H',  # T
    'VAL': 'H',  # V
    'TRP': 'H',  # W
    'TYR': 'H',  # Y
}

dict_HP = {
    'ALA': 'H',  # A
    'CYS': 'H',  # C
    'ASP': 'P',  # D
    'GLU': 'P',  # E
    'PHE': 'H',  # F
    'GLY': 'H',  # G
    'HIS': 'P',  # H
    'ILE': 'H',  # I
    'LYS': 'P',  # K
    'LEU': 'H',  # L
    'NLE': 'H',  # L
    'MET': 'H',  # M
    'ASN': 'P',  # N
    'PRO': 'H',  # P
    'GLN': 'P',  # Q
    'ARG': 'P',  # R
    'SER': 'P',  # S
    'THR': 'P',  # T
    'VAL': 'H',  # V
    'TRP': 'H',  # W
    'TYR': 'P',  # Y
}



dict_HPNX = {
    'ARG': 'P',
    'HIS': 'P',
    'LYS': 'P',
    'ASP': 'N',
    'GLU': 'N',
    'ASN': 'X',
    'CYS': 'X',
    'GLN': 'X',
    'SER': 'X',
    'THR': 'X',
    'TYR': 'X',
    'ALA': 'H',
    'GLY': 'H',
    'ILE': 'H',
    'LEU': 'H',
    'NLE': 'H',
    'MET': 'H',
    'PHE': 'H',
    'PRO': 'H',
    'TRP': 'H',
    'VAL': 'H'
}


dict_hHPNX = {
    'ARG': 'P',
    'HIS': 'P',
    'LYS': 'P',
    'ASP': 'N',
    'GLU': 'N',
    'ASN': 'X',
    'CYS': 'X',
    'GLN': 'X',
    'SER': 'X',
    'THR': 'X',
    'TYR': 'X',
    'ALA': 'h',
    'GLY': 'H',
    'ILE': 'H',
    'LEU': 'H',
    'NLE': 'H',
    'MET': 'H',
    'PHE': 'H',
    'PRO': 'H',
    'TRP': 'H',
    'VAL': 'h'
}

dict_YhHX = {
    'ARG': 'Y',
    'HIS': 'Y',
    'LYS': 'X',
    'ASP': 'X',
    'GLU': 'Y',
    'ASN': 'Y',
    'CYS': 'H',
    'GLN': 'X',
    'SER': 'Y',
    'THR': 'X',
    'TYR': 'Y',
    'ALA': 'h',
    'GLY': 'Y',
    'ILE': 'H',
    'LEU': 'H',
    'NLE': 'H',
    'MET': 'H',
    'PHE': 'H',
    'PRO': 'X',
    'TRP': 'X',
    'VAL': 'h'
}

final_letter_map = {
    'H' : 1,
    'P' : 2,
    'N' : 3,
    'X' : 4,
    'Y' : 5,
    'h' : 6,
}

extra_5bead = {
    "N": 21,
    "CA": 22,
    "C": 23,
    "O": 24,
}

embedding_map_H = { key : final_letter_map[val] for key, val in dict_H.items() }
for key, val in extra_5bead.items():
    embedding_map_H[key] = val

embedding_map_HP = { key : final_letter_map[val] for key, val in dict_HP.items() }
for key, val in extra_5bead.items():
    embedding_map_HP[key] = val

embedding_map_YhHX = { key : final_letter_map[val] for key, val in dict_YhHX.items() }
for key, val in extra_5bead.items():
    embedding_map_YhHX[key] = val

embedding_map_HPNX = { key : final_letter_map[val] for key, val in dict_HPNX.items() }
for key, val in extra_5bead.items():
    embedding_map_HPNX[key] = val

embedding_map_hHPNX = { key : final_letter_map[val] for key, val in dict_hHPNX.items() }
for key, val in extra_5bead.items():
    embedding_map_hHPNX[key] = val


class CGEmbeddingMap(dict):
    """
    General class for defining embedding maps as Dict
    """

    def __init__(self, embedding_map_dict: Dict[str, int]):
        for k, v in embedding_map_dict.items():
            self[k] = v


class CGEmbeddingMapFiveBead(CGEmbeddingMap):
    """
    Five-bead embedding map defined by:
        - N : backbone nitrogen
        - CA : backbone alpha carbon (specialized for glycing)
        - C : backbone carbonyl carbon
        - O : backbone carbonyl oxygen
        - CB : residue-specific beta carbon
    """

    def __init__(self):
        super().__init__(embedding_map_fivebead)


class CGEmbeddingMapCA(CGEmbeddingMap):
    """
    One-bead embedding map defined by:
        - CA : backbone alpha carbon, carrying aminoacid identity
    """

    def __init__(self):
        ca_dict = {key: emb for key, emb in embedding_map_fivebead.items() if emb <= 20}
        super().__init__(ca_dict)

class CGEmbeddingMapH(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_map_H)
class CGEmbeddingMapHP(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_map_HP)

class CGEmbeddingMapYhHX(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_map_YhHX)

class CGEmbeddingMapHPNX(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_map_HPNX)

class CGEmbeddingMaphHPNX(CGEmbeddingMap):
    def __init__(self):
        super().__init__(embedding_map_hHPNX)

all_residues = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]


def embedding_fivebead(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_fivebead[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_fivebead["GLY"]
        else:
            atom_type = embedding_map_fivebead[name]
    elif name == "CB":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_ca(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_fivebead[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_ca_H(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_H[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_ca_HP(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_HP[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_ca_HPNX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_HPNX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_ca_YhHX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_YhHX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_ca_hHPNX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    CA embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name == "CA":
        atom_type = embedding_map_hHPNX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_fivebead_HP(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_HP[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_HP["GLY"]
        else:
            atom_type = embedding_map_HP[name]
    elif name == "CB":
        atom_type = embedding_map_HP[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_fivebead_YhHX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_YhHX[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_YhHX["GLY"]
        else:
            atom_type = embedding_map_YhHX[name]
    elif name == "CB":
        atom_type = embedding_map_YhHX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type

def embedding_fivebead_HPNX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_HPNX[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_HPNX["GLY"]
        else:
            atom_type = embedding_map_HPNX[name]
    elif name == "CB":
        atom_type = embedding_map_HPNX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type


def embedding_fivebead_hHPNX(atom_df):
    """
    Helper function for mapping high-resolution topology to
    5-bead embedding map.
    """
    name, res = atom_df["name"], atom_df["resName"]
    if name in ["N", "C", "O"]:
        atom_type = embedding_map_hHPNX[name]
    elif name == "CA":
        if res == "GLY":
            atom_type = embedding_map_hHPNX["GLY"]
        else:
            atom_type = embedding_map_hHPNX[name]
    elif name == "CB":
        atom_type = embedding_map_hHPNX[res]
    else:
        print(f"Unknown atom name given: {name}")
        atom_type = "NA"
    return atom_type