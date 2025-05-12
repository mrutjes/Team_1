from rdkit import Chem

ALLOWED_ATOMS = ["H", "C", "N", "O", "S", "Br", "F", "Cl", "I", "Si", "B"]

ELECTRONEGATIVITY = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "Br": 2.96,
    "F": 3.98, "Cl": 3.16, "I": 2.66, "Si": 1.90, "B": 2.04
}

BOND_ORDER_MAP = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}

BOND_TYPE_ENCODING = {
    "covalent": 0,
    "polar": 1,
    "ionic": 2
}
