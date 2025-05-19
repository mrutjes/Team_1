import torch
from torch_geometric.data import Data

from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
from helper.constants import ALLOWED_ATOMS, ELECTRONEGATIVITY, BOND_ORDER_MAP, BOND_TYPE_ENCODING, ALLOWED_HYBRIDIZATIONS
from functions.data_loader import data_loader

class MolecularGraphFromSMILES:
    """
    Converts a SMILES string into a graph representation suitable for GNN models,
    with annotated borylation site and computed yield value.
    """

    def __init__(self, smiles: str):
        """
        Initializes the object with a SMILES string, adds atom mapping, 
        loads yield and borylation data, and builds the RDKit molecule.
        """

        self.smiles = smiles
        self.smiles_new = self.add_atom_mapping_to_smiles()

        self.df_merged = data_loader("data/compounds_yield.csv", "data/compounds_smiles.csv")
        self.df_idx = self.df_merged.index[self.df_merged['smiles_raw'] == smiles]
        self.df_index = int(self.df_idx[0])
        self.borylation_index = self.df_merged['borylation_site'].iloc[self.df_index]
        self.yield_value = float(self.df_merged['yield'].iloc[0])
        self.borylation_index = int(self.borylation_index)

        self.mol = Chem.MolFromSmiles(self.smiles_new)

        self.atoms = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        self.atom_objects = [atom for atom in self.mol.GetAtoms()]
        self.bond_objects = [bond for bond in self.mol.GetBonds()]


    def add_atom_mapping_to_smiles(self) -> str:
        """
        Adds atom mapping numbers to each atom in the SMILES 
        string to match SMILES and RDKit indices, to make sure these align.
        """

        mol = Chem.MolFromSmiles(self.smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        
        smiles_with_map = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=False)

        return smiles_with_map
    
    
    def get_smiles_to_graph_index_map(self) -> dict:
        """
        Returns a mapping from SMILES atom map numbers to RDKit atom indices.
        """
        mapping = {}
        for atom in self.mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num >= 0:
                mapping[map_num] = atom.GetIdx()

        return mapping


    def _one_hot(self, value, choices):
        """
        Returns a one-hot encoded vector for a given value from a list of choices.
        """

        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding


    def to_pyg_data(self) -> Data:
        """    
        Converts the molecule to a PyTorch Geometric Data object with atom and bond features, 
        including a borylation mask and yield label.
        """

        mapping = self.get_smiles_to_graph_index_map()
        graph_index = mapping[self.borylation_index]

        x = []
        for i, atom in enumerate(self.atom_objects):
            symbol = atom.GetSymbol()
            one_hot_symbol = self._one_hot(symbol, ALLOWED_ATOMS)
            one_hot_aromatic = [int(atom.GetIsAromatic()), int(not atom.GetIsAromatic())]
            is_borylation_site = [1] if i == graph_index else [0]

            hybrid = atom.GetHybridization()
            one_hot_hybrid = self._one_hot(hybrid, ALLOWED_HYBRIDIZATIONS)

            valence = atom.GetTotalValence()
            num_Hs = atom.GetTotalNumHs()
            mass = atom.GetMass() / 200  # normaal tussen 0–1
            degree = atom.GetDegree()

            feature_vector = (
                one_hot_symbol +
                one_hot_aromatic +
                [atom.GetFormalCharge()] +
                [int(atom.IsInRing())] +
                [ELECTRONEGATIVITY.get(symbol, 0.0)] +
                is_borylation_site +
                one_hot_hybrid +
                [valence, num_Hs, mass, degree]
            )
            x.append(feature_vector)

        x = torch.tensor(x, dtype=torch.float)

        edge_index = []
        edge_attr = []
        for bond in self.bond_objects:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            sym_i = self.atoms[i]
            sym_j = self.atoms[j]

            diff = abs(ELECTRONEGATIVITY.get(sym_i, 0) - ELECTRONEGATIVITY.get(sym_j, 0))
            if diff > 1.7:
                bond_type = "ionic"
            elif diff > 0.4:
                bond_type = "polar"
            else:
                bond_type = "covalent"

            attr = [
                BOND_ORDER_MAP.get(bond.GetBondType(), 1),
                int(bond.GetIsAromatic()),
                BOND_TYPE_ENCODING[bond_type]
            ]

            edge_index += [[i, j], [j, i]]
            edge_attr += [attr, attr]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        borylation_mask = torch.zeros(len(self.atoms))
        borylation_mask[graph_index] = 1.0

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([self.yield_value], dtype=torch.float),
            borylation_mask=borylation_mask
        )

        return data
    

    def visualize(self, with_labels=True):
        """
        Visualizes the molecular graph using NetworkX with atoms as nodes and bonds as edges.
        """

        G = nx.Graph()
        for i, el in enumerate(self.atoms):
            G.add_node(i, label=el)
        for bond in self.bond_objects:
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            order = BOND_ORDER_MAP.get(bond.GetBondType(), 1)
            G.add_edge(i, j, label=str(order))

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=with_labels,
                labels=nx.get_node_attributes(G, 'label'),
                node_color='lightblue', node_size=700, font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
