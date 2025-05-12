import torch
from torch_geometric.data import Data

from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
from helper.constants import ALLOWED_ATOMS, ELECTRONEGATIVITY, BOND_ORDER_MAP, BOND_TYPE_ENCODING

class MolecularGraphFromSMILES:
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.atoms = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        self.atom_objects = [atom for atom in self.mol.GetAtoms()]
        self.bond_objects = [bond for bond in self.mol.GetBonds()]

    def _one_hot(self, value, choices):
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding

    def to_pyg_data(self) -> Data:
        x = []
        for atom in self.atom_objects:
            symbol = atom.GetSymbol()
            one_hot_symbol = self._one_hot(symbol, ALLOWED_ATOMS)
            one_hot_aromatic = [int(atom.GetIsAromatic()), int(not atom.GetIsAromatic())]
            feature_vector = (
                one_hot_symbol +
                one_hot_aromatic +
                [atom.GetFormalCharge()] +
                [int(atom.IsInRing())] +
                [ELECTRONEGATIVITY.get(symbol, 0.0)]
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

            # electronegativiteitsschatting voor bindingstype
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

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def visualize(self, with_labels=True):
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
