import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import softmax

class MPNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats,
                 num_step_message_passing, num_step_set2set, num_layer_set2set,
                 readout_feats, activation, dropout):
        super(MPNN, self).__init__()

        # Activatiefunctie instellen
        self.activation_fn = self._get_activation_fn(activation)
        self.dropout = dropout

        # Projecteer node features naar hidden features
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats),
            self.activation_fn  # Activatiefunctie als object
        )

        self.num_step_message_passing = num_step_message_passing

        # Edge network
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)

        self.gnn_layer = NNConv(
            in_channels=hidden_feats,
            out_channels=hidden_feats,
            nn=edge_network,
            aggr='add'
        )

        # GRU voor node-updates
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        # Node-level heads (klassificatie en regressie)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_feats, 1),  # Sigmoid komt pas bij loss
        )
        self.node_regressor = nn.Linear(hidden_feats, 1)

        # Graph-level readout
        self.readout = Set2Set(hidden_feats * 2, num_step_set2set, num_layers=num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )

        # Yield regressor
        self.yield_regressor = nn.Sequential(
            nn.Linear(readout_feats, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_activation_fn(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.ReLU())  # Default is ReLU

    def forward(self, x, edge_index, edge_attr, batch):
        # Projecteer node features naar hidden features
        node_feats = self.project_node_feats(x)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation_fn(self.gnn_layer(node_feats, edge_index, edge_attr)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)

        node_aggr.append(node_feats)
        node_aggr_cat = torch.cat(node_aggr, dim=1)

        # Node-level outputs
        p_borylation = self.node_classifier(node_feats).squeeze(-1)

        # Graph-level output
        readout = self.readout(node_aggr_cat, batch)
        graph_feats = self.sparsify(readout)
        predicted_yield = self.yield_regressor(graph_feats).squeeze(-1)

        return p_borylation, predicted_yield
