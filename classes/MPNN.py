import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import softmax

class MPNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats=64,
                 num_step_message_passing=3, num_step_set2set=3, num_layer_set2set=1,
                 readout_feats=1024):
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )

        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)

        self.gnn_layer = NNConv(
            in_feats=hidden_feats,
            out_feats=hidden_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )

        self.activation = nn.ReLU()
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        # Node-level heads
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_feats, 1),  # Sigmoid komt pas bij loss
        )
        self.node_regressor = nn.Linear(hidden_feats, 1)

        # Graph-level readout
        self.readout = Set2Set(input_dim=hidden_feats * 2,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )

        self.yield_regressor = nn.Sequential(
            nn.Linear(readout_feats, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Project node features
        node_feats = self.project_node_feats(x)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(node_feats, edge_index, edge_attr)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)

        node_aggr.append(node_feats)
        node_aggr_cat = torch.cat(node_aggr, dim=1)

        # Node-level outputs
        p_borylation = self.node_classifier(node_feats).squeeze(-1)
        reactivity_score = self.node_regressor(node_feats).squeeze(-1)

        # Graph-level output
        readout = self.readout(node_aggr_cat, batch)
        graph_feats = self.sparsify(readout)
        predicted_yield = self.yield_regressor(graph_feats).squeeze(-1)

        return p_borylation, reactivity_score, predicted_yield