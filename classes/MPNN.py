import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set, global_mean_pool
from torch_geometric.utils import softmax

class MPNN(nn.Module):
    """
    A Message Passing Neural Network (MPNN) for graph-based learning tasks such as
    site prediction (node-level) and yield prediction (graph-level) in molecular graphs.

    ### Architecture
    The architecture is driven by the desired feature representations at different levels
    (nodes and graphs). Node and edge features are projected into a hidden feature space
    that is iteratively updated through message passing (NNConv) and a GRU.

    Node-level predictions (e.g., reactivity scores) are obtained after message passing,
    while graph-level representations are computed by aggregating node features and
    processing them through a separate readout and regression head.

    ### Components
    - Node embedding projection (linear layer + activation)
    - NNConv for edge-conditioned message passing
    - GRU for sequential node feature updates
    - Node-level prediction heads for classification and regression
    - Graph-level readout using `global_mean_pool` and MLP-based regression head

    ### Parameters
    - node_in_feats: Dimensionality of input node features
    - edge_in_feats: Dimensionality of input edge features
    - hidden_feats: Dimensionality of hidden feature space
    - num_step_message_passing: Number of message passing iterations
    - readout_feats: Dimensionality of graph-level readout embedding
    - activation: Activation function (string, e.g. 'relu', 'tanh')
    - dropout: Dropout rate

    ### Returns
    - p_borylation: Node-level predictions of borylation sites (binary classification)
    - predicted_yield: Graph-level predicted yield (single value per graph)
    """
    
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats,
                 num_step_message_passing,
                 readout_feats, activation, dropout):
        super(MPNN, self).__init__()

        self.activation_fn = self._get_activation_fn(activation)
        self.dropout = dropout

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats),
            self.activation_fn
        )

        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)

        self.gnn_layer = NNConv(
            in_channels=hidden_feats,
            out_channels=hidden_feats,
            nn=edge_network,
            aggr='add'
        )

        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_feats, 1),
            nn.Dropout(dropout)
        )
        self.node_regressor = nn.Linear(hidden_feats, 1)


        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 2, readout_feats), nn.PReLU(), nn.Dropout(dropout),
        )

        self.yield_regressor = nn.Sequential(
            nn.Linear(readout_feats, 128), nn.ReLU(),
            nn.Linear(128, 1),
            nn.Dropout(dropout),
        )

    def _get_activation_fn(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(name.lower(), nn.ReLU())


    def forward(self, x, edge_index, edge_attr, batch):
        """
        Executes a forward pass of the MPNN.

        Parameters:
        - x (Tensor): Node features [num_nodes, node_in_feats]
        - edge_index (LongTensor): Edge indices [2, num_edges]
        - edge_attr (Tensor): Edge features [num_edges, edge_in_feats]
        - batch (LongTensor): Batch vector mapping nodes to graphs [num_nodes]

        Returns:
        - p_borylation (Tensor): Node-level prediction scores [num_nodes]
        - predicted_yield (Tensor): Graph-level yield predictions [num_graphs]

        The model performs message passing with NNConv and GRU, followed by node-level classification
        and graph-level readout using global mean pooling and MLP-based regression.
        """

        node_feats = self.project_node_feats(x)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation_fn(self.gnn_layer(node_feats, edge_index, edge_attr)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)

        node_aggr.append(node_feats)
        node_aggr_cat = torch.cat(node_aggr, dim=1)

        p_borylation = self.node_classifier(node_feats).squeeze(-1)

        readout = global_mean_pool(node_aggr_cat, batch)
        graph_feats = self.sparsify(readout)
        predicted_yield = self.yield_regressor(graph_feats).squeeze(-1)

        return p_borylation, predicted_yield
