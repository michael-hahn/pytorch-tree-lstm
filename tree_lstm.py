import torch


# Optimized TreeLSTM implimentation.  It is very fast but hard to read the code.
class TreeLSTM(torch.nn.Module):
    def __init__(self, feature_size, hidden_layer_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_layer_size = hidden_layer_size

        # bias is only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.feature_size, 3 * self.hidden_layer_size)
        self.U_iou = torch.nn.Linear(self.hidden_layer_size, 3 * self.hidden_layer_size, bias=False)

        # f terms are seperate from the iou terms because they involve sums over child nodes
        self.W_f = torch.nn.Linear(self.feature_size, self.hidden_layer_size)
        self.U_f = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size, bias=False)

    def forward(self, features, node_evaluation_order, edge_evaluation_order, edge_offsets):

        # Total number of nodes in every tree in the batch
        batch_size = node_evaluation_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device

        # h and c states for every node in the batch
        # stored as class members for ease of updating without passing h & c to _run_lstm() on each iteration
        self.h = torch.zeros(batch_size, self.hidden_layer_size, device=device)
        self.c = torch.zeros(batch_size, self.hidden_layer_size, device=device)

        # h_sum storage buffer
        self.h_sum = torch.zeros(batch_size, self.hidden_layer_size, device=device)

        # populate the h and c states respecting computation order
        for n in range(node_evaluation_order.max() + 1):
            self._run_lstm(n, node_evaluation_order, edge_evaluation_order, features, edge_offsets)

        Y = self.classifier(self.h)
        if not self.training:
            Y = torch.sigmoid(Y)

        return Y

    def _run_lstm(self, iteration, node_evaluation_order, edge_evaluation_order, features, edge_offsets):
        """
        """
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_evaluation_order is a tensor of size N x 1
        # edge_evaluation_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # edge_offsets is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_evaluation_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_evaluation_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration > 0:
            # edge_offsets is a tensor of size e x 2
            edge_offsets = edge_offsets[edge_mask, :]

            # parent_offsets and child_offsets are tensors of size e x 1
            # parent_offsets and child_offsets contain the integer indexes needed to index into the feature and 
            # hidden state arrays to retrieve the data for those parent/child nodes.
            parent_offsets = edge_offsets[:, 0]
            child_offsets = edge_offsets[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = self.h[child_offsets, :]
            child_c = self.c[child_offsets, :]

            # Add child hidden states to parent offset locations
            self.h_sum[parent_offsets, :] += self.h[child_offsets, :]

        # i, o and u are tensors of size n x M
        iou = self.W_iou(x) + self.U_iou(self.h_sum[node_mask, :])
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        self.c[node_mask, :] = i * u

        # At iteration 0 one of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sun over the child memory cell states
        if iteration > 0:
            # f is a tensor of size e x M
            f = self.W_f(features[parent_offsets, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            self.c[parent_offsets, :] += fc

        self.h[node_mask, :] = o * torch.tanh(self.c[node_mask])
