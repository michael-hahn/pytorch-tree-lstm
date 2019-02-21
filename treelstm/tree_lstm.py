import torch


class TreeLSTM(torch.nn.Module):
    r"""Pytorch TreeLSTM model Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.
    For each element in the input sequence, each layer computes the following
    function:
    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}
    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{(t-1)}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.
    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.
    Args:
        in_features: The number of expected features in the input `x`
        out_features: The number of features in the hidden state `h`
    Inputs: features, node_evaluation_order, adjacency_list, edge_evaluation_order
        - **features** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
        - **node_evaluation_order** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
        - **adjacency_list** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.
        - **edge_evaluation_order** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.
    Outputs: h, c
        - **h** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.
          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the cell state for `t = seq_len`.
    Examples::
        >>> model = TreeLSTM(20, 10)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> h, c = model(features, input, (h0, c0))
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, features, node_evaluation_order, adjacency_list, edge_evaluation_order):
        r"""Computes the gradient of current tensor w.r.t. graph leaves.
        The graph is differentiated using the chain rule. If the tensor is
        Arguments:
            features (Tensor): Gradient w.r.t. the
                tensor. If it is a tensor, it will be automatically converted
            node_evaluation_order (Tensor): If ``False``, the graph used to compute
                the grads will be freed. Note that in nearly all cases setting
            adjacency_list (Tensor): If ``False``, the graph used to compute
                the grads will be freed. Note that in nearly all cases setting
            edge_evaluation_order (Tensor): If ``False``, the graph used to compute
                the grads will be freed. Note that in nearly all cases setting
        """

        # Total number of nodes in every tree in the batch
        batch_size = node_evaluation_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        # h_sum storage buffer
        h_sum = torch.zeros(batch_size, self.out_features, device=device)

        # populate the h and c states respecting computation order
        for n in range(node_evaluation_order.max() + 1):
            self._run_lstm(n, h, c, h_sum, features, node_evaluation_order, adjacency_list, edge_evaluation_order)

        return h, c

    def _run_lstm(self, iteration, h, c, h_sum, features, node_evaluation_order, adjacency_list, edge_evaluation_order):
        # Helper function to evaluate all tree nodes for the current iteration.

        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_evaluation_order is a tensor of size N x 1
        # edge_evaluation_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

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
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            h_sum[parent_indexes, :] += h[child_indexes, :]

        # i, o and u are tensors of size n x M
        iou = self.W_iou(x) + self.U_iou(h_sum[node_mask, :])
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        c[node_mask, :] = i * u

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration > 0:
            # f is a tensor of size e x M
            f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            c[parent_indexes, :] += fc

        h[node_mask, :] = o * torch.tanh(c[node_mask])
