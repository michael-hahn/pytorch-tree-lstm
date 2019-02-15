import torch


def _label_node_walk_order(node, n=0):
    node['index'] = n
    for child in node['children']:
        n = n + 1
        _label_node_walk_order(child, n)


def _gather_features(node):
    features = [node['features']]
    for child in node['children']:
        features.extend([child['features']])
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list


def _find_order(node):
    order = 0
    for child in node['children']:
        order = max(order, _find_order(child) + 1)
    return order


def _gather_node_evaluation_order(node):
    node_evaluation_order = [_find_order(node)]
    for child in node['children']:
        node_evaluation_order.extend(_gather_node_evaluation_order(child))
    return node_evaluation_order


def _gather_edge_evaluation_order(node):
    adjacency_list = []
    for child in node['children']:
        # edges should be evaluated in the same order as their parent node
        adjacency_list.append(_find_order(node))
        adjacency_list.extend(_gather_edge_evaluation_order(child))
    return adjacency_list


def convert_tree_to_tensors(tree, device):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_walk_order(tree)

    features = _gather_features(tree)
    node_evaluation_order = _gather_node_evaluation_order(tree)
    adjacency_list = _gather_adjacency_list(tree)
    edge_evaluation_order = _gather_edge_evaluation_order(tree)

    return {
        'features': torch.tensor(features, device=device, dtype=torch.float32),
        'node_evaluation_order': torch.tensor(node_evaluation_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_evaluation_order': torch.tensor(edge_evaluation_order, device=device, dtype=torch.int64),
    }


if __name__ == '__main__':
    tree = {
        'features': [1, 0, 0, 0],
        'children': [
            {'features': [0, 1, 0, 0], 'children': []},
            {'features': [0, 0, 1, 0], 'children': [
                {'features': [0, 0, 0, 1], 'children': []}
            ]},
        ],
    }

    result = convert_tree_to_tensors(tree, torch.device('cpu'))

    for k, v in result.items():
        print(f'{k}:\n{v}\n')
