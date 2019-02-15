import torch


def batch(batch):
    tree_sizes = [b['features'].shape[0] for b in batch]

    batched_features = torch.cat([b['features'] for b in batch])
    batched_node_evaluation_order = torch.cat([b['node_evaluation_order'] for b in batch])
    batched_edge_evaluation_order = torch.cat([b['edge_evaluation_order'] for b in batch])

    batched_adjacency_list = []
    offset = 0
    for n, b in zip(tree_sizes, batch):
        batched_adjacency_list.append(b['adjacency_list'] + offset)
        offset += n
    batched_adjacency_list = torch.cat(batched_adjacency_list)

    return {
        'features': batched_features,
        'node_evaluation_order': batched_node_evaluation_order,
        'edge_evaluation_order': batched_edge_evaluation_order,
        'adjacency_list': batched_adjacency_list,
        'tree_sizes': tree_sizes
    }


def unbatch(tensor, tree_sizes):
    return torch.split(tensor, tree_sizes, dim=0)
