import torch


# vertices: B x n_frames x n_vertices x 3
# edge_indices: n_edges x 2
def compute_edge_lengths(vertices, edge_indices):
    edges_0 = vertices[:, :, [x for x, _ in edge_indices]]
    edges_1 = vertices[:, :, [x for _, x in edge_indices]]

    edge_lengths = torch.norm(
        edges_0 * 1000 - edges_1 * 1000,
        dim=3,
    )

    return edge_lengths
