from __future__ import annotations

import numpy as np

TArr = np.ndarray


def pad_arrays(arrays: list[TArr]) -> TArr:
    """Pad a list of arrays with zeros along the first dimension.

    All dimensions other than the first must have the same shape across arrays.

    Args:
        arrays: List of numpy arrays to pad.

    Returns:
        np.ndarray: Batched, padded array, shape [B, L, *] where L is length of longest array.
    """

    if len(arrays) == 0:
        return np.array([])

    max_len = max(arr.shape[0] for arr in arrays)
    batch_shape = (len(arrays), max_len, *arrays[0].shape[1:])
    padded = np.zeros(batch_shape, dtype=arrays[0].dtype)

    for i, arr in enumerate(arrays):
        padded[i, : arr.shape[0]] = arr

    return padded


def one_hot_encode(indices: TArr, vocab_size: int) -> TArr:
    """Create one-hot encodings from indices.

    Args:
        indices: Indices into one-hot vectors, shape [*].
        vocab_size: Length of returned vectors.

    Returns:
        np.ndarray: One-hot encoded vectors, shape [*, vocab_size].
    """

    one_hots = np.zeros((*indices.shape, vocab_size), dtype=np.long)
    np.put_along_axis(one_hots, np.expand_dims(indices, -1), 1.0, axis=-1)
    return one_hots


def adj_from_edges(edge_indices: TArr, edge_types: TArr, n_nodes: int, symmetric: bool = False) -> TArr:
    """Create adjacency matrix from edge indices and types.

    Args:
        edge_indices: Edge list, shape [n_edges, 2]. Pairs of (from_idx, to_idx).
        edge_types: Edge types, shape [n_edges].
        n_nodes: Number of nodes in the adjacency matrix.
        symmetric: If True, fill both (i,j) and (j,i) for each edge.

    Returns:
        np.ndarray: Adjacency matrix, shape [n_nodes, n_nodes].
    """

    adj = np.zeros((n_nodes, n_nodes), dtype=edge_types.dtype)

    if len(edge_indices) == 0:
        return adj

    adj[edge_indices[:, 0], edge_indices[:, 1]] = edge_types

    if symmetric:
        adj[edge_indices[:, 1], edge_indices[:, 0]] = edge_types

    return adj
