
#  Partially based on codebase by Leland McInnes (https://github.com/lmcinnes/umap)

from __future__ import print_function

import numpy as np
import numba
import scipy
from scipy.optimize import curve_fit
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances

import warnings



#INT32_MIN = np.iinfo(np.int32).min + 1
#INT32_MAX = np.iinfo(np.int32).max - 1



from collections import deque, namedtuple
from warnings import warn

import numpy as np
import numba

#from umap.sparse import sparse_mul, sparse_diff, sparse_sum

#from umap.utils import tau_rand_int, norm

import scipy.sparse
import locale

locale.setlocale(locale.LC_NUMERIC, "C")


EPS = 1e-8

RandomProjectionTreeNode = namedtuple(
    "RandomProjectionTreeNode",
    ["indices", "is_leaf", "hyperplane", "offset", "left_child", "right_child"],
)

FlatTree = namedtuple("FlatTree", ["hyperplanes", "offsets", "children", "indices"])


@numba.njit(fastmath=True)
def angular_random_projection_split(data, indices, rng_state):

    dim = data.shape[1]

    
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_norm = norm(data[left])
    right_norm = norm(data[right])

    if abs(left_norm) < EPS:
        left_norm = 1.0

    if abs(right_norm) < EPS:
        right_norm = 1.0

    
    
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = (data[left, d] / left_norm) - (
            data[right, d] / right_norm
        )

    hyperplane_norm = norm(hyperplane_vector)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0

    for d in range(dim):
        hyperplane_vector[d] = hyperplane_vector[d] / hyperplane_norm

    
    
    
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if abs(margin) < EPS:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, None


@numba.njit(fastmath=True, nogil=True)
def euclidean_random_projection_split(data, indices, rng_state):

    dim = data.shape[1]

    
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    
    
    hyperplane_offset = 0.0
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = data[left, d] - data[right, d]
        hyperplane_offset -= (
            hyperplane_vector[d] * (data[left, d] + data[right, d]) / 2.0
        )

    
    
    
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if abs(margin) < EPS:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, hyperplane_offset


@numba.njit(fastmath=True)
def sparse_angular_random_projection_split(inds, indptr, data, indices, rng_state):

    
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = data[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = data[indptr[right] : indptr[right + 1]]

    left_norm = norm(left_data)
    right_norm = norm(right_data)

    if abs(left_norm) < EPS:
        left_norm = 1.0

    if abs(right_norm) < EPS:
        right_norm = 1.0

    
    
    normalized_left_data = left_data / left_norm
    normalized_right_data = right_data / right_norm
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, normalized_left_data, right_inds, normalized_right_data
    )

    hyperplane_norm = norm(hyperplane_data)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0
    for d in range(hyperplane_data.shape[0]):
        hyperplane_data[d] = hyperplane_data[d] / hyperplane_norm

    
    
    
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0

        i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
        i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

        mul_inds, mul_data = sparse_mul(
            hyperplane_inds, hyperplane_data, i_inds, i_data
        )
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

        if abs(margin) < EPS:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

    return indices_left, indices_right, hyperplane, None


@numba.njit(fastmath=True)
def sparse_euclidean_random_projection_split(inds, indptr, data, indices, rng_state):

    
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = data[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = data[indptr[right] : indptr[right + 1]]

    
    
    hyperplane_offset = 0.0
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, left_data, right_inds, right_data
    )
    offset_inds, offset_data = sparse_sum(left_inds, left_data, right_inds, right_data)
    offset_data = offset_data / 2.0
    offset_inds, offset_data = sparse_mul(
        hyperplane_inds, hyperplane_data, offset_inds, offset_data
    )

    for d in range(offset_data.shape[0]):
        hyperplane_offset -= offset_data[d]

    
    
    
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
        i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

        mul_inds, mul_data = sparse_mul(
            hyperplane_inds, hyperplane_data, i_inds, i_data
        )
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

        if abs(margin) < EPS:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

    
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

    return indices_left, indices_right, hyperplane, hyperplane_offset


def make_euclidean_tree(data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = euclidean_random_projection_split(
            data, indices, rng_state
        )

        left_node = make_euclidean_tree(data, left_indices, rng_state, leaf_size)
        right_node = make_euclidean_tree(data, right_indices, rng_state, leaf_size)

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_angular_tree(data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = angular_random_projection_split(
            data, indices, rng_state
        )

        left_node = make_angular_tree(data, left_indices, rng_state, leaf_size)
        right_node = make_angular_tree(data, right_indices, rng_state, leaf_size)

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_sparse_euclidean_tree(inds, indptr, data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = sparse_euclidean_random_projection_split(
            inds, indptr, data, indices, rng_state
        )

        left_node = make_sparse_euclidean_tree(
            inds, indptr, data, left_indices, rng_state, leaf_size
        )
        right_node = make_sparse_euclidean_tree(
            inds, indptr, data, right_indices, rng_state, leaf_size
        )

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_sparse_angular_tree(inds, indptr, data, indices, rng_state, leaf_size=30):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = sparse_angular_random_projection_split(
            inds, indptr, data, indices, rng_state
        )

        left_node = make_sparse_angular_tree(
            inds, indptr, data, left_indices, rng_state, leaf_size
        )
        right_node = make_sparse_angular_tree(
            inds, indptr, data, right_indices, rng_state, leaf_size
        )

        node = RandomProjectionTreeNode(
            None, False, hyperplane, offset, left_node, right_node
        )
    else:
        node = RandomProjectionTreeNode(indices, True, None, None, None, None)

    return node


def make_tree(data, rng_state, leaf_size=30, angular=False):

    is_sparse = scipy.sparse.isspmatrix_csr(data)
    indices = np.arange(data.shape[0])

    
    if is_sparse:
        inds = data.indices
        indptr = data.indptr
        spdata = data.data

        if angular:
            return make_sparse_angular_tree(
                inds, indptr, spdata, indices, rng_state, leaf_size
            )
        else:
            return make_sparse_euclidean_tree(
                inds, indptr, spdata, indices, rng_state, leaf_size
            )
    else:
        if angular:
            return make_angular_tree(data, indices, rng_state, leaf_size)
        else:
            return make_euclidean_tree(data, indices, rng_state, leaf_size)


def num_nodes(tree):
    if tree.is_leaf:
        return 1
    else:
        return 1 + num_nodes(tree.left_child) + num_nodes(tree.right_child)


def num_leaves(tree):
    if tree.is_leaf:
        return 1
    else:
        return num_leaves(tree.left_child) + num_leaves(tree.right_child)


def max_sparse_hyperplane_size(tree):
    if tree.is_leaf:
        return 0
    else:
        return max(
            tree.hyperplane.shape[1],
            max_sparse_hyperplane_size(tree.left_child),
            max_sparse_hyperplane_size(tree.right_child),
        )


def recursive_flatten(
    tree, hyperplanes, offsets, children, indices, node_num, leaf_num
):
    if tree.is_leaf:
        children[node_num, 0] = -leaf_num
        indices[leaf_num, : tree.indices.shape[0]] = tree.indices
        leaf_num += 1
        return node_num, leaf_num
    else:
        if len(tree.hyperplane.shape) > 1:
            
            hyperplanes[node_num][:, : tree.hyperplane.shape[1]] = tree.hyperplane
        else:
            hyperplanes[node_num] = tree.hyperplane
        offsets[node_num] = tree.offset
        children[node_num, 0] = node_num + 1
        old_node_num = node_num
        node_num, leaf_num = recursive_flatten(
            tree.left_child,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
        )
        children[old_node_num, 1] = node_num + 1
        node_num, leaf_num = recursive_flatten(
            tree.right_child,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
        )
        return node_num, leaf_num


def flatten_tree(tree, leaf_size):
    n_nodes = num_nodes(tree)
    n_leaves = num_leaves(tree)

    if len(tree.hyperplane.shape) > 1:
        
        max_hyperplane_nnz = max_sparse_hyperplane_size(tree)
        hyperplanes = np.zeros(
            (n_nodes, tree.hyperplane.shape[0], max_hyperplane_nnz), dtype=np.float32
        )
    else:
        hyperplanes = np.zeros((n_nodes, tree.hyperplane.shape[0]), dtype=np.float32)

    offsets = np.zeros(n_nodes, dtype=np.float32)
    children = -1 * np.ones((n_nodes, 2), dtype=np.int64)
    indices = -1 * np.ones((n_leaves, leaf_size), dtype=np.int64)
    recursive_flatten(tree, hyperplanes, offsets, children, indices, 0, 0)
    return FlatTree(hyperplanes, offsets, children, indices)


@numba.njit()
def select_side(hyperplane, offset, point, rng_state):
    margin = offset
    for d in range(point.shape[0]):
        margin += hyperplane[d] * point[d]

    if abs(margin) < EPS:
        side = tau_rand_int(rng_state) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit()
def search_flat_tree(point, hyperplanes, offsets, children, indices, rng_state):
    node = 0
    while children[node, 0] > 0:
        side = select_side(hyperplanes[node], offsets[node], point, rng_state)
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0]]


def make_forest(data, n_neighbors, n_trees, rng_state, angular=False):

    result = []
    leaf_size = max(10, n_neighbors)
    try:
        result = [
            flatten_tree(make_tree(data, rng_state, leaf_size, angular), leaf_size)
            for i in range(n_trees)
        ]
    except (RuntimeError, RecursionError, SystemError):
        warn(
            "Random Projection forest initialisation failed due to recursion"
            "limit being reached. Something is a little strange with your "
            "data, and this may take longer than normal to compute."
        )

    return result


def rptree_leaf_array(rp_forest):

    if len(rp_forest) > 0:
        leaf_array = np.vstack([tree.indices for tree in rp_forest])
    else:
        leaf_array = np.array([[-1]])

    return leaf_array












import numpy as np
import numba

_mock_identity = np.eye(2, dtype=np.float64)
_mock_ones = np.ones(2, dtype=np.float64)


@numba.njit(fastmath=True)
def euclidean(x, y):

    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit()
def standardised_euclidean(x, y, sigma=_mock_ones):

    result = 0.0
    for i in range(x.shape[0]):
        result += ((x[i] - y[i]) ** 2) / sigma[i]

    return np.sqrt(result)


@numba.njit()
def manhattan(x, y):

    result = 0.0
    for i in range(x.shape[0]):
        result += np.abs(x[i] - y[i])

    return result


@numba.njit()
def chebyshev(x, y):

    result = 0.0
    for i in range(x.shape[0]):
        result = max(result, np.abs(x[i] - y[i]))

    return result


@numba.njit()
def minkowski(x, y, p=2):

    result = 0.0
    for i in range(x.shape[0]):
        result += (np.abs(x[i] - y[i])) ** p

    return result ** (1.0 / p)


@numba.njit()
def weighted_minkowski(x, y, w=_mock_ones, p=2):

    result = 0.0
    for i in range(x.shape[0]):
        result += (w[i] * np.abs(x[i] - y[i])) ** p

    return result ** (1.0 / p)


@numba.njit()
def mahalanobis(x, y, vinv=_mock_identity):
    result = 0.0

    diff = np.empty(x.shape[0], dtype=np.float64)

    for i in range(x.shape[0]):
        diff[i] = x[i] - y[i]

    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
        result += tmp * diff[i]

    return np.sqrt(result)


@numba.njit()
def hamming(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        if x[i] != y[i]:
            result += 1.0

    return float(result) / x.shape[0]


@numba.njit()
def canberra(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        denominator = np.abs(x[i]) + np.abs(y[i])
        if denominator > 0:
            result += np.abs(x[i] - y[i]) / denominator

    return result


@numba.njit()
def bray_curtis(x, y):
    numerator = 0.0
    denominator = 0.0
    for i in range(x.shape[0]):
        numerator += np.abs(x[i] - y[i])
        denominator += np.abs(x[i] + y[i])

    if denominator > 0.0:
        return float(numerator) / denominator
    else:
        return 0.0


@numba.njit()
def jaccard(x, y):
    num_non_zero = 0.0
    num_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_non_zero += x_true or y_true
        num_equal += x_true and y_true

    if num_non_zero == 0.0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero


@numba.njit()
def matching(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return float(num_not_equal) / x.shape[0]


@numba.njit()
def dice(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit()
def kulsinski(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + x.shape[0]) / (
            num_not_equal + x.shape[0]
        )


@numba.njit()
def rogers_tanimoto(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@numba.njit()
def russellrao(x, y):
    num_true_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true

    if num_true_true == np.sum(x != 0) and num_true_true == np.sum(y != 0):
        return 0.0
    else:
        return float(x.shape[0] - num_true_true) / (x.shape[0])


@numba.njit()
def sokal_michener(x, y):
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_not_equal += x_true != y_true

    return (2.0 * num_not_equal) / (x.shape[0] + num_not_equal)


@numba.njit()
def sokal_sneath(x, y):
    num_true_true = 0.0
    num_not_equal = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_not_equal += x_true != y_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@numba.njit()
def haversine(x, y):
    if x.shape[0] != 2:
        raise ValueError("haversine is only defined for 2 dimensional data")
    sin_lat = np.sin(0.5 * (x[0] - y[0]))
    sin_long = np.sin(0.5 * (x[1] - y[1]))
    result = np.sqrt(sin_lat ** 2 + np.cos(x[0]) * np.cos(y[0]) * sin_long ** 2)
    return 2.0 * np.arcsin(result)


@numba.njit()
def yule(x, y):
    num_true_true = 0.0
    num_true_false = 0.0
    num_false_true = 0.0
    for i in range(x.shape[0]):
        x_true = x[i] != 0
        y_true = y[i] != 0
        num_true_true += x_true and y_true
        num_true_false += x_true and (not y_true)
        num_false_true += (not x_true) and y_true

    num_false_false = x.shape[0] - num_true_true - num_true_false - num_false_true

    if num_true_false == 0.0 or num_false_true == 0.0:
        return 0.0
    else:
        return (2.0 * num_true_false * num_false_true) / (
            num_true_true * num_false_false + num_true_false * num_false_true
        )


@numba.njit()
def cosine(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return 1.0
    else:
        return 1.0 - (result / np.sqrt(norm_x * norm_y))


@numba.njit()
def correlation(x, y):
    mu_x = 0.0
    mu_y = 0.0
    norm_x = 0.0
    norm_y = 0.0
    dot_product = 0.0

    for i in range(x.shape[0]):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= x.shape[0]
    mu_y /= x.shape[0]

    for i in range(x.shape[0]):
        shifted_x = x[i] - mu_x
        shifted_y = y[i] - mu_y
        norm_x += shifted_x ** 2
        norm_y += shifted_y ** 2
        dot_product += shifted_x * shifted_y

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / np.sqrt(norm_x * norm_y))


named_distances = {
    
    "euclidean": euclidean,
    "l2": euclidean,
    "manhattan": manhattan,
    "taxicab": manhattan,
    "l1": manhattan,
    "chebyshev": chebyshev,
    "linfinity": chebyshev,
    "linfty": chebyshev,
    "linf": chebyshev,
    "minkowski": minkowski,
    
    "seuclidean": standardised_euclidean,
    "standardised_euclidean": standardised_euclidean,
    "wminkowski": weighted_minkowski,
    "weighted_minkowski": weighted_minkowski,
    "mahalanobis": mahalanobis,
    
    "canberra": canberra,
    "cosine": cosine,
    "correlation": correlation,
    "haversine": haversine,
    "braycurtis": bray_curtis,
    
    "hamming": hamming,
    "jaccard": jaccard,
    "dice": dice,
    "matching": matching,
    "kulsinski": kulsinski,
    "rogerstanimoto": rogers_tanimoto,
    "russellrao": russellrao,
    "sokalsneath": sokal_sneath,
    "sokalmichener": sokal_michener,
    "yule": yule,
}














import time

import numpy as np
import numba


@numba.njit(parallel=True)
def fast_knn_indices(X, n_neighbors):

    knn_indices = np.empty(
        (X.shape[0], n_neighbors), dtype=np.int32
    )
    for row in numba.prange(X.shape[0]):
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices


@numba.njit("i4(i8[:])")
def tau_rand_int(state):

    state[0] = (
        ((state[0] & 4294967294) << 12) & 0xFFFFFFFF
    ) ^ ((((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19)
    state[1] = (
        ((state[1] & 4294967288) << 4) & 0xFFFFFFFF
    ) ^ ((((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25)
    state[2] = (
        ((state[2] & 4294967280) << 17) & 0xFFFFFFFF
    ) ^ ((((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11)

    return state[0] ^ state[1] ^ state[2]


@numba.njit("f4(i8[:])")
def tau_rand(state):

    integer = tau_rand_int(state)
    return abs(float(integer) / 0x7FFFFFFF)


@numba.njit()
def norm(vec):

    result = 0.0
    for i in range(vec.shape[0]):
        result += vec[i] ** 2
    return np.sqrt(result)


@numba.njit()
def rejection_sample(n_samples, pool_size, rng_state):

    result = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = tau_rand_int(rng_state) % pool_size
            for k in range(i):
                if j == result[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result


@numba.njit("f8[:, :, :](i8,i8)")
def make_heap(n_points, size):

    result = np.zeros(
        (3, int(n_points), int(size)), dtype=np.float64
    )
    result[0] = -1
    result[1] = np.inf
    result[2] = 0

    return result


@numba.njit("i8(f8[:,:,:],i8,f8,i8,i8)")
def heap_push(heap, row, weight, index, flag):

    row = int(row)
    indices = heap[0, row]
    weights = heap[1, row]
    is_new = heap[2, row]

    if weight >= weights[0]:
        return 0

    
    for i in range(indices.shape[0]):
        if index == indices[i]:
            return 0

    
    weights[0] = weight
    indices[0] = index
    is_new[0] = flag

    
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heap.shape[2]:
            break
        elif ic2 >= heap.shape[2]:
            if weights[ic1] > weight:
                i_swap = ic1
            else:
                break
        elif weights[ic1] >= weights[ic2]:
            if weight < weights[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if weight < weights[ic2]:
                i_swap = ic2
            else:
                break

        weights[i] = weights[i_swap]
        indices[i] = indices[i_swap]
        is_new[i] = is_new[i_swap]

        i = i_swap

    weights[i] = weight
    indices[i] = index
    is_new[i] = flag

    return 1


@numba.njit("i8(f8[:,:,:],i8,f8,i8,i8)")
def unchecked_heap_push(heap, row, weight, index, flag):

    indices = heap[0, row]
    weights = heap[1, row]
    is_new = heap[2, row]

    if weight >= weights[0]:
        return 0

    
    weights[0] = weight
    indices[0] = index
    is_new[0] = flag

    
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= heap.shape[2]:
            break
        elif ic2 >= heap.shape[2]:
            if weights[ic1] > weight:
                i_swap = ic1
            else:
                break
        elif weights[ic1] >= weights[ic2]:
            if weight < weights[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if weight < weights[ic2]:
                i_swap = ic2
            else:
                break

        weights[i] = weights[i_swap]
        indices[i] = indices[i_swap]
        is_new[i] = is_new[i_swap]

        i = i_swap

    weights[i] = weight
    indices[i] = index
    is_new[i] = flag

    return 1


@numba.njit()
def siftdown(heap1, heap2, elt):

    while elt * 2 + 1 < heap1.shape[0]:
        left_child = elt * 2 + 1
        right_child = left_child + 1
        swap = elt

        if heap1[swap] < heap1[left_child]:
            swap = left_child

        if (
            right_child < heap1.shape[0]
            and heap1[swap] < heap1[right_child]
        ):
            swap = right_child

        if swap == elt:
            break
        else:
            heap1[elt], heap1[swap] = (
                heap1[swap],
                heap1[elt],
            )
            heap2[elt], heap2[swap] = (
                heap2[swap],
                heap2[elt],
            )
            elt = swap


@numba.njit()
def deheap_sort(heap):

    indices = heap[0]
    weights = heap[1]

    for i in range(indices.shape[0]):

        ind_heap = indices[i]
        dist_heap = weights[i]

        for j in range(ind_heap.shape[0] - 1):
            ind_heap[0], ind_heap[
                ind_heap.shape[0] - j - 1
            ] = (
                ind_heap[ind_heap.shape[0] - j - 1],
                ind_heap[0],
            )
            dist_heap[0], dist_heap[
                dist_heap.shape[0] - j - 1
            ] = (
                dist_heap[dist_heap.shape[0] - j - 1],
                dist_heap[0],
            )

            siftdown(
                dist_heap[: dist_heap.shape[0] - j - 1],
                ind_heap[: ind_heap.shape[0] - j - 1],
                0,
            )

    return indices.astype(np.int64), weights


@numba.njit("i8(f8[:, :, :],i8)")
def smallest_flagged(heap, row):

    ind = heap[0, row]
    dist = heap[1, row]
    flag = heap[2, row]

    min_dist = np.inf
    result_index = -1

    for i in range(ind.shape[0]):
        if flag[i] == 1 and dist[i] < min_dist:
            min_dist = dist[i]
            result_index = i

    if result_index >= 0:
        flag[result_index] = 0.0
        return int(ind[result_index])
    else:
        return -1


@numba.njit(parallel=True)
def build_candidates(
    current_graph,
    n_vertices,
    n_neighbors,
    max_candidates,
    rng_state,
):

    candidate_neighbors = make_heap(
        n_vertices, max_candidates
    )
    for i in range(n_vertices):
        for j in range(n_neighbors):
            if current_graph[0, i, j] < 0:
                continue
            idx = current_graph[0, i, j]
            isn = current_graph[2, i, j]
            d = tau_rand(rng_state)
            heap_push(candidate_neighbors, i, d, idx, isn)
            heap_push(candidate_neighbors, idx, d, i, isn)
            current_graph[2, i, j] = 0

    return candidate_neighbors


@numba.njit(parallel=True)
def new_build_candidates(
    current_graph,
    n_vertices,
    n_neighbors,
    max_candidates,
    rng_state,
    rho=0.5,
):  

    new_candidate_neighbors = make_heap(
        n_vertices, max_candidates
    )
    old_candidate_neighbors = make_heap(
        n_vertices, max_candidates
    )

    for i in numba.prange(n_vertices):
        for j in range(n_neighbors):
            if current_graph[0, i, j] < 0:
                continue
            idx = current_graph[0, i, j]
            isn = current_graph[2, i, j]
            d = tau_rand(rng_state)
            if tau_rand(rng_state) < rho:
                c = 0
                if isn:
                    c += heap_push(
                        new_candidate_neighbors,
                        i,
                        d,
                        idx,
                        isn,
                    )
                    c += heap_push(
                        new_candidate_neighbors,
                        idx,
                        d,
                        i,
                        isn,
                    )
                else:
                    heap_push(
                        old_candidate_neighbors,
                        i,
                        d,
                        idx,
                        isn,
                    )
                    heap_push(
                        old_candidate_neighbors,
                        idx,
                        d,
                        i,
                        isn,
                    )

                if c > 0:
                    current_graph[2, i, j] = 0

    return new_candidate_neighbors, old_candidate_neighbors


@numba.njit(parallel=True)
def submatrix(dmat, indices_col, n_neighbors):

    n_samples_transform, n_samples_fit = dmat.shape
    submat = np.zeros(
        (n_samples_transform, n_neighbors), dtype=dmat.dtype
    )
    for i in numba.prange(n_samples_transform):
        for j in numba.prange(n_neighbors):
            submat[i, j] = dmat[i, indices_col[i, j]]
    return submat



def ts():
    return time.ctime(time.time())
















import numpy as np
import numba













#from umap.rp_tree import search_flat_tree


def make_nn_descent(dist, dist_args):


    @numba.njit()
    def nn_descent(
        data,
        n_neighbors,
        rng_state,
        max_candidates=50,
        n_iters=10,
        delta=0.001,
        rho=0.5,
        rp_tree_init=True,
        leaf_array=None,
        verbose=False,
    ):
        n_vertices = data.shape[0]

        current_graph = make_heap(data.shape[0], n_neighbors)
        for i in range(data.shape[0]):
            indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
            for j in range(indices.shape[0]):
                d = dist(data[i], data[indices[j]], *dist_args)
                heap_push(current_graph, i, d, indices[j], 1)
                heap_push(current_graph, indices[j], d, i, 1)

        if rp_tree_init:
            for n in range(leaf_array.shape[0]):
                for i in range(leaf_array.shape[1]):
                    if leaf_array[n, i] < 0:
                        break
                    for j in range(i + 1, leaf_array.shape[1]):
                        if leaf_array[n, j] < 0:
                            break
                        d = dist(
                            data[leaf_array[n, i]], data[leaf_array[n, j]], *dist_args
                        )
                        heap_push(
                            current_graph, leaf_array[n, i], d, leaf_array[n, j], 1
                        )
                        heap_push(
                            current_graph, leaf_array[n, j], d, leaf_array[n, i], 1
                        )

        for n in range(n_iters):
            if verbose:
                print("\t", n, " / ", n_iters)

            candidate_neighbors = build_candidates(
                current_graph, n_vertices, n_neighbors, max_candidates, rng_state
            )

            c = 0
            for i in range(n_vertices):
                for j in range(max_candidates):
                    p = int(candidate_neighbors[0, i, j])
                    if p < 0 or tau_rand(rng_state) < rho:
                        continue
                    for k in range(max_candidates):
                        q = int(candidate_neighbors[0, i, k])
                        if (
                            q < 0
                            or not candidate_neighbors[2, i, j]
                            and not candidate_neighbors[2, i, k]
                        ):
                            continue

                        d = dist(data[p], data[q], *dist_args)
                        c += heap_push(current_graph, p, d, q, 1)
                        c += heap_push(current_graph, q, d, p, 1)

            if c <= delta * n_neighbors * data.shape[0]:
                break

        return deheap_sort(current_graph)

    return nn_descent


def make_initialisations(dist, dist_args):
    @numba.njit(parallel=True)
    def init_from_random(n_neighbors, data, query_points, heap, rng_state):
        for i in range(query_points.shape[0]):
            indices = rejection_sample(n_neighbors, data.shape[0], rng_state)
            for j in range(indices.shape[0]):
                if indices[j] < 0:
                    continue
                d = dist(data[indices[j]], query_points[i], *dist_args)
                heap_push(heap, i, d, indices[j], 1)
        return

    @numba.njit(parallel=True)
    def init_from_tree(tree, data, query_points, heap, rng_state):
        for i in range(query_points.shape[0]):
            indices = search_flat_tree(
                query_points[i],
                tree.hyperplanes,
                tree.offsets,
                tree.children,
                tree.indices,
                rng_state,
            )

            for j in range(indices.shape[0]):
                if indices[j] < 0:
                    continue
                d = dist(data[indices[j]], query_points[i], *dist_args)
                heap_push(heap, i, d, indices[j], 1)

        return

    return init_from_random, init_from_tree


def initialise_search(
    forest, data, query_points, n_neighbors, init_from_random, init_from_tree, rng_state
):
    results = make_heap(query_points.shape[0], n_neighbors)
    init_from_random(n_neighbors, data, query_points, results, rng_state)
    if forest is not None:
        for tree in forest:
            init_from_tree(tree, data, query_points, results, rng_state)

    return results


def make_initialized_nnd_search(dist, dist_args):
    @numba.njit(parallel=True)
    def initialized_nnd_search(data, indptr, indices, initialization, query_points):

        for i in numba.prange(query_points.shape[0]):

            tried = set(initialization[0, i])

            while True:

                
                vertex = smallest_flagged(initialization, i)

                if vertex == -1:
                    break
                candidates = indices[indptr[vertex] : indptr[vertex + 1]]
                for j in range(candidates.shape[0]):
                    if (
                        candidates[j] == vertex
                        or candidates[j] == -1
                        or candidates[j] in tried
                    ):
                        continue
                    d = dist(data[candidates[j]], query_points[i], *dist_args)
                    unchecked_heap_push(initialization, i, d, candidates[j], 1)
                    tried.add(candidates[j])

        return initialization

    return initialized_nnd_search

















import numpy as np
import numba












import locale

locale.setlocale(locale.LC_NUMERIC, "C")


@numba.njit()
def arr_unique(arr):
    aux = np.sort(arr)
    flag = np.concatenate((np.ones(1, dtype=np.bool_), aux[1:] != aux[:-1]))
    return aux[flag]



@numba.njit()
def arr_union(ar1, ar2):
    if ar1.shape[0] == 0:
        return ar2
    elif ar2.shape[0] == 0:
        return ar1
    else:
        return arr_unique(np.concatenate((ar1, ar2)))




@numba.njit()
def arr_intersect(ar1, ar2):
    aux = np.concatenate((ar1, ar2))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


@numba.njit()
def sparse_sum(ind1, data1, ind2, data2):
    result_ind = arr_union(ind1, ind2)
    result_data = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] + data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            val = data1[i1]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
        else:
            val = data2[i2]
            if val != 0:
                result_ind[nnz] = j2
                result_data[nnz] = val
                nnz += 1
            i2 += 1

    
    while i1 < ind1.shape[0]:
        val = data1[i1]
        if val != 0:
            result_ind[nnz] = i1
            result_data[nnz] = val
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        val = data2[i2]
        if val != 0:
            result_ind[nnz] = i2
            result_data[nnz] = val
            nnz += 1
        i2 += 1

    
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


@numba.njit()
def sparse_diff(ind1, data1, ind2, data2):
    return sparse_sum(ind1, data1, ind2, -data2)


@numba.njit()
def sparse_mul(ind1, data1, ind2, data2):
    result_ind = arr_intersect(ind1, ind2)
    result_data = np.zeros(result_ind.shape[0], dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] * data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1

    
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


def make_sparse_nn_descent(sparse_dist, dist_args):

    @numba.njit(parallel=True)
    def nn_descent(
        inds,
        indptr,
        data,
        n_vertices,
        n_neighbors,
        rng_state,
        max_candidates=50,
        n_iters=10,
        delta=0.001,
        rho=0.5,
        rp_tree_init=True,
        leaf_array=None,
        verbose=False,
    ):

        current_graph = make_heap(n_vertices, n_neighbors)
        for i in range(n_vertices):
            indices = rejection_sample(n_neighbors, n_vertices, rng_state)
            for j in range(indices.shape[0]):

                from_inds = inds[indptr[i] : indptr[i + 1]]
                from_data = data[indptr[i] : indptr[i + 1]]

                to_inds = inds[indptr[indices[j]] : indptr[indices[j] + 1]]
                to_data = data[indptr[indices[j]] : indptr[indices[j] + 1]]

                d = sparse_dist(from_inds, from_data, to_inds, to_data, *dist_args)

                heap_push(current_graph, i, d, indices[j], 1)
                heap_push(current_graph, indices[j], d, i, 1)

        if rp_tree_init:
            for n in range(leaf_array.shape[0]):
                for i in range(leaf_array.shape[1]):
                    if leaf_array[n, i] < 0:
                        break
                    for j in range(i + 1, leaf_array.shape[1]):
                        if leaf_array[n, j] < 0:
                            break

                        from_inds = inds[
                            indptr[leaf_array[n, i]] : indptr[leaf_array[n, i] + 1]
                        ]
                        from_data = data[
                            indptr[leaf_array[n, i]] : indptr[leaf_array[n, i] + 1]
                        ]

                        to_inds = inds[
                            indptr[leaf_array[n, j]] : indptr[leaf_array[n, j] + 1]
                        ]
                        to_data = data[
                            indptr[leaf_array[n, j]] : indptr[leaf_array[n, j] + 1]
                        ]

                        d = sparse_dist(
                            from_inds, from_data, to_inds, to_data, *dist_args
                        )

                        heap_push(
                            current_graph, leaf_array[n, i], d, leaf_array[n, j], 1
                        )
                        heap_push(
                            current_graph, leaf_array[n, j], d, leaf_array[n, i], 1
                        )

        for n in range(n_iters):
            if verbose:
                print("\t", n, " / ", n_iters)

            candidate_neighbors = build_candidates(
                current_graph, n_vertices, n_neighbors, max_candidates, rng_state
            )

            c = 0
            for i in range(n_vertices):
                for j in range(max_candidates):
                    p = int(candidate_neighbors[0, i, j])
                    if p < 0 or tau_rand(rng_state) < rho:
                        continue
                    for k in range(max_candidates):
                        q = int(candidate_neighbors[0, i, k])
                        if (
                            q < 0
                            or not candidate_neighbors[2, i, j]
                            and not candidate_neighbors[2, i, k]
                        ):
                            continue

                        from_inds = inds[indptr[p] : indptr[p + 1]]
                        from_data = data[indptr[p] : indptr[p + 1]]

                        to_inds = inds[indptr[q] : indptr[q + 1]]
                        to_data = data[indptr[q] : indptr[q + 1]]

                        d = sparse_dist(
                            from_inds, from_data, to_inds, to_data, *dist_args
                        )

                        c += heap_push(current_graph, p, d, q, 1)
                        c += heap_push(current_graph, q, d, p, 1)

            if c <= delta * n_neighbors * n_vertices:
                break

        return deheap_sort(current_graph)

    return nn_descent


@numba.njit()
def general_sset_intersection(
    indptr1,
    indices1,
    data1,
    indptr2,
    indices2,
    data2,
    result_row,
    result_col,
    result_val,
    mix_weight=0.5,
):

    left_min = max(data1.min() / 2.0, 1.0e-8)
    right_min = max(data2.min() / 2.0, 1.0e-8)

    for idx in range(result_row.shape[0]):
        i = result_row[idx]
        j = result_col[idx]

        left_val = left_min
        for k in range(indptr1[i], indptr1[i + 1]):
            if indices1[k] == j:
                left_val = data1[k]

        right_val = right_min
        for k in range(indptr2[i], indptr2[i + 1]):
            if indices2[k] == j:
                right_val = data2[k]

        if left_val > left_min or right_val > right_min:
            if mix_weight < 0.5:
                result_val[idx] = left_val * pow(
                    right_val, mix_weight / (1.0 - mix_weight)
                )
            else:
                result_val[idx] = (
                    pow(left_val, (1.0 - mix_weight) / mix_weight) * right_val
                )

    return


@numba.njit()
def sparse_euclidean(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += aux_data[i] ** 2
    return np.sqrt(result)


@numba.njit()
def sparse_manhattan(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i])
    return result


@numba.njit()
def sparse_chebyshev(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result = max(result, np.abs(aux_data[i]))
    return result


@numba.njit()
def sparse_minkowski(ind1, data1, ind2, data2, p=2.0):
    aux_inds, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i]) ** p
    return result ** (1.0 / p)


@numba.njit()
def sparse_hamming(ind1, data1, ind2, data2, n_features):
    num_not_equal = sparse_diff(ind1, data1, ind2, data2)[0].shape[0]
    return float(num_not_equal) / n_features


@numba.njit()
def sparse_canberra(ind1, data1, ind2, data2):
    abs_data1 = np.abs(data1)
    abs_data2 = np.abs(data2)
    denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)
    denom_data = 1.0 / denom_data
    numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
    numer_data = np.abs(numer_data)

    val_inds, val_data = sparse_mul(numer_inds, numer_data, denom_inds, denom_data)

    return np.sum(val_data)


@numba.njit()
def sparse_bray_curtis(ind1, data1, ind2, data2):  
    abs_data1 = np.abs(data1)
    abs_data2 = np.abs(data2)
    denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)

    if denom_data.shape[0] == 0:
        return 0.0

    denominator = np.sum(denom_data)

    numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
    numer_data = np.abs(numer_data)

    numerator = np.sum(numer_data)

    return float(numerator) / denominator


@numba.njit()
def sparse_jaccard(ind1, data1, ind2, data2):
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_equal = arr_intersect(ind1, ind2).shape[0]

    if num_non_zero == 0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero


@numba.njit()
def sparse_matching(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return float(num_not_equal) / n_features


@numba.njit()
def sparse_dice(ind1, data1, ind2, data2):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit()
def sparse_kulsinski(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + n_features) / (
            num_not_equal + n_features
        )


@numba.njit()
def sparse_rogers_tanimoto(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return (2.0 * num_not_equal) / (n_features + num_not_equal)


@numba.njit()
def sparse_russellrao(ind1, data1, ind2, data2, n_features):
    if ind1.shape[0] == ind2.shape[0] and np.all(ind1 == ind2):
        return 0.0

    num_true_true = arr_intersect(ind1, ind2).shape[0]

    if num_true_true == np.sum(data1 != 0) and num_true_true == np.sum(data2 != 0):
        return 0.0
    else:
        return float(n_features - num_true_true) / (n_features)


@numba.njit()
def sparse_sokal_michener(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return (2.0 * num_not_equal) / (n_features + num_not_equal)


@numba.njit()
def sparse_sokal_sneath(ind1, data1, ind2, data2):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@numba.njit()
def sparse_cosine(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = norm(data1)
    norm2 = norm(data2)

    for i in range(aux_data.shape[0]):
        result += aux_data[i]

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    else:
        return 1.0 - (result / (norm1 * norm2))


@numba.njit()
def sparse_correlation(ind1, data1, ind2, data2, n_features):

    mu_x = 0.0
    mu_y = 0.0
    dot_product = 0.0

    if ind1.shape[0] == 0 and ind2.shape[0] == 0:
        return 0.0
    elif ind1.shape[0] == 0 or ind2.shape[0] == 0:
        return 1.0

    for i in range(data1.shape[0]):
        mu_x += data1[i]
    for i in range(data2.shape[0]):
        mu_y += data2[i]

    mu_x /= n_features
    mu_y /= n_features

    shifted_data1 = np.empty(data1.shape[0], dtype=np.float32)
    shifted_data2 = np.empty(data2.shape[0], dtype=np.float32)

    for i in range(data1.shape[0]):
        shifted_data1[i] = data1[i] - mu_x
    for i in range(data2.shape[0]):
        shifted_data2[i] = data2[i] - mu_y

    norm1 = np.sqrt(
        (norm(shifted_data1) ** 2) + (n_features - ind1.shape[0]) * (mu_x ** 2)
    )
    norm2 = np.sqrt(
        (norm(shifted_data2) ** 2) + (n_features - ind2.shape[0]) * (mu_y ** 2)
    )

    dot_prod_inds, dot_prod_data = sparse_mul(ind1, shifted_data1, ind2, shifted_data2)

    common_indices = set(dot_prod_inds)

    for i in range(dot_prod_data.shape[0]):
        dot_product += dot_prod_data[i]

    for i in range(ind1.shape[0]):
        if ind1[i] not in common_indices:
            dot_product -= shifted_data1[i] * (mu_y)

    for i in range(ind2.shape[0]):
        if ind2[i] not in common_indices:
            dot_product -= shifted_data2[i] * (mu_x)

    all_indices = arr_union(ind1, ind2)
    dot_product += mu_x * mu_y * (n_features - all_indices.shape[0])

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / (norm1 * norm2))


sparse_named_distances = {
    
    "euclidean": sparse_euclidean,
    "manhattan": sparse_manhattan,
    "l1": sparse_manhattan,
    "taxicab": sparse_manhattan,
    "chebyshev": sparse_chebyshev,
    "linf": sparse_chebyshev,
    "linfty": sparse_chebyshev,
    "linfinity": sparse_chebyshev,
    "minkowski": sparse_minkowski,
    
    "canberra": sparse_canberra,
    
    
    "hamming": sparse_hamming,
    "jaccard": sparse_jaccard,
    "dice": sparse_dice,
    "matching": sparse_matching,
    "kulsinski": sparse_kulsinski,
    "rogerstanimoto": sparse_rogers_tanimoto,
    "russellrao": sparse_russellrao,
    "sokalmichener": sparse_sokal_michener,
    "sokalsneath": sparse_sokal_sneath,
    "cosine": sparse_cosine,
    "correlation": sparse_correlation,
}

sparse_need_n_features = (
    "hamming",
    "matching",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "correlation",
)















import numpy as np
import numba
import scipy
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree







#INT32_MIN = np.iinfo(np.int32).min + 1
#INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

def nearest_neighbors(
    X,
    n_neighbors,
    metric,
    metric_kwds,
    angular,
    random_state,
    verbose=False,
):

    if verbose:
        print("Finding Nearest Neighbors")

    if metric == "precomputed":
        
        
        knn_indices = fast_knn_indices(X, n_neighbors)
        
        
        knn_dists = X[
            np.arange(X.shape[0])[:, None], knn_indices
        ].copy()

        rp_forest = []
    else:
        if callable(metric):
            distance_func = metric
        elif metric in named_distances:
            distance_func = named_distances[metric]
        else:
            raise ValueError(
                "Metric is neither callable, "
                + "nor a recognised string"
            )

        if metric in (
            "cosine",
            "correlation",
            "dice",
            "jaccard",
        ):
            angular = True

        rng_state = random_state.randint(
            np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3
        ).astype(np.int64)

        if scipy.sparse.isspmatrix_csr(X):
            if metric in sparse.sparse_named_distances:
                distance_func = sparse.sparse_named_distances[
                    metric
                ]
                if metric in sparse.sparse_need_n_features:
                    metric_kwds["n_features"] = X.shape[1]
            else:
                raise ValueError(
                    "Metric {} not supported for sparse "
                    + "data".format(metric)
                )
            metric_nn_descent = sparse.make_sparse_nn_descent(
                distance_func, tuple(metric_kwds.values())
            )

            
            n_trees = 5 + int(
                round((X.shape[0]) ** 0.5 / 20.0)
            )
            n_iters = max(
                5, int(round(np.log2(X.shape[0])))
            )
            if verbose:
                print(
                    "Building RP forest with",
                    str(n_trees),
                    "trees",
                )

            rp_forest = make_forest(
                X, n_neighbors, n_trees, rng_state, angular
            )
            leaf_array = rptree_leaf_array(rp_forest)

            if verbose:
                print(
                    "NN descent for",
                    str(n_iters),
                    "iterations",
                )
            knn_indices, knn_dists = metric_nn_descent(
                X.indices,
                X.indptr,
                X.data,
                X.shape[0],
                n_neighbors,
                rng_state,
                max_candidates=60,
                rp_tree_init=True,
                leaf_array=leaf_array,
                n_iters=n_iters,
                verbose=verbose,
            )
        else:
            metric_nn_descent = make_nn_descent(
                distance_func, tuple(metric_kwds.values())
            )
            
            n_trees = 5 + int(
                round((X.shape[0]) ** 0.5 / 20.0)
            )
            n_iters = max(
                5, int(round(np.log2(X.shape[0])))
            )

            if verbose:
                print(
                    "Building RP forest with",
                    str(n_trees),
                    "trees",
                )
            rp_forest = make_forest(
                X, n_neighbors, n_trees, rng_state, angular
            )
            leaf_array = rptree_leaf_array(rp_forest)
            if verbose:
                print(
                    "NN descent for",
                    str(n_iters),
                    "iterations",
                )
            knn_indices, knn_dists = metric_nn_descent(
                X,
                n_neighbors,
                rng_state,
                max_candidates=60,
                rp_tree_init=True,
                leaf_array=leaf_array,
                n_iters=n_iters,
                verbose=verbose,
            )

        if np.any(knn_indices < 0):
            warn(
                "Failed to correctly find n_neighbors for some samples."
                "Results may be less than ideal. Try re-running with"
                "different parameters."
            )
    if verbose:
        print("Finished Nearest Neighbor Search")
    return knn_indices, knn_dists, rp_forest

@numba.njit(
    fastmath=True
)  
def smooth_knn_dist(
    distances,
    k,
    n_iter=64,
    local_connectivity=1.0,
    bandwidth=1.0,
    cardinality=None
):

    if cardinality is None:
        target = np.log2(k) * bandwidth
    else:
        target = cardinality
    rho = np.zeros(distances.shape[0])
    result = np.zeros(distances.shape[0])

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index]
                        - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if (
                result[i]
                < MIN_K_DIST_SCALE * mean_ith_distances
            ):
                result[i] = (
                    MIN_K_DIST_SCALE * mean_ith_distances
                )
        else:
            if (
                result[i]
                < MIN_K_DIST_SCALE * mean_distances
            ):
                result[i] = (
                    MIN_K_DIST_SCALE * mean_distances
                )

    return result, rho

@numba.njit(parallel=True, fastmath=True)
def compute_membership_strengths(
    knn_indices, knn_dists, sigmas, rhos
):

    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int64)
    cols = np.zeros(knn_indices.size, dtype=np.int64)
    vals = np.zeros(knn_indices.size, dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0:
                val = 1.0
            else:
                val = np.exp(
                    -(
                        (knn_dists[i, j] - rhos[i])
                        / (sigmas[i])
                    )
                )

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals

def create_tree(data, metric):
    if metric == 'euclidean':
        ckd = cKDTree(data)
    else:
        ckd = KDTree(data, metric=metric)
    return ckd

def query_tree(data, ckd, k, metric):
    if metric == 'euclidean':
        ckdout = ckd.query(x=data, k=k, workers=-1)
    else:
        ckdout = ckd.query(data, k=k)
    return ckdout

def partitioned_nearest_neighbors(X, Y, k, metric='euclidean'):
    tree = create_tree(Y, metric)
    nns = query_tree(X, tree, k, metric)
    knn_indices = nns[1]
    knn_dists = nns[0]
    return knn_indices, knn_dists













import numpy as np

import scipy.sparse
import scipy.sparse.csgraph

from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from warnings import warn


def component_layout(
    data, n_components, component_labels, dim, metric="euclidean", metric_kwds={}
):

    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)

    for label in range(n_components):
        component_centroids[label] = data[component_labels == label].mean(axis=0)

    distance_matrix = pairwise_distances(
        component_centroids, metric=metric, **metric_kwds
    )
    affinity_matrix = np.exp(-distance_matrix ** 2)

    component_embedding = SpectralEmbedding(
        n_components=dim, affinity="precomputed"
    ).fit_transform(affinity_matrix)
    component_embedding /= component_embedding.max()

    return component_embedding


def multi_component_layout(
    data,
    graph,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds={},
):


    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        meta_embedding = component_layout(
            data,
            n_components,
            component_labels,
            dim,
            metric=metric,
            metric_kwds=metric_kwds,
        )
    else:
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]

    for label in range(n_components):
        component_graph = graph.tocsr()[component_labels == label, :].tocsc()
        component_graph = component_graph[:, component_labels == label].tocoo()

        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0

        if component_graph.shape[0] < 2 * dim:
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )
            continue

        diag_data = np.asarray(component_graph.sum(axis=0))
        
        
        
        
        I = scipy.sparse.identity(component_graph.shape[0], dtype=np.float64)
        D = scipy.sparse.spdiags(
            1.0 / np.sqrt(diag_data),
            0,
            component_graph.shape[0],
            component_graph.shape[0],
        )
        L = I - D * component_graph * D

        k = dim + 1
        num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(component_graph.shape[0])))
        try:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
            order = np.argsort(eigenvalues)[1:k]
            component_embedding = eigenvectors[:, order]
            expansion = data_range / np.max(np.abs(component_embedding))
            component_embedding *= expansion
            result[component_labels == label] = (
                component_embedding + meta_embedding[label]
            )
        except scipy.sparse.linalg.ArpackError:
            warn(
                "WARNING: spectral initialisation failed! The eigenvector solver\n"
                "failed. This is likely due to too small an eigengap. Consider\n"
                "adding some noise or jitter to your data.\n\n"
                "Falling back to random initialisation!"
            )
            result[component_labels == label] = (
                random_state.uniform(
                    low=-data_range,
                    high=data_range,
                    size=(component_graph.shape[0], dim),
                )
                + meta_embedding[label]
            )

    return result


def spectral_layout(data, graph, dim, random_state, metric="euclidean", metric_kwds={}):

    n_samples = graph.shape[0]
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)

    if n_components > 1:
        warn(
            "Embedding a total of {} separate connected components using meta-embedding (experimental)".format(
                n_components
            )
        )
        return multi_component_layout(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )

    diag_data = np.asarray(graph.sum(axis=0))
    
    
    
    
    I = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    D = scipy.sparse.spdiags(
        1.0 / np.sqrt(diag_data), 0, graph.shape[0], graph.shape[0]
    )
    L = I - D * graph * D

    k = dim + 1
    num_lanczos_vectors = max(2 * k + 1, int(np.sqrt(graph.shape[0])))
    try:
        if L.shape[0] < 2000000:
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                L,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=1e-4,
                v0=np.ones(L.shape[0]),
                maxiter=graph.shape[0] * 5,
            )
        else:
            eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                L, random_state.normal(size=(L.shape[0], k)), largest=False, tol=1e-8
            )
        order = np.argsort(eigenvalues)[1:k]
        return eigenvectors[:, order]
    except scipy.sparse.linalg.ArpackError:
        warn(
            "WARNING: spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        return random_state.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))












import numpy as np
import numba


@numba.njit()
def clip(val):

    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
    },
)
def rdist(x, y):

    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):  
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = head_embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = tau_rand_int(rng_state) % n_vertices

                other = head_embedding[k]

                dist_squared = rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (0.001 + dist_squared) * (
                        a * pow(dist_squared, b) + 1
                    )
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )

    return head_embedding









def fuzzy_simplicial_set(
    Xs,
    joint,
    joint_idxs,
    weights,
    n_neighbors,
    cardinality,
    metrics,
    metric_kwds,
    joint_metrics,
    angular,
    set_op_mix_ratio,
    local_connectivity,
    n_epochs,
    random_state,
    verbose,
):

    len_Xs = [len(i) for i in Xs]
    
    rows, cols, vals = np.array([]), np.array([]), np.array([])

    for i in range(len(Xs)):

        X_n_neighbors = int(round(n_neighbors * len_Xs[i]/sum(len_Xs)))
        if X_n_neighbors < 2:
            weights[(i,i)] *= X_n_neighbors/2
            X_n_neighbors = 2

        if Xs[i].shape[0] < 4096:
            X = Xs[i]
            if scipy.sparse.issparse(Xs[i]):
                X = Xs[i].toarray()
            dmat = pairwise_distances(Xs[i], metric=metrics[i], **metric_kwds[i])
            knn_indices, knn_dists, _ = nearest_neighbors(
                dmat,
                X_n_neighbors,
                'precomputed',
                {},
                angular,
                np.random.RandomState(random_state),
                verbose=verbose,
            )
        else:
            knn_indices, knn_dists, _ = nearest_neighbors(
                Xs[i],
                X_n_neighbors,
                metrics[i],
                metric_kwds[i],
                angular,
                np.random.RandomState(random_state),
                verbose=verbose,
            )

        sigmas, rhos = smooth_knn_dist(
            knn_dists,
            0,
            local_connectivity=local_connectivity,
            cardinality=cardinality * X_n_neighbors/n_neighbors
        )

        X_rows, X_cols, X_vals = compute_membership_strengths(
            knn_indices, knn_dists, sigmas, rhos
        )

        rows = np.concatenate([rows, X_rows + sum(len_Xs[:i])])
        cols = np.concatenate([cols, X_cols + sum(len_Xs[:i])])
        vals = np.concatenate([vals, X_vals])

    for k in joint.keys():
        XY = joint[k]
        idxs = joint_idxs[k]
        metric = joint_metrics[k]

        XY_n_neighbors = int(round(n_neighbors * len_Xs[k[1]]/sum(len_Xs) * len(idxs[1])/len_Xs[k[1]]))
        YX_n_neighbors = int(round(n_neighbors * len_Xs[k[0]]/sum(len_Xs) * len(idxs[0])/len_Xs[k[0]]))

        if XY_n_neighbors < 2:
            weights[(k[0],k[1])] *= XY_n_neighbors/2
            XY_n_neighbors = 2
        if YX_n_neighbors < 2:
            weights[(k[1],k[0])] *= YX_n_neighbors/2
            YX_n_neighbors = 2

        
        if metric == 'precomputed':
            XY_knn_indices = np.argsort(XY, axis=1)[:,XY_n_neighbors]
            XY_knn_dists = np.sort(XY, axis=1)[:,XY_n_neighbors]

            YX_knn_indices = np.argsort(XY.T, axis=1)[:,YX_n_neighbors]
            YX_knn_dists = np.sort(XY.T, axis=1)[:,YX_n_neighbors]

        else:
            XY_knn_indices, XY_knn_dists = partitioned_nearest_neighbors(XY[0], XY[1], 
                                                                         XY_n_neighbors, metric)
            YX_knn_indices, YX_knn_dists = partitioned_nearest_neighbors(XY[1], XY[0], 
                                                                         YX_n_neighbors, metric)

        XY_sigmas, XY_rhos = smooth_knn_dist(
            XY_knn_dists,
            0,
            local_connectivity=local_connectivity,
            cardinality=cardinality * XY_n_neighbors/n_neighbors
        )
        YX_sigmas, YX_rhos = smooth_knn_dist(
            YX_knn_dists,
            0,
            local_connectivity=local_connectivity,
            cardinality=cardinality * YX_n_neighbors/n_neighbors
        )

        XY_rows, XY_cols, XY_vals = compute_membership_strengths(
            XY_knn_indices, XY_knn_dists, XY_sigmas, XY_rhos
        )
        YX_rows, YX_cols, YX_vals = compute_membership_strengths(
            YX_knn_indices, YX_knn_dists, YX_sigmas, YX_rhos
        )

        rows = np.concatenate([rows, idxs[0][XY_rows] + sum(len_Xs[:k[0]])])
        cols = np.concatenate([cols, idxs[1][XY_cols] + sum(len_Xs[:k[1]])])
        vals = np.concatenate([vals, XY_vals])

        rows = np.concatenate([rows, idxs[1][YX_rows] + sum(len_Xs[:k[1]])])
        cols = np.concatenate([cols, idxs[0][YX_cols] + sum(len_Xs[:k[0]])])
        vals = np.concatenate([vals, YX_vals])

    fs = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(sum(len_Xs), sum(len_Xs))
    )
    fs.eliminate_zeros()

    transpose = fs.transpose()

    prod_matrix = fs.multiply(transpose)

    fs = (
        set_op_mix_ratio
        * (fs + transpose - prod_matrix)
        + (1.0 - set_op_mix_ratio) * prod_matrix
    )

    
    fs.sum_duplicates()
    fs.data[fs.data < (fs.data.max() / float(n_epochs))] = 0.0
    fs.eliminate_zeros()
    full_graph = fs

    graphs = []
    for i in range(len(Xs)):
        graphs += [fs[sum(len_Xs[:i]):sum(len_Xs[:i+1]), 
                      sum(len_Xs[:i]):sum(len_Xs[:i+1])].tocoo()]
    joint_graphs = {}
    for k in joint.keys():
        joint_graphs[k] = fs[sum(len_Xs[:k[0]]):sum(len_Xs[:k[0]+1]), 
                             sum(len_Xs[:k[1]]):sum(len_Xs[:k[1]+1])].tocoo()

    return graphs, joint_graphs, full_graph, weights

def init_layout(init, 
                Xs, 
                graphs, 
                n_components,
                metrics,
                metric_kwds,
                random_state):

    len_Xs = [len(i) for i in Xs]

    if init == 'random':
        embeddings = []
        for i in range(len(Xs)):
            embeddings += [np.random.RandomState(random_state).uniform(low=-10.0, high=10.0, 
                            size=(len_Xs[i], n_components),
                           ).astype(np.float32)]
    elif init == 'spectral':
        embeddings = []
        for i in range(len(Xs)):
            try:
                X_embedding = spectral_layout(
                    Xs[i],
                    graphs[i],
                    n_components,
                    np.random.RandomState(random_state),
                    metric=metrics[i],
                    metric_kwds=metric_kwds[i],
                )
                expansion = 10.0 / np.abs(X_embedding).max()
                X_embedding = (X_embedding * expansion).astype(np.float32) + \
                              np.random.RandomState(random_state).normal(scale=0.0001, 
                                                  size=[len_Xs[i], n_components]
                                                 ).astype(np.float32)
            except:
                X_embedding = np.random.RandomState(random_state).uniform(low=-10.0, high=10.0, 
                                                   size=(len_Xs[i], n_components),
                                                   ).astype(np.float32)
            embeddings += [X_embedding]
    else:
        if len(init.shape) == 2:
            if (np.unique(init, axis=0).shape[0] < init.shape[0]):
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:,1])
                embedding = init + np.random.RandomState(random_state).normal(
                    scale=0.001 * nndist,
                    size=init.shape
                ).astype(np.float32)
            else:
                embedding = init
            embeddings = []
            for i in range(len(Xs)):
                embeddings += [embedding[sum(len_Xs[:i]):sum(len_Xs[:i+1])]]

    for i in range(len(embeddings)):
        embeddings[i] = (10.0 * (embeddings[i] - np.min(embeddings[i], 0))
                         / (np.max(embeddings[i], 0) - np.min(embeddings[i], 0))
                        ).astype(np.float32, order="C")
    return embeddings


def optimize_layout(
    embeddings,
    graphs,
    joint_graphs,
    weights,
    n_epochs,
    a,
    b,
    random_state,
    gamma=1.0,
    initial_alpha=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
):


    len_Xs = np.array([len(i) for i in embeddings])
    dim = embeddings[0].shape[1]
    move_other = True
    alpha = initial_alpha

    heads = [i.row for i in graphs]
    tails = [i.col for i in graphs]
    n_vertices = [i.shape[1] for i in graphs]

    epochs_per_sample = [make_epochs_per_sample(i.data, n_epochs) for i in graphs]
    epochs_per_negative_sample = [i/negative_sample_rate for i in epochs_per_sample]
    epoch_of_next_negative_sample = [i.copy() for i in epochs_per_negative_sample]
    epoch_of_next_sample = [i.copy() for i in epochs_per_sample]

    joint_heads = {k: np.concatenate([joint_graphs[k].row, 
                                      joint_graphs[k].col + len_Xs[k[0]]]) for k in joint_graphs.keys()}
    joint_tails = {k: np.concatenate([joint_graphs[k].col + len_Xs[k[0]], 
                                      joint_graphs[k].row]) for k in joint_graphs.keys()}
    joint_n_vertices = {k: len_Xs[k[0]] + len_Xs[k[1]] for k in joint_graphs.keys()}
    joint_epochs_per_sample = {k: make_epochs_per_sample(
                                np.concatenate([joint_graphs[k].data, joint_graphs[k].data]), n_epochs) for k in joint_graphs.keys()}
    joint_epochs_per_negative_sample = {k: joint_epochs_per_sample[k]/negative_sample_rate for k in joint_graphs.keys()}
    joint_epoch_of_next_negative_sample = {k: np.copy(joint_epochs_per_negative_sample[k]) for k in joint_graphs.keys()}
    joint_epoch_of_next_sample = {k: np.copy(joint_epochs_per_sample[k]) for k in joint_graphs.keys()}



    optimize_fn = numba.njit(
        _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel
    )

    for n in range(n_epochs):

        for i in range(len(embeddings)):

            if weights[(i,i)] != 0:
                new_embedding = optimize_fn(
                    np.copy(embeddings[i]),
                    heads[i],
                    tails[i],
                    n_vertices[i],
                    epochs_per_sample[i],
                    a,
                    b,
                    np.random.RandomState(random_state).randint(np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3).astype(np.int64),
                    gamma,
                    dim,
                    move_other,
                    alpha,
                    epochs_per_negative_sample[i],
                    epoch_of_next_negative_sample[i],
                    epoch_of_next_sample[i],
                    n,
                )
                embeddings[i] += (new_embedding - embeddings[i]) * weights[(i,i)]

        for k in joint_graphs.keys():

            if weights[(k[0], k[1])] != 0 or weights[(k[1], k[0])] != 0:
                new_embeddings = optimize_fn(
                    np.concatenate([embeddings[k[0]], embeddings[k[1]]]),
                    joint_heads[k],
                    joint_tails[k],
                    joint_n_vertices[k],
                    joint_epochs_per_sample[k],
                    a,
                    b,
                    np.random.RandomState(random_state).randint(np.iinfo(np.int32).min + 1, np.iinfo(np.int32).max - 1, 3).astype(np.int64),
                    gamma,
                    dim,
                    move_other,
                    alpha,
                    joint_epochs_per_negative_sample[k],
                    joint_epoch_of_next_negative_sample[k],
                    joint_epoch_of_next_sample[k],
                    n,
                )

                embeddings[k[0]] += (new_embeddings[:len(embeddings[k[0]])] - embeddings[k[0]]) * weights[(k[0], k[1])]
                embeddings[k[1]] += (new_embeddings[len(embeddings[k[0]]):] - embeddings[k[1]]) * weights[(k[1], k[0])]

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return embeddings


def find_ab_params(spread, min_dist):


    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


def make_epochs_per_sample(weights, n_epochs):

    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result


def elaborate_relation_dict(dict, list_elems=True):
    new = {}
    for k in dict.keys():
        if len(k) == 2 and type(k[0]) != tuple and type(k[1]) != tuple:
            new[k] = dict[k]
        elif len(k) == 2:
            k_0 = k[0]
            k_1 = k[1]
            if type(k[0]) != tuple:
                k_0 = (k_0,)
            if type(k[1]) != tuple:
                k_1 = (k_1,)
            for i in range(len(k_0)):
                for j in range(len(k_1)):
                    if list_elems:
                        new[(k_0[i], k_1[j])] = [dict[k][0][i], dict[k][1][j]] 
                    else:
                        new[(k_0[i], k_1[j])] = dict[k]
        else:
            for i in range(len(k)):
                for j in range(i+1, len(k)):
                    if list_elems:
                        new[(k[i], k[j])] = [dict[k][i], dict[k][j]]
                    else:
                        new[(k[i], k[j])] = dict[k]
    return new

def find_weights(strengths, len_Xs, joint_idxs):

    if type(strengths) != dict:
        strengths = np.clip(strengths, 0, 1)
        weights = {}
        for i in range(len(len_Xs)):
            for j in range(len(len_Xs)):
                if i == j:
                    weights[(i,j)] = strengths[i]
                    
                else:
                    weights[(i,j)] = 1 - strengths[i]
    else:
        weights = elaborate_relation_dict(strengths, list_elems=False)
        for i in range(len(len_Xs)):
            for j in range(len(len_Xs)):
                if (i,j) not in weights.keys():
                    weights[(i,j)] = 1

    
    
    weight_sums = []
    for i in range(len(len_Xs)):
        weight_sum = 0
        for j in range(len(len_Xs)):
            weight_sum += weights[(i,j)] * len_Xs[j]
        weight_sums += [weight_sum]
    for i in range(len(len_Xs)):
        for j in range(len(len_Xs)):
            weights[(i,j)] *= sum(len_Xs) / weight_sums[i]

    
    for k in weights.keys():
        if k[0] != k[1]:
            if k in joint_idxs.keys():
                weights[k] *= len(joint_idxs[k][1])/len_Xs[k[1]]
            elif k[::-1] in joint_idxs.keys():
                weights[k] *= len(joint_idxs[k[::-1]][0])/len_Xs[k[1]]
            else:
                weights[k] = 0

    return weights

def MultiGraph(**kwds):
    return MultiMAP(**kwds, graph_only=True)

def MultiMAP(Xs,
             joint={},
             joint_idxs={},

             metrics=None,
             metric_kwds=None,
             joint_metrics={},

             n_neighbors=None,
             cardinality=None,
             angular=False,
             set_op_mix_ratio=1.0,
             local_connectivity=1.0,

             n_components=2,
             spread=1.0,
             min_dist=None,
             init='spectral',
             n_epochs=None,
             a=None,
             b=None,
             strengths=None,

             random_state=0,
             
             verbose=False,

             graph_only=False,
            ):
    '''
    Run MultiMAP on a collection of dimensionality reduction matrices. Returns a ``(parameters, 
    neighbor_graph, embedding)`` tuple, with the embedding optionally skipped if ``graph_only=True``.
    
    Input
    -----
    Xs : list of ``np.array``
        The dimensionality reductions of the datasets to integrate, observations as rows.
        
        >>> Xs = [DR_A, DR_B, DR_C]
    joint : dict of ``np.array``
        The joint dimensionality reductions generated for all pair combinations of the input 
        datasets. The keys are to be two-integer tuples, specifying the indices of the two
        datasets in ``Xs``
        
        >>> joint = {(0,1):DR_AB, (0,2):DR_AC, (1,2):DR_BC}
    graph_only : ``bool``, optional (default: ``False``)
        If ``True``, skip producing the embedding and only return the neighbour graph.
    
    All other arguments as described in ``MultiMAP.Integration()``.
    '''
    
    #turn off warnings if we're not verbose
    if not verbose:
        warnings.simplefilter('ignore')
    
    for i in range(len(Xs)):
        if not scipy.sparse.issparse(Xs[i]):
            Xs[i] = np.array(Xs[i])
    len_Xs = [len(i) for i in Xs]

    if not joint:
        joint = {tuple(range(len(Xs))): Xs}
    
    joint = elaborate_relation_dict(joint, list_elems=True)
    joint_idxs = elaborate_relation_dict(joint_idxs, list_elems=True)
    joint_metrics = elaborate_relation_dict(joint_metrics, list_elems=False)
    for k in joint.keys():
        joint[k] = [i.toarray() if scipy.sparse.issparse(i) else np.array(i) for i in joint[k]]   
        if k not in joint_idxs.keys():
            if k[::-1] in joint_idxs.keys():
                joint_idxs[k] = joint_idxs[k[::-1]]
            else:
                joint_idxs[k] = [np.arange(len_Xs[k[0]]), np.arange(len_Xs[k[1]])]
        if k not in joint_metrics.keys():
            if k[::-1] in joint_metrics.keys():
                joint_metrics[k] = joint_metrics[k[::-1]]
            else:
                joint_metrics[k] = 'euclidean'

    if metrics is None:
        metrics = ['euclidean' for i in range(len(Xs))]
    if metric_kwds is None:
        metric_kwds = [{} for i in range(len(Xs))]



    
    
    

    if n_neighbors is None:
        n_neighbors = 15 * len(Xs)
    if cardinality is None:
        cardinality = np.log2(n_neighbors)
    if min_dist is None:
        min_dist = 0.5 * 15/n_neighbors

    if scipy.sparse.issparse(init):
        init = init.toarray()
    else:
        init = np.array(init)
    if n_epochs is None:
        if np.sum(len_Xs) <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)

    if strengths is None:
        strengths = np.ones(len(Xs))*0.5
    weights = find_weights(strengths, len_Xs, joint_idxs)

    if verbose:
        print("Constructing fuzzy simplicial sets ...")
    graphs, joint_graphs, full_graph, weights = fuzzy_simplicial_set(
        Xs,
        joint,
        joint_idxs,
        weights,
        n_neighbors,
        cardinality,
        metrics,
        metric_kwds,
        joint_metrics,
        angular,
        set_op_mix_ratio,
        local_connectivity,
        n_epochs,
        random_state,
        verbose=False
    )

    #set up parameter output
    params = {'n_neighbors': n_neighbors,
              'metric': metrics[0],
              'multimap': {'cardinality': cardinality,
                           'set_op_mix_ratio': set_op_mix_ratio,
                           'local_connectivity': local_connectivity,
                           'n_components': n_components,
                           'spread': spread,
                           'min_dist': min_dist,
                           'init': init,
                           'n_epochs': n_epochs,
                           'a': a,
                           'b': b,
                           'strengths': strengths,
                           'random_state': random_state}}

    #return parameter and graph tuple
    #TODO: add the distances graph to this once it exists
    if graph_only:
        return (params, full_graph)

    if verbose:
        print("Initializing embedding ...")
    embeddings = init_layout(
        init, 
        Xs, 
        graphs, 
        n_components,  
        metrics,
        metric_kwds,
        random_state
    )

    if verbose:
        print("Optimizing embedding ...")
    embeddings = optimize_layout(
        embeddings,
        graphs,
        joint_graphs,
        weights,
        n_epochs,
        a,
        b,
        random_state,
        gamma=1.0,
        initial_alpha=1.0,
        negative_sample_rate=5.0,
        parallel=False,
        verbose=verbose
    )
    #undo warning reset
    if not verbose:
        warnings.resetwarnings()
    
    #return an embedding/graph/parameters tuple
    #TODO: add the distances graph to this once it exists
    return (params, full_graph, np.concatenate(embeddings))

import sklearn

def tfidf(X, n_components, binarize=True, random_state=0):
    from sklearn.feature_extraction.text import TfidfTransformer
    
    sc_count = np.copy(X)
    if binarize:
        sc_count = np.where(sc_count < 1, sc_count, 1)
    
    tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
    normed_count = tfidf.fit_transform(sc_count)

    lsi = sklearn.decomposition.TruncatedSVD(n_components=n_components, random_state=random_state)
    lsi_r = lsi.fit_transform(normed_count)
    
    X_lsi = lsi_r[:,1:]
    return X_lsi


import scipy
import numpy as np
import anndata
from ._sc_compat import pca_X as _pca_X, scale_X as _scale_X

def TFIDF_LSI(adata, n_comps=50, binarize=True, random_state=0):
	'''
	Computes LSI based on a TF-IDF transformation of the data. Putative dimensionality 
	reduction for scATAC-seq data prior to MultiMAP. Adds an ``.obsm['X_lsi']`` field to 
	the object it was ran on.
	
	Input
	-----
	adata : ``AnnData``
		The object to run TFIDF + LSI on. Will use ``.X`` as the input data.
	n_comps : ``int``
		The number of components to generate. Default: 50
	binarize : ``bool``
		Whether to binarize the data prior to the computation. Often done during scATAC-seq 
		processing. Default: True
	random_state : ``int``
		The seed to use for randon number generation. Default: 0
	'''
	
	#this is just a very basic wrapper for the non-adata function
	if scipy.sparse.issparse(adata.X):
		adata.obsm['X_lsi'] = tfidf(adata.X.todense(), n_components=n_comps, binarize=binarize, random_state=random_state)
	else:
		adata.obsm['X_lsi'] = tfidf(adata.X, n_components=n_comps, binarize=binarize, random_state=random_state)

def Wrapper(flagged, use_reps, embedding, seed, **kwargs):
	'''
	A function that computes the paired PCAs between the datasets to integrate, calls MultiMAP
	proper, and returns a  (parameters, connectivities, embedding) tuple. Embedding optional
	depending on ``embedding``.
	
	Input
	-----
	flagged : list of ``AnnData``
		Preprocessed objects to integrate. Need to have the single-dataset DRs computed at 
		this stage. Need to have ``.obs[\'multimap_index\']`` defined, incrementing integers
		matching the object's index in the list. Both ``Integrate()`` and ``Batch()`` make 
		these.
	
	All other arguments as described in ``MultiMAP.Integration()``.
	'''
	#MultiMAP wants the shared PCAs delivered as a dictionary, with the subset indices 
	#tupled up as a key. let's make that then
	joint = {}
	#process all dataset pairs
	for ind1 in np.arange(len(flagged)-1):
		for ind2 in np.arange(ind1+1, len(flagged)):
			subset = (ind1, ind2)
			#collapse into a single object and run a PCA
			adata = flagged[ind1].concatenate(flagged[ind2], join='inner')
			_pca_X(adata)
			#preserve space by deleting the intermediate object and just keeping its PCA
			#and multimap index thing
			X_pca = adata.obsm['X_pca'].copy()
			multimap_index = adata.obs['multimap_index'].values
			del adata
			#store the results in joint, which involves some further acrobatics
			joint[subset] = []
			#extract the coordinates for this particular element in the original list, using 
			#the multimap_index .obs column we created before. handy!
			for i in subset:
				joint[subset].append(X_pca[multimap_index == i, :])
	
	#with the joint prepped, we just need to extract the primary dimensionality reductions 
	#and we're good to go here
	Xs = []
	for adata, use_rep in zip(flagged, use_reps):
		Xs.append(adata.obsm[use_rep])
	
	#set seed
	np.random.seed(seed)
	
	#and with that, we're now truly free to call the MultiMAP function
	#need to negate embedding and provide that as graph_only for the function to understand
	mmp = MultiMAP(Xs=Xs, joint=joint, graph_only=(not embedding), **kwargs)
	
	#and that's it. spit this out for the other wrappers to use however
	return mmp

def Integration(adatas, use_reps, scale=True, embedding=True, seed=0, **kwargs):
	'''
	Run MultiMAP to integrate a number of AnnData objects from various multi-omics experiments
	into a single joint dimensionally reduced space. Returns a joint object with the resulting 
	embedding stored in ``.obsm[\'X_multimap\']`` (if instructed) and appropriate graphs in 
	``.obsp``. The final object will be a concatenation of the individual ones provided on 
	input, so in the interest of ease of exploration it is recommended to have non-scaled data 
	in ``.X``.
	
	Input
	-----
	adatas : list of ``AnnData``
		The objects to integrate. The ``.var`` spaces will be intersected across subsets of 
		the objects to compute shared PCAs, so make sure that you have ample features in 
		common between the objects. ``.X`` data will be used for computation.
	use_reps : list of ``str``
		The ``.obsm`` fields for each of the corresponding ``adatas`` to use as the 
		dimensionality reduction to represent the full feature space of the object. Needs 
		to be precomputed and present in the object at the time of calling the function.
	scale : ``bool``, optional (default: ``True``)
		Whether to scale the data to N(0,1) on a per-dataset basis prior to computing the 
		cross-dataset PCAs. Improves integration.
	embedding : ``bool``, optional (default: ``True``)
		Whether to compute the MultiMAP embedding. If ``False``, will just return the graph,
		which can be used to compute a regular UMAP. This can produce a manifold quicker,
		but at the cost of accuracy.
	n_neighbors : ``int`` or ``None``, optional (default: ``None``)
		The number of neighbours for each node (data point) in the MultiGraph. If ``None``, 
		defaults to 15 times the number of input datasets.
	n_components : ``int`` (default: 2)
		The number of dimensions of the MultiMAP embedding.
	seed : ``int`` (default: 0)
		RNG seed.
	strengths: ``list`` of ``float`` or ``None`` (default: ``None``)
		The relative contribution of each dataset to the layout of the embedding. The 
		higher the strength the higher the weighting of its cross entropy in the layout loss. 
		If provided, needs to be a list with one 0-1 value per dataset; if ``None``, defaults 
		to 0.5 for each dataset.
	cardinality : ``float`` or ``None``, optional (default: ``None``)
		The target sum of the connectivities of each neighbourhood in the MultiGraph. If 
		``None``, defaults to ``log2(n_neighbors)``.
	
	The following parameter definitions are sourced from UMAP 0.5.1:
	
	n_epochs : int (optional, default None)
		The number of training epochs to be used in optimizing the
		low dimensional embedding. Larger values result in more accurate
		embeddings. If None is specified a value will be selected based on
		the size of the input dataset (200 for large datasets, 500 for small).
	init : string (optional, default 'spectral')
		How to initialize the low dimensional embedding. Options are:
			* 'spectral': use a spectral embedding of the fuzzy 1-skeleton
			* 'random': assign initial embedding positions at random.
			* A numpy array of initial embedding positions.
	min_dist : float (optional, default 0.1)
		The effective minimum distance between embedded points. Smaller values
		will result in a more clustered/clumped embedding where nearby points
		on the manifold are drawn closer together, while larger values will
		result on a more even dispersal of points. The value should be set
		relative to the ``spread`` value, which determines the scale at which
		embedded points will be spread out.
	spread : float (optional, default 1.0)
		The effective scale of embedded points. In combination with ``min_dist``
		this determines how clustered/clumped the embedded points are.
	set_op_mix_ratio : float (optional, default 1.0)
		Interpolate between (fuzzy) union and intersection as the set operation
		used to combine local fuzzy simplicial sets to obtain a global fuzzy
		simplicial sets. Both fuzzy set operations use the product t-norm.
		The value of this parameter should be between 0.0 and 1.0; a value of
		1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
		intersection.
	local_connectivity : int (optional, default 1)
		The local connectivity required -- i.e. the number of nearest
		neighbors that should be assumed to be connected at a local level.
		The higher this value the more connected the manifold becomes
		locally. In practice this should be not more than the local intrinsic
		dimension of the manifold.
	a : float (optional, default None)
		More specific parameters controlling the embedding. If None these
		values are set automatically as determined by ``min_dist`` and
		``spread``.
	b : float (optional, default None)
		More specific parameters controlling the embedding. If None these
		values are set automatically as determined by ``min_dist`` and
		``spread``.
	'''
	
	#the main thing will be pulling out the various subsets of the adatas, sticking them 
	#together, running joint PCAs, and then splitting up the joint PCAs into datasets of 
	#origin. to do so, let's introduce a helper .obs column in copied versions of adatas
	flagged = []
	for i, adata in enumerate(adatas):
		flagged.append(adata.copy())
		#while we're at it, may as well potentially scale our data copy
		if scale:
			_scale_X(flagged[-1])
		flagged[-1].obs['multimap_index'] = i
	
	#call the wrapper. returns (params, connectivities, embedding), with embedding optional
	mmp = Wrapper(flagged=flagged, use_reps=use_reps, embedding=embedding, seed=seed, **kwargs)
	
	#make one happy collapsed object and shove the stuff in correct places
	#outer join to capture as much gene information as possible for annotation
	adata = anndata.concat(adatas, join='outer')
	if embedding:
		adata.obsm['X_multimap'] = mmp[2]
	#the graph is weighted, the higher the better, 1 best. sounds similar to connectivities
	#TODO: slot distances into .obsp['distances']
	adata.obsp['connectivities'] = mmp[1]
	#set up .uns['neighbors'], setting method to umap as these are connectivities
	adata.uns['neighbors'] = {}
	adata.uns['neighbors']['params'] = mmp[0]
	adata.uns['neighbors']['params']['method'] = 'umap'
	adata.uns['neighbors']['distances_key'] = 'distances'
	adata.uns['neighbors']['connectivities_key'] = 'connectivities'
	return adata

def Batch(adata, batch_key='batch', scale=True, embedding=True, seed=0, dimred_func=None, rep_name='X_pca', **kwargs):
	'''
	Run MultiMAP to correct batch effect within a single AnnData object. Loses the flexibility 
	of individualised dimensionality reduction choices, but doesn't require a list of separate 
	objects for each batch/dataset to integrate. Runs PCA on a per-batch/dataset basis prior 
	to performing an analysis analogous to  ``Integration()``. Adds appropriate ``.obsp`` graphs 
	and ``.obsm[\'X_multimap\']`` (if instructed) to the input.
	
	Input
	-----
	adata : ``AnnData``
		The object to process. ``.X`` data will be used in the computation.
	batch_key : ``str``, optional (default: "batch")
		The ``.obs`` column of the input object with the categorical variable defining the 
		batch/dataset grouping to integrate on.
	scale : ``bool``, optional (default: ``True``)
		Whether to scale the data to N(0,1) on a per-dataset basis prior to computing the 
		cross-dataset PCAs. Improves integration.
	embedding : ``bool``, optional (default: ``True``)
		Whether to compute the MultiMAP embedding. If ``False``, will just return the graph,
		which can be used to compute a regular UMAP. This can produce a manifold quicker,
		but at the cost of accuracy.
	dimred_func : function or ``None``, optional (default: ``None``)
		The function to use to compute dimensionality reduction on a per-dataset basis. Must 
		accept an ``AnnData`` on input and modify it by inserting its dimensionality reduction 
		into ``.obsm``. If ``None``, a scanpy-equivalent PCA on ``.X`` is used.
	rep_name : ``str``, optional (default: "X_pca")
		The ``.obsm`` field that the dimensionality reduction function stores its output under.
	
	All other arguments as described in ``Integration()``.
	'''
	
	#as promised in the docstring, set dimred_func to PCA if not provided
	if dimred_func is None:
		dimred_func = _pca_X
	
	#essentially what this function does is preps data to run through the other wrapper
	#so what needs to happen is the object needs to be partitioned up, have DR ran,
	#and passed as a list to the wrapper function
	flagged = []
	flagged_ids = []
	use_reps = []
	for i,batch in enumerate(np.unique(adata.obs[batch_key])):
		#extract the single batch data
		flagged.append(adata[adata.obs[batch_key]==batch].copy())
		#potentially scale
		if scale:
			_scale_X(flagged[-1])
		#and run DR
		dimred_func(flagged[-1])
		#and stick on the index for multimap to pull stuff apart later
		flagged[-1].obs['multimap_index'] = i
		#and add an entry to the list of .obsm keys for the other function
		use_reps.append(rep_name)
		#and store the cell name ordering for later
		flagged_ids = flagged_ids + list(flagged[-1].obs_names)
	
	#call the wrapper. returns (params, connectivities, embedding), with embedding optional
	mmp = Wrapper(flagged=flagged, use_reps=use_reps, embedding=embedding, seed=seed, **kwargs)
	
	#this output has the cells ordered as a concatenation of the individual flagged objects
	#so need to figure out how to reorder the output to get the original cell order
	#doing the following operation sets the desired order to adata.obs_names
	#and checks the index for each in flagged_ids
	#so taking something in flagged_ids order and using sort_order on it will match obs_names
	sort_order = [flagged_ids.index(i) for i in list(adata.obs_names)]
	
	#stick stuff where it's supposed to go
	if embedding:
		adata.obsm['X_multimap'] = mmp[2][sort_order,:]
	#the graph is weighted, the higher the better, 1 best. sounds similar to connectivities
	#TODO: slot distances into .obsp['distances']
	adata.obsp['connectivities'] = mmp[1][sort_order,:][:,sort_order]
	#set up .uns['neighbors'], setting method to umap as these are connectivities
	adata.uns['neighbors'] = {}
	adata.uns['neighbors']['params'] = mmp[0]
	adata.uns['neighbors']['params']['method'] = 'umap'
	adata.uns['neighbors']['distances_key'] = 'distances'
	adata.uns['neighbors']['connectivities_key'] = 'connectivities'