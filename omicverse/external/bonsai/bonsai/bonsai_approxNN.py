from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
# import faiss
# from bonsai.bonsai_helpers import time_format
# from icecream import ic

# ic.configureOutput(prefix=time_format)


def getNNsBruteForce(points, k=50):
    """
    Code to run this NN-search:
    nns = getNNsBruteForce(points, k=50)
    where points is genes x cells
    """
    # similarities = cosine_distances(points.T)
    similarities = euclidean_distances(points.T)
    knn = np.argsort(similarities, axis=0)[:k]
    return knn.T


def getNNssklearn(pointsT, index=None, k=50, metric='cosine', addPoints=True):
    """
    Code to run this NN-search:
    index, nns = getNNssklearn(points, k=50)
    where points is genes x cells
    This function does an exhaustive search, but does build an index. If an index is given as an argument, the
    data in points is queried with respect to this index. If no index is given, then the points will be used to
    build an index and query the points themselves.
    If index is given as argument, this index is not updated with the new points.
    Metric can currently be 'cosine' or 'euclidean'.
    """
    toBeAdded = True
    if index is None:
        index = NearestNeighbors(n_neighbors=k, metric=metric)
        index.fit(pointsT)
        toBeAdded = False

    # select indices of k nearest neighbors of the vectors in the input list
    if toBeAdded and addPoints:
        # Add new points to index
        indexedPointsT = np.vstack((getattr(index, '_fit_X'), pointsT))
        index.fit(indexedPointsT)
    n_neighbors = min(k, index.n_samples_fit_ - 1)
    neighbors = index.kneighbors(pointsT, n_neighbors=n_neighbors, return_distance=False)

    return index, neighbors


def removeNNssklearn(index, inds):
    pointsT = np.delete(getattr(index, '_fit_X'), inds, axis=0)
    index.fit(pointsT)
    if hasattr(index, 'IDs'):
        index.IDs = np.delete(index.IDs, inds)


# def getNNfaissbruteforce(pointsT, index=None, k=50, metric='cosine', addPoints=True):
#     """
#     Code to run this NN-search:
#     index, nns = getNNfaissbruteforce(points, k=50)
#     where pointsT is cells x genes
#     This function does an exhaustive search, but does build an index. If an index is given as an argument, the
#     data in points is queried with respect to this index. If no index is given, then the points will be used to
#     build an index and query the points themselves.
#     If index is given as argument, this index is updated with the new points.
#     """
#     f = pointsT.shape[1]
#     if index is None:
#         addPoints = True
#         if metric == 'cosine':
#             index = faiss.IndexFlatIP(f)  # Inner product metric (equivalent but faster on normalized vectors)
#         else:
#             index = faiss.IndexFlatL2(f)  # build the index, f=size of vectors
#     if addPoints:
#         index.add(pointsT)  # add vectors to the index
#
#     D, nns = index.search(pointsT, k)  # actual search
#     return index, nns


def removeVect(index, inds):
    if index.nnType == 'faissBrute':
        removeFaiss(index, inds)
    elif index.nnType == 'faissLSH':
        print("Removing a vector from LSH-index is not supported.")
    else:
        removeNNssklearn(index, inds)


def removeFaiss(index, inds):
    index.remove_ids(inds)
    if hasattr(index, 'IDs'):
        index.IDs = np.delete(index.IDs, inds)


# def getNNfaissIVF(pointsT, index=None, k=50, nlist_factor=10, nprobe=5, metric='cosine'):
#     """
#     Code to run this NN-search:
#     index, nns = getNNfaissbruteforce(points, k=50)
#     where points is genes x cells
#     This function does an exhaustive search, but does build an index. If an index is given as an argument, the
#     data in points is queried with respect to this index. If no index is given, then the points will be used to
#     build an index and query the points themselves.
#     If index is given as argument, this index is updated with the new points.
#     """
#     c, f = pointsT.shape
#     if index is None:
#         if metric == 'cosine':
#             quantizer = faiss.IndexFlatIP(f)
#         else:
#             quantizer = faiss.IndexFlatL2(f)  # the other index
#         nlist = int(nlist_factor * np.sqrt(c))
#         index = faiss.IndexIVFFlat(quantizer, f, nlist)
#         index.train(pointsT)  # add vectors to the index
#
#     index.add(pointsT)
#     index.nprobe = nprobe
#     D, nns = index.search(pointsT, k)  # actual search
#     return index, nns


# def getNNsfaissLSH(points, index=None, k=50, n_bits=128, addPoints=True):
#     pointsT = np.ascontiguousarray(np.float32(points.T))
#     f = pointsT.shape[1]
#     if index is None:
#         addPoints = True
#         index = faiss.IndexLSH(f, n_bits)
#         index.train(pointsT)
#     if addPoints:
#         index.add(pointsT)
#     D, nns = index.search(pointsT, k)
#     return index, nns


def getFracCorrectNNs(nns, true_nns):
    n_correct = 0
    for cell in range(nns.shape[1]):
        n_correct += np.sum((nns - true_nns[:, cell][:, None]) == 0)
    correctFrac = n_correct / np.prod(nns.shape)
    return correctFrac


def getApproxNNs(points, index=None, k=50, n_bits_factor=10, metric='cosine', pointsIds=None, addPoints=True, th1=150000, th2=150000):
    """

    :param points: (genes(features) x cells)-matrix for which NNs are required. If index=None, points is used to both
    build the index and then to get NNs between these points. Otherwise, the existing index is used, the new "points"
    are added, and the NNs for these new points are returned.
    :param index: trained index that was returned by an earlier run of this function
    :param k: number of nearest neighbours required
    :param n_bits_factor: Is multiplied with log(nCells) to get how many projections are used in LSH (large datasets)
    :param metric: What metric to use for neighbour estimation, currently 'cosine' and 'euclidean' are supported.
    :return index: index where all points are added
    :return nns: (cells x k)-matrix with (approximate) nearest neighbours for all cells given in "points"
    """
    if index is not None:
        nnType = index.nnType
        k = min(index.n_samples_fit_, k)
    else:  # nCells < th1:
        nnType = 'sklearn'
    # elif nCells < th2:
    #     nnType = 'faissBrute'
    # else:
    #     nnType = 'faissLSH'

    # We always require the transpose of the points-matrix and it should often be a contiguous array
    if nnType != 'sklearn':
        pointsT = np.ascontiguousarray(np.float32(points.T))
        # TODO: Try replacing faissLSH by Annoy
    else:
        pointsT = points.T
    nCells, nGenes = pointsT.shape

    # if (metric == 'cosine') and nnType in ['faissBrute', 'faissLSH']:
    #     # In this case we just normalize all vectors, then use Euclidean distance. This is equivalent.
    #     faiss.normalize_L2(pointsT)

    if nnType == 'sklearn':
        index, nns = getNNssklearn(pointsT, k=k, index=index, metric=metric, addPoints=addPoints)
        index.nnType = 'sklearn'
    # elif nnType == 'faissBrute':
    #     index, nns = getNNfaissbruteforce(pointsT, index=index, k=k, metric=metric, addPoints=addPoints)
    #     index.nnType = 'faissBrute'
    # else:
    #     # This point is only reached when we have more than 10000 cells. In that case we do an approximate nn-search
    #     n_bits = int(n_bits_factor * np.sqrt(nCells))
    #     index, nns = getNNsfaissLSH(points, index=index, k=k, n_bits=n_bits, addPoints=addPoints)
    #     index.nnType = 'faissLSH'

    # Update Ids in index by adding new points
    if addPoints and (pointsIds is not None):
        if hasattr(index, 'IDs'):
            index.IDs = np.hstack((index.IDs, pointsIds))
        else:
            index.IDs = pointsIds

    return index, nns


if __name__ == '__main__':
    # Load data
    f = 2000
    n = 20000
    np.random.seed(42)
    points = np.zeros((f, n))
    points[:, 0] = np.random.normal(loc=0.0, scale=1.0, size=f)
    for ind in range(1, n):
        points[:, ind] = points[:, ind - 1] + np.random.normal(loc=0.0, scale=0.1, size=f)

    points += np.random.normal(loc=0.0, scale=.1, size=(f, n))
    points -= np.mean(points, axis=1)[:, None]
    points = normalize(points, axis=0)

    K = 50

    # start = time.time()
    # nns = true_nns = getNNsBruteForce(points, k=K)
    # correctFrac = getFracCorrectNNs(nns, true_nns)
    # print("Brute force similarity search took %f seconds, got %f correct." % (time.time() - start, correctFrac))
    # ic(nns)
    #

    """sklearn also using brute force"""
    start = time.time()
    index, nns = getNNssklearn(points, k=K)
    true_nns = nns
    end = time.time()
    correctFrac = getFracCorrectNNs(nns, true_nns)
    print("Brute force sklearn search took %f seconds, got %f correct." % (end - start, correctFrac))
    ic(nns)

    start = time.time()
    index, nns = getNNfaissbruteforce(points, k=K)
    end = time.time()
    correctFrac = getFracCorrectNNs(nns, true_nns)
    print("Brute force faiss search took %f seconds, got %f correct." % (end - start, correctFrac))
    ic(nns)

    start = time.time()
    nprobe = 5
    index, nns = getNNfaissIVF(points, nlist=100, k=K, nprobe=nprobe)
    end = time.time()
    correctFrac = getFracCorrectNNs(nns, true_nns)
    print("IVF faiss search with %d probes took %f seconds, got %f correct." % (nprobe, end - start, correctFrac))
    ic(nns)

    start = time.time()
    n_bits = int(2 * np.log(n))
    index, nns, getNNsfaissLSH(points, n_bits=n_bits, k=K)
    end = time.time()
    correctFrac = getFracCorrectNNs(nns, true_nns)
    print("LSH forest faiss search with %d bits took %f seconds, got %f correct." % (n_bits, end - start, correctFrac))
    ic(nns)
