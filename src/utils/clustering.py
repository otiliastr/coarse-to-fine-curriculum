import logging
import numpy as np

__author__ = 'Otilia Stretcu'


def _find_closest_cluster_min(dist_matrix, clusters, node_to_cluster):
    n = dist_matrix.shape[0]
    closest_cluster = []
    closest_cluster_dists = []
    # For each cluster, find the edge of min weight going out of it.
    for cluster in clusters:
        cluster_nodes = np.asarray(cluster)
        # Create a mask that marks the nodes in the current cluster.
        cluster_mask = np.zeros((n,), dtype=np.bool)
        cluster_mask[cluster_nodes] = True
        non_cluster_mask = np.logical_not(cluster_mask)
        non_cluster_nodes = np.where(non_cluster_mask)[0]
        # Only consider the edges from nodes belonging to the current
        # cluster and nodes from a different cluster.
        dists = dist_matrix[cluster_mask][:, non_cluster_mask]
        # Find the min of these distances and the node from a different
        # cluster this connects to.
        if len(dists.shape) < 2:
            if len(cluster_nodes) == 1:
                dists = dists[None]
            else:
                dists = dists[:, None]
        min_dists = np.amin(dists, axis=1)
        cluster_node_id = np.argmin(min_dists)
        non_cluster_node_id = np.argmin(dists[cluster_node_id])
        smallest_dist = dists[cluster_node_id][non_cluster_node_id]
        non_cluster_node_id = non_cluster_nodes[non_cluster_node_id]
        # Find the cluster of this closest node.
        closest_cluster.append(node_to_cluster[non_cluster_node_id])
        closest_cluster_dists.append(smallest_dist)

    return closest_cluster, closest_cluster_dists


def find_closest_cluster(dist_matrix, clusters):
    """Finds the closest cluster to each cluster.

    Arguments:
        dist_matrix: A numpy array representing a distance matrix between all pairs of nodes.
        clusters: A list of lists, where each inner list contains the node ids that are in that
            cluster.
    Returns:
        closest_cluster
        closest_cluster_dists
    """
    # Compute the number of nodes.
    n = dist_matrix.shape[0]
    # Map each node id to its cluster.
    node_to_cluster = np.zeros((n,), dtype=np.int)
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = cluster_id

    # For each cluster, find the nearest cluster to it, where nearest is
    # defined depending on the linkage method.
    closest_cluster, closest_cluster_dists = _find_closest_cluster_min(
        dist_matrix, clusters, node_to_cluster)

    return closest_cluster, closest_cluster_dists


def merge_clusters(clusters, closest_cluster, num_clusters, last_cluster_id):
    # Merge each cluster with the one closest to it. This will lead to a
    # list of cluster sets.
    color = ['not_visited' for _ in range(num_clusters)]
    cluster_to_set = np.zeros((num_clusters,), dtype=np.int)
    cluster_sets = []
    cluster_set_to_id = []
    for i in range(num_clusters):
        if color[i] == 'not_visited':
            current_set = {i}
            color[i] = 'in_progress'
            j = i
            while closest_cluster[j] is not None and color[closest_cluster[j]] == 'not_visited':
                current_set.add(closest_cluster[j])
                color[closest_cluster[j]] = 'in_progress'
                j = closest_cluster[j]
            # If the cluster where we stopped doesn't have a closest cluster or it was visited
            # during the current traversal starting at i, then we form a new set.
            if closest_cluster[j] is None or color[closest_cluster[j]] == 'in_progress':
                # This a new set of clusters.
                set_index = len(cluster_sets)
                cluster_sets.append(current_set)
                last_cluster_id += 1
                cluster_set_to_id.append(last_cluster_id)
            else:
                # This has to be added to an existing set of clusters.
                set_index = cluster_to_set[closest_cluster[j]]
                cluster_sets[set_index].update(current_set)

            for j in list(current_set):
                color[j] = 'visited'
                cluster_to_set[j] = set_index
    # Expand the cluster sets to actual clusters.)
    clusters = [
        [elem for cluster_index in list(c)
         for elem in clusters[cluster_index]]
        for c in cluster_sets]
    return clusters


def affinity_clustering(dist_matrix, eps=1e-7):
    """Performs affinity clustering based on the provided distance matrix.

    Our implementation is based on a publication of Bateni et al., "Affinity clustering:
    Hierarchical clustering at scale." NeurIPS, 2017. Specifically, it is based on the authors'
    description of the algorithm, not on their released map-reduce-based code.

    Args:
        dist_matrix: A numpy array of shape (num_classes, num_classes) representing the distance
            matrix between labels.
        eps: A small float value representing the standard deviation of Gaussian noise added to
            all distances to break ties.
    Returns:
        A list of lists containing the label clusters per level. Each outer list corresponds to a
        level in the hierarchy. For each level, each inner list contains the label ids that are
        clustered together at this level. Levels are indexed from bottom to top.
        E.g., for the following hierarchy containing k=5 labels:
                           c
                        /     \
                       a      b
                     / | \   /\
                    1  2 3  4 5
        the returned list of lists will be:
        [[1, 2, 3, 4, 5], [[1, 2, 3], [4, 5]], [[1], [2], [3], [4], [5]].
    """
    # Add a small value eps to all edges to break ties.
    num_elem = dist_matrix.shape[0]
    dist_matrix = dist_matrix + np.random.rand(num_elem, num_elem) * eps

    # Num samples.
    n = dist_matrix.shape[0]
    # Each sample is it's own cluster in the beginning.
    clusters = [[i] for i in range(n)]
    # Keep track of clusters formed as we go.
    last_cluster_id = len(clusters) - 1

    # Merge clusters iteratively.
    num_clusters = len(clusters)
    clusters_per_level = [clusters]
    level = 0
    while num_clusters > 1:
        # For each cluster, find the cluster that is nearest to it.
        closest_cluster, closest_cluster_dists = find_closest_cluster(dist_matrix, clusters)

        # Merge each cluster with the one closest to it.
        new_clusters = merge_clusters(clusters, closest_cluster, num_clusters, last_cluster_id)

        # If everything was merged in the first iteration, we remove the most expensive edge.
        if level == 0:
            while len(new_clusters) == 1:
                logging.info('All nodes were merged in the first iteration. Removing the most '
                             'expensive edge and trying again.')
                idx = np.argmax(closest_cluster_dists)
                closest_cluster[idx] = None
                closest_cluster_dists[idx] = -np.inf
                new_clusters = merge_clusters(clusters, closest_cluster, num_clusters,
                                              last_cluster_id)

        num_clusters = len(new_clusters)
        clusters = new_clusters
        clusters_per_level.append(clusters)
        level += 1

    return clusters_per_level
