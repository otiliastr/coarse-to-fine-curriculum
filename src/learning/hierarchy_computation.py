import logging
import numpy as np

from collections import Counter

from .trainer import create_predict_dataset
from ..utils.clustering import affinity_clustering
from ..utils.printing import print_metrics_dict

__author__ = 'Otilia Stretcu'


def compute_dist_matrix(load_dist_matrix, path_dist_matrix, trainer, data, num_epochs,
                        batch_size, summary_dir=None, load_from_checkpoint=False,
                        checkpoints_dir=None):
    """Compute distance matrix between labels.

    This matrix can be loaded from file or computed using a baseline model.
    The baseline model can be loaded from a checkpoint.
    If neither a distance matrix is provided, nor a model checkpoint, then a model will be trained
    from scratch using the provided trainer, and used to compute the confusion matrix on the
    validation set.

    Args:
        load_dist_matrix: Boolean specifying whether to load the distance matrix from disk.
        path_dist_matrix: String representing the path where to load the distance matrix from,
            which is used only if `load_dist_matrix` is True.
        trainer: A Trainer object.
        data: A Dataset object.
        num_epochs: Number of epochs to train for in order to compute the confusion matrix.
            batch_size: Batch size used during training.
        summary_dir: A string representing directory where to save TensorFlow summaries for the
            trained model. If None, no strings are saved.
        load_from_checkpoint: A boolean specifying whether to load the model from a checkpoint.
        checkpoints_dir: String representing the path to a model checkpoint file which is used
            if `load_from_checkpoint` is True.

    Returns:
        dist_matrix: A numpy array of shape (num_classes, num_classes) representing the distance
            matrix between labels.
        test_metric: A value representing the test metric (e.g., accuracy) of the trained model
            evaluated on the test set.
        other_metrics: A dictionary of evaluation results of the trained model
            evaluated on the test set.
    """

    if load_dist_matrix:
        logging.info('Loading similarity matrix from %s...', path_dist_matrix)
        dist_matrix = np.load(path_dist_matrix)
        test_metric = None
        other_metrics = None
    else:
        if load_from_checkpoint:
            logging.info('Loading baseline model from the checkpoint at: %s',
                         checkpoints_dir)
            trainer.restore(checkpoints_dir, data=data)
            test_metric, other_metrics = trainer.test(data, batch_size, data.get_indices_test())
        else:
            logging.info('-' * 100)
            logging.info('Training original model on %d classes...',
                         data.num_classes)
            test_metric, other_metrics = trainer.train(
                data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                checkpoints_dir=checkpoints_dir,
                summary_dir=summary_dir+'-original' if summary_dir else None)
        logging.info('Test %s for loaded model: %.2f', trainer.val_metric, test_metric)
        print_metrics_dict(other_metrics)

        # Compute pair-wise similarities between classes.
        logging.info('Computing class similarity matrix...')
        dist_matrix = 1.0 - compute_confusion_matrix(data, trainer)
        if path_dist_matrix:
            np.save(path_dist_matrix, dist_matrix)

    return dist_matrix, test_metric, other_metrics


def compute_confusion_matrix(data, trainer, batch_size=32):
    """Compute the confusion matrix of the model stored in `trainer` on the validation data.

    Args:
        data: A Dataset object.
        trainer: A Trainer object.
        batch_size: Batch size.

    Returns:
        A numpy array of shape (num_classes, num_classes) representing the distance matrix
            between labels.
    """
    # Make predictions for validation samples.
    indices = data.get_indices_val()
    dataset = create_predict_dataset(data, batch_size, indices)
    predictions = [trainer.predict(inputs_batch, normalize=True) for inputs_batch in dataset]
    predictions = np.concatenate(predictions)

    # For each label, compute all other labels that it is confused with when the model
    # makes a mistake.
    dist_matrix = np.zeros((data.num_classes, data.num_classes))
    labels = data.get_labels(indices)
    for label in range(data.num_classes):
        indices_label = np.where(labels == label)[0]
        predictions_for_label = predictions[indices_label]
        predicted_wrong_labels = np.argmax(predictions_for_label, axis=1)
        # Compute how many times the label is confused with each other label.
        total = float(predicted_wrong_labels.shape[0])
        predicted_wrong_labels = Counter(predicted_wrong_labels)
        # Convert these counts to ratios, such that each row of the confusion matrix sums to 1.
        for l, c in predicted_wrong_labels.items():
            dist_matrix[label][l] = c / total
    return dist_matrix


def compute_hierarchy(dist_matrix, make_symmetric=True, eps=1e-7):
    """Compute hierarchy using a distance matrix.

    Args:
        dist_matrix: A numpy array of shape (k, k) for some integer k, representing a distance
            matrix. It needs not be symmetric.
        make_symmetric: Boolean specifying whether to make the distance matrix symmetric.
        eps: Small float value to be added to all distances in order to break ties.

    Returns:
        A list of lists containing the label clusters per level. Each outer list corresponds to a
        level in the hierarchy. For each level, each inner list contains the label ids that are
        clustered together at this level. Levels are indexed from top to bottom.
        E.g. for following hierarchy containing k=5 labels
                           c
                        /     \
                       a      b
                     / | \   /\
                    1  2 3  4 5
        the returned list of lists will be:
        [[[1, 2, 3], [4, 5]], [[1], [2], [3], [4], [5]]
        Note that the highest level of the hierarchy which combines all labels [1, 2, 3, 4, 5]
        has been removed, since we do not need it for training the curriculum (this would
        correspond to all labels being in the same class).
    """
    # Make the distance matrix symmetrical.
    if dist_matrix is not None:
        if make_symmetric:
            dist_matrix = dist_matrix + dist_matrix.T

    print('-------------- Final distance matrix used for clustering -----------------------')
    print(np.array2string(dist_matrix, separator=', ', precision=2))

    clusters_per_level = affinity_clustering(dist_matrix, eps=eps)
    # Reverse the order of the levels, so it goes from coarse clusters to fine grained.
    clusters_per_level = clusters_per_level[::-1]
    # Remove the first level, which merges all labels in 1 cluster.
    clusters_per_level = clusters_per_level[1:]
    return clusters_per_level

