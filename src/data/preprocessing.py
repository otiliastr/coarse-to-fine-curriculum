import logging
import numpy as np

from collections import Counter

__author__ = 'Otilia Stretcu'


###############################################################################################
# Functions for splitting data.                                                               #
###############################################################################################

def select_num_samples_per_cls_indices(num_samples_per_cls, labels, num_cls=None):
    logging.info('Selecting a subset of %d samples per class...', num_samples_per_cls)

    # Identify number of unique classes.
    if num_cls is None:
        unique_cls = np.unique(labels)
        num_cls = len(unique_cls)

    # Group together samples IDs from same class.
    indices_per_cls = [[] for _ in range(num_cls)]
    for i in range(labels.shape[0]):
        indices_per_cls[labels[i]].append(i)

    # Select num_samples_per_cls at random from  each class.
    selected_indices = []
    remaining_indices = []
    for l in range(num_cls):
        ind = np.asarray(indices_per_cls[l])
        np.random.shuffle(ind)
        assert ind.shape[0] >= num_samples_per_cls, \
            'Not all classes have enough samples to satisfy the requested # samples per class.'
        selected_indices.append(ind[:num_samples_per_cls])
        remaining_indices.append(ind[num_samples_per_cls:])
    selected_indices = np.concatenate(selected_indices)
    remaining_indices = np.concatenate(remaining_indices)
    logging.info('Selected a random subset of %d samples.' % len(selected_indices))
    return selected_indices, remaining_indices


def select_num_samples_per_cls(num_samples_per_cls, inputs, labels, num_cls=None):
    selected_indices, remaining_indices = select_num_samples_per_cls_indices(
        num_samples_per_cls, labels, num_cls=num_cls)
    remaining_inputs = inputs[remaining_indices]
    inputs = inputs[selected_indices]
    remaining_labels = labels[remaining_indices]
    labels = labels[selected_indices]
    print('Remaining num labels per class:', Counter(remaining_labels))
    return inputs, labels, remaining_inputs, remaining_labels


def split_train_val(inputs_train,
                    labels_train=None,
                    target_num_train_per_class=None,
                    ratio_val=None):
    assert (target_num_train_per_class is None) ^ (ratio_val is None), \
        "Either target_num_train_per_class must be None, or target_num_val must be None."
    if ratio_val is not None:
        return split_train_val_given_ratio_val(inputs_train, labels_train, ratio_val)
    return split_train_val_given_train_per_class(inputs_train, labels_train,
                                                 target_num_train_per_class)


def split_train_val_given_ratio_val(train_inputs, train_labels, ratio_val):
    num_train = len(train_inputs)
    num_val = int(num_train * ratio_val)

    # Split the train samples into train and validation.
    ind = np.arange(0, num_train)
    np.random.shuffle(ind)
    ind_val = ind[:num_val]
    ind_train = ind[num_val:]
    if isinstance(train_inputs, list):
        val_inputs = [train_inputs[i] for i in ind_val]
        train_inputs = [train_inputs[i] for i in ind_train]
    else:
        val_inputs = train_inputs[ind_val]
        train_inputs = train_inputs[ind_train]
    if train_labels is not None:
        val_labels = train_labels[ind_val]
        train_labels = train_labels[ind_train]
        return train_inputs, train_labels, val_inputs, val_labels
    return train_inputs, val_inputs


def split_train_val_given_ratio_val_indices(num_samples, ratio_val, seed=None):
    rng = np.random.RandomState(seed=seed)
    num_val = int(num_samples * ratio_val)
    # Split the train samples into train and validation.
    ind = np.arange(0, num_samples)
    rng.shuffle(ind)
    ind_val = ind[:num_val]
    ind_train = ind[num_val:]
    return ind_train, ind_val


def split_train_val_given_train_per_class(train_images, train_labels, target_num_train_per_class):
    num_classes = max(train_labels) + 1

    # Select `target_num_train_per_class` samples from each class for training.
    ind_train = []
    for i in range(num_classes):
        ind_class = np.where(train_labels == i)[0]
        assert len(ind_class) >= target_num_train_per_class, \
            'Not enough labels for class %d to select %d labels per class.' % (i, target_num_train_per_class)
        np.random.shuffle(ind_class)
        selected_ind = ind_class[:target_num_train_per_class]
        ind_train.extend(selected_ind)

    # Having selected the train indices, the remaining are validation.
    num_train = len(ind_train)
    ind_train = np.asarray(ind_train)
    ind_train_set = set(ind_train)
    ind_val = [i for i in range(num_train) if i not in ind_train_set]
    ind_val = np.asarray(ind_val)
    val_images = train_images[ind_val]
    val_labels = train_labels[ind_val]
    train_images = train_images[ind_train]
    train_labels = train_labels[ind_train]

    return train_images, train_labels, val_images, val_labels


###############################################################################################
# Functions for preprocessing data.                                                           #
###############################################################################################

def get_feature_preproc_fn(online_preprocessing, **kwargs):
    if online_preprocessing == 'norm':
        return convert_pixels
    return no_preproc


def no_preproc(x, **kwargs):
    """Identity function that can be passed used as feature_preproc_fn in Dataset, since lambda is
    not pickleable and doesn't take an arbitrary number of params."""
    return x


def convert_pixels(image, **kwargs):
    """Converts an image containing pixel values in [0, 255] to [0, 1] floats."""
    image = image.astype('float32')
    image /= 255.0
    return image


def apply_normalization(train_inputs, test_inputs, normalization=None, val_inputs=None):
    if normalization is None:
        if val_inputs is not None:
            return train_inputs, test_inputs, val_inputs
        return train_inputs, test_inputs
    normalizations = normalization.split(',')
    for normalization in normalizations:
        if normalization == 'center':
            train_inputs = np.float32(train_inputs / 255.0)
            test_inputs = np.float32(test_inputs / 255.0)
            if val_inputs is not None:
                val_inputs = np.float32(val_inputs / 255.0)
            # Subtract pixel mean.
            train_inputs_mean = np.mean(train_inputs, axis=0)
            train_inputs -= train_inputs_mean
            if val_inputs is not None:
                val_inputs -= train_inputs_mean
            test_inputs -= train_inputs_mean
        elif normalization == 'norm':
            train_inputs = convert_pixels(train_inputs)
            test_inputs = convert_pixels(test_inputs)
            if val_inputs is not None:
                val_inputs = convert_pixels(val_inputs)
        else:
            raise ValueError('Unsupported nomalization type `%s`. Valid options '
                             'are: None, `norm`, `center`.' % normalization)
    if val_inputs is not None:
        return train_inputs, test_inputs, val_inputs
    return train_inputs, test_inputs
