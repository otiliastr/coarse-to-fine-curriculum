import logging
import numpy as np
import tensorflow_datasets as tfds

from .dataset import Dataset
from .preprocessing import apply_normalization, get_feature_preproc_fn, select_num_samples_per_cls,\
    split_train_val_given_ratio_val

__author__ = 'Otilia Stretcu'

CLASS_NAMES = {
    'mnist': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'svhn_cropped': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    'cifar10': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                'horse', 'ship', 'truck'],
    'cifar100': [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'],
    'fashion_mnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
}


def load_data(dataset_name, ratio_val, normalization=None, ratio_test=None,
              max_samples=None, data_path=None):
    """Loads the dataset with the provided name.

    Args:
        dataset_name: A string representing the dataset name.
        ratio_val: A float number between [0, 1] representing the ratio of the training data to
            set aside for validation.
        normalization: A string representing the type of data normalization to perform.
        ratio_test: A float number between [0, 1] representing the ratio of the training data to
            set aside for test, in case a test set is not provided.
        max_samples:  Maximum number of train examples allowed. If a validation set is not provided,
            the validation set is split out from the max_samples examples.
        data_path: String representing a

    Returns:

    """
    if dataset_name == 'tiny_imagenet':
        from .tiny_imagenet import load_tiny_imagenet
        data = load_tiny_imagenet(
            data_path=data_path,
            ratio_val=ratio_val,
            normalization=normalization,
            max_samples=max_samples)
    elif dataset_name == 'shapes':
        from .shapes import load_shapes
        data = load_shapes(
            dataset_name=dataset_name,
            path=data_path,
            normalization=normalization,
            ratio_test=ratio_test,
            ratio_val=ratio_val,
            max_samples=max_samples)
    else:
        data = load_data_tf_datasets(
            dataset_name,
            normalization=normalization,
            ratio_val=ratio_val,
            ratio_test=ratio_test,
            max_samples=max_samples,
            coarse_labels=dataset_name == 'cifar100-coarse')

    return data


def load_data_tf_datasets(dataset_name, normalization=None, ratio_val=None, ratio_test=None,
                          max_samples=None, coarse_labels=False, online_preprocessing=None):
    if dataset_name == 'svhn':
        dataset_name += '_cropped'
    # Load train data.
    data = tfds.load(dataset_name, batch_size=-1)
    data_subset = tfds.as_numpy(data['train'])
    train_inputs = data_subset['image']
    train_labels = data_subset['coarse_label'] if coarse_labels else data_subset['label']
    # Remove dimension of size 1 from the labels.
    train_labels = np.squeeze(train_labels)

    if max_samples is not None and train_inputs.shape[0] > max_samples:
        num_cls = max(train_labels) + 1
        assert max_samples > num_cls, \
            'The requested number of samples, %d, is not higher than the total ' \
            'number of classes, %d.' % (max_samples, num_cls)
        num_samples_per_cls = int(max_samples // num_cls)
        train_inputs, train_labels, _, _ = select_num_samples_per_cls(
            num_samples_per_cls, train_inputs, train_labels, num_cls=num_cls)

    # Load test data.
    if 'test' in data:
        data_subset = tfds.as_numpy(data['test'])
        test_inputs = data_subset['image']
        test_labels = data_subset['coarse_label'] if coarse_labels else data_subset['label']
        # Remove dimension of size 1 from the labels.
        test_labels = np.squeeze(test_labels)
    else:
        logging.info('No test data, splitting %f from train...', ratio_test)
        # Split.
        train_inputs, train_labels, test_inputs, test_labels = \
            split_train_val_given_ratio_val(train_inputs, train_labels, ratio_val=ratio_test)
    # Load validation data.
    if 'validation' in data:
        data_subset = tfds.as_numpy(data['validation'])
        val_inputs = data_subset['image']
        val_labels = data_subset['coarse_label'] if coarse_labels else data_subset['label']
        # Remove dimension of size 1 from the labels.
        val_labels = np.squeeze(val_labels)
    else:
        logging.info('No validation data, splitting %f from train...', ratio_val)
        train_inputs, train_labels, val_inputs, val_labels = split_train_val_given_ratio_val(
            train_inputs, train_labels, ratio_val=ratio_val)

    # Potentially apply normalization.
    train_inputs, test_inputs, val_inputs = apply_normalization(
        train_inputs, test_inputs, val_inputs=val_inputs,
        normalization=normalization)

    # Get the class names.
    class_names = CLASS_NAMES[dataset_name] if dataset_name in CLASS_NAMES else None

    # Select the preprocessing function to apply online as batches are requested.
    feature_preproc_fn = get_feature_preproc_fn(online_preprocessing)

    # Create dataset.
    data = Dataset.build_from_splits(
        name=dataset_name+'-coarse' if coarse_labels else dataset_name,
        inputs_train=train_inputs,
        labels_train=train_labels,
        inputs_test=test_inputs,
        labels_test=test_labels,
        ratio_val=ratio_val,
        inputs_val=val_inputs,
        labels_val=val_labels,
        class_names=class_names,
        feature_preproc_fn=feature_preproc_fn)

    return data
