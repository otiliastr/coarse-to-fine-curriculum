import numpy as np
import os

from matplotlib.pyplot import imread

from .dataset import Dataset
from .preprocessing import apply_normalization, get_feature_preproc_fn,\
    select_num_samples_per_cls, split_train_val_given_ratio_val

__author__ = 'Otilia Stretcu'


def load_shapes(dataset_name, path, normalization='norm', ratio_test=0.1, ratio_val=0.1,
                online_preprocessing=None, max_samples=None):
    assert path is not None, "The data path must be provided!"

    # Load all png files in this folder.
    path_png = os.path.join(path, 'png')
    images = []
    sample_ids = []
    for filename in os.listdir(path_png):
        if filename.endswith(".png"):
            sample_id = int(filename.split('.')[0])
            image = imread(os.path.join(path_png, filename))
            sample_ids.append(sample_id)
            images.append(image)

    # Make sure we are not missing any images.
    num_samples = len(sample_ids)
    assert set(sample_ids) == set(range(num_samples))

    # Load the labels.
    path_captions = os.path.join(path, 'full_captions.txt')
    with open(path_captions, 'r') as f:
        labels = [line.strip('\n') for line in f.readlines()]

    # Convert labels from strings to label IDs in range [0, num_classes).
    unique_labels = sorted(list(set(labels)))
    label_name_to_id = {name: i for i, name in enumerate(unique_labels)}
    labels = [label_name_to_id[l] for l in labels]

    # Save images and labels as numpy arrays, and make sure the order of images and labels
    # are matched.
    images = np.asarray(images)
    sample_ids = np.asarray(sample_ids)
    train_inputs = np.zeros_like(images)
    train_inputs[sample_ids] = images
    train_labels = np.asarray(labels)

    # Separate a test set.
    train_inputs, train_labels, test_inputs, test_labels = \
        split_train_val_given_ratio_val(train_inputs, train_labels, ratio_val=ratio_test)

    if max_samples is not None and train_inputs.shape[0] > max_samples:
        num_cls = len(unique_labels)
        num_samples_per_cls = max_samples // num_cls
        train_inputs, train_labels, remaining_inputs, remaining_labels = select_num_samples_per_cls(
            num_samples_per_cls, train_inputs, train_labels, num_cls=num_cls)

    # Potentially apply normalization.
    train_inputs, test_inputs = apply_normalization(
        train_inputs, test_inputs, normalization=normalization)

    # Select the preprocessing function to apply online as batches are requested.
    feature_preproc_fn = get_feature_preproc_fn(online_preprocessing)

    # Create dataset.
    data = Dataset.build_from_splits(
        name=dataset_name,
        inputs_train=train_inputs,
        labels_train=train_labels,
        inputs_test=test_inputs,
        labels_test=test_labels,
        ratio_val=ratio_val,
        class_names=unique_labels,
        feature_preproc_fn=feature_preproc_fn)

    return data
