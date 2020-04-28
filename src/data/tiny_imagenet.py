"""Loader for the Tiny Imagenet dataset.

Data is available at: https://tiny-imagenet.herokuapp.com/
"""

import glob
import logging
import numpy as np
import os
import tensorflow as tf

from io import BytesIO
from urllib.request import urlopen
from tqdm import tqdm
from zipfile import ZipFile

from .preprocessing import select_num_samples_per_cls, get_feature_preproc_fn
from .dataset import Dataset
from .preprocessing import apply_normalization, split_train_val_given_ratio_val

URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
FOLDER = 'tiny-imagenet-200'


def load_tiny_imagenet(data_path, normalization='norm', ratio_val=None, online_preprocessing=None,
                       max_samples=None, dataset_name='tiny-imagenet'):
    data_path = maybe_download(data_path)

    # Load WordNet IDs for the classes in this dataset.
    with open(os.path.join(data_path, 'wnids.txt'), 'r') as f:
        wn_ids = f.read().splitlines()
    assert len(wn_ids) == 200

    # Create an index between 0,..., 199 for each class.
    wn_id_to_class_index = {wn_id: index for index, wn_id in enumerate(wn_ids)}

    wn_ids_set = set(wn_ids)

    # Load the human-readable names associated with these class ids.
    with open(os.path.join(data_path, 'words.txt'), 'r') as f:
        wn_id_to_name = list(map(
            lambda line: line.split('\t'),
            f.read().splitlines()))
        wn_id_to_name = {wn_id: process_class_name(name)
                         for wn_id, name in wn_id_to_name
                         if wn_id in wn_ids_set}
    class_names = [wn_id_to_name[wn_id] for wn_id in wn_ids]

    # Load train data.
    train_inputs, train_labels = read_train_data(data_path, wn_ids, wn_id_to_class_index)

    if max_samples is not None and train_inputs.shape[0] > max_samples:
        num_cls = max(train_labels) + 1
        assert max_samples > num_cls, \
            'The requested number of samples, %d, is less than the total ' \
            'number of classes, %d.' % (max_samples, num_cls)
        num_samples_per_cls = int(max_samples // num_cls)
        train_inputs, train_labels, _, _ = select_num_samples_per_cls(
            num_samples_per_cls, train_inputs, train_labels, num_cls=num_cls)

    # Load test data. We use the provided validation set as test data, because the test labels
    # are not provided. For validation we will use instead a fraction of the training set.
    test_inputs, test_labels = read_val_data(data_path, wn_id_to_class_index)

    # Select new validation data.
    logging.info('No validation data, splitting %f from train...', ratio_val)
    train_inputs, train_labels, val_inputs, val_labels = split_train_val_given_ratio_val(
        train_inputs, train_labels, ratio_val=ratio_val)

    # Potentially apply normalization.
    train_inputs, test_inputs, val_inputs = apply_normalization(
        train_inputs, test_inputs, val_inputs=val_inputs, normalization=normalization)

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
        inputs_val=val_inputs,
        labels_val=val_labels,
        class_names=class_names,
        feature_preproc_fn=feature_preproc_fn)

    return data


def maybe_download(path):
    if os.path.exists(os.path.join(path, FOLDER)):
        return maybe_download(os.path.join(path, FOLDER))
    if not (os.path.exists(os.path.join(path, 'wnids.txt')) and
        os.path.exists(os.path.join(path, 'words.txt')) and
        os.path.exists(os.path.join(path, 'train')) and
        os.path.exists(os.path.join(path, 'val'))):
            logging.info('Data does not exist at path `%s`. Downloading it from %s.' % (path, URL))
            with urlopen(URL) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(path)
            return maybe_download(os.path.join(path, FOLDER))
    return path


def process_class_name(name):
    # Remove space when it comes right after comma, and replace comma with arrow.
    name = name.replace(', ', '->')
    # Replace remaining spaces with underscore.
    name = name.replace(' ', '_')
    return name


def read_and_preprocess_image(file_path):
    # Load the raw data from the file as a string.
    img = tf.read_file(file_path)
    # Convert the compressed string to a 3D uint8 tensor.
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    return img.numpy()


def read_train_data(data_path, wn_ids, wn_id_to_class_index):
    """Load training data for Tiny Imagenet dataset."""
    logging.info('Loading training data...')

    # Load training data for each class present in wn_ids.
    # The data is stored such that the folder train/[wn_id]/images is a folder containing
    #  all train images belonging to class [wn_id].
    train_inputs = []
    train_labels = []
    pbar = tqdm(total=len(wn_ids))
    for wn_id in wn_ids:
        class_data_path = os.path.join(data_path, 'train', wn_id, 'images')
        regex = os.path.join(class_data_path, '*.JPEG')
        filenames = glob.glob(regex)
        num_images = len(filenames)

        # Load the raw data from the file as a string.
        class_images = [read_and_preprocess_image(filename) for filename in filenames]
        class_labels = [wn_id_to_class_index[wn_id]] * num_images
        train_inputs.extend(class_images)
        train_labels.extend(class_labels)
        pbar.update(1)

    train_inputs = np.stack(train_inputs)
    train_labels = np.asarray(train_labels)

    return train_inputs, train_labels


def read_val_data(data_path, wn_id_to_class_index):
    """Loads validation data for Tiny Image dataset."""
    logging.info('Loading validation data...')

    # The images are stored in the folder val/images as jpeg files.
    images_path = os.path.join(data_path, 'val', 'images')
    regex = os.path.join(images_path, '*.JPEG')
    filenames = glob.glob(regex)

    # Load the labels from file.
    with open(os.path.join(data_path, 'val', 'val_annotations.txt'), 'r') as f:
        filename_to_wn_id = dict([line.split()[:2] for line in f.readlines()])

    # Load the raw images from the files.
    inputs = [read_and_preprocess_image(filename) for filename in filenames]
    labels = [wn_id_to_class_index[filename_to_wn_id[filename.split('/')[-1]]]
              for filename in filenames]

    inputs = np.stack(inputs)
    labels = np.asarray(labels)

    return inputs, labels
