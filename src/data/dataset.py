import numpy as np

from .preprocessing import no_preproc, split_train_val

__author__ = 'Otilia Stretcu'


class Dataset(object):
    def __init__(self, name, features, labels, indices_train, indices_test, indices_val,
                 num_classes=None, feature_preproc_fn=no_preproc, class_names=None):
        self.name = name
        self.features = features
        self.labels = labels

        self.indices_train = indices_train
        self.indices_val = indices_val
        self.indices_test = indices_test
        self.feature_preproc_fn = feature_preproc_fn

        self.num_val = len(self.indices_val)
        self.num_test = len(self.indices_test)

        self.num_samples = len(features)
        self.features_shape = features[0].shape
        self.num_features = np.prod(features[0].shape)
        self.num_classes = 1 + max(labels) if num_classes is None else num_classes
        self.class_names = class_names

    @staticmethod
    def build_from_splits(name,
                          inputs_train,
                          labels_train,
                          inputs_test,
                          labels_test,
                          inputs_val=None,
                          labels_val=None,
                          ratio_val=0.1,
                          num_classes=None,
                          class_names=None,
                          feature_preproc_fn=no_preproc):
        if inputs_val is None:
            indices_train, indices_val = split_train_val(
                inputs_train=np.arange(inputs_train.shape[0]),
                ratio_val=ratio_val)
            inputs_val = inputs_train[indices_val]
            inputs_train = inputs_train[indices_train]
            labels_val = labels_train[indices_val]
            labels_train = labels_train[indices_train]

        num_train = len(inputs_train)
        num_val = len(inputs_val)
        num_test = len(inputs_test)

        features = np.concatenate((inputs_train, inputs_val, inputs_test))
        labels = np.concatenate((labels_train, labels_val, labels_test))
        indices_train = np.arange(num_train)
        indices_val = np.arange(num_train, num_train+num_val)
        indices_test = np.arange(num_train+num_val, num_train+num_val+num_test)

        return Dataset(name=name,
                       features=features,
                       labels=labels,
                       indices_train=indices_train,
                       indices_test=indices_test,
                       indices_val=indices_val,
                       num_classes=num_classes,
                       class_names=class_names,
                       feature_preproc_fn=feature_preproc_fn)

    @staticmethod
    def build_from_features(name, features, labels, indices_train, indices_test,
                            indices_val=None, ratio_val=0.2, seed=None,
                            num_classes=None, class_names=None,
                            feature_preproc_fn=lambda x: x):
        if indices_val is None:
            rng = np.random.RandomState(seed=seed)
            indices_train, indices_val = split_train_val(
                np.arange(indices_train.shape[0]), ratio_val, rng)

        return Dataset(name=name,
                       features=features,
                       labels=labels,
                       indices_train=indices_train,
                       indices_test=indices_test,
                       indices_val=indices_val,
                       num_classes=num_classes,
                       class_names=class_names,
                       feature_preproc_fn=feature_preproc_fn)

    def get_labels(self, indices):
        return self.labels[indices]

    def get_indices_train(self):
        return self.indices_train

    def get_indices_val(self):
        return self.indices_val

    def get_indices_test(self):
        return self.indices_test

    def get_features(self, indices, is_train=False, **kwargs):
        f = self.features[indices]
        f = self.feature_preproc_fn(f, is_train=is_train, **kwargs)
        return f

    def copy(self, name=None, features=None, labels=None, indices_train=None,
             indices_test=None, indices_val=None, num_classes=None,
             class_names=None, feature_preproc_fn=None):
        name = name if name is not None else self.name
        features = features if features is not None else self.features
        labels = labels if labels is not None else self.labels
        indices_train = indices_train if indices_train is not None else self.indices_train
        indices_test = indices_test if indices_test is not None else self.indices_test
        indices_val = indices_val if indices_val is not None else self.indices_val
        num_classes = num_classes if num_classes is not None else self.num_classes
        class_names = class_names if class_names is not None else self.class_names
        feature_preproc_fn = feature_preproc_fn if feature_preproc_fn is not None else self.feature_preproc_fn

        return self.__class__(name=name,
                              features=features,
                              labels=labels,
                              indices_train=indices_train,
                              indices_test=indices_test,
                              indices_val=indices_val,
                              num_classes=num_classes,
                              class_names=class_names,
                              feature_preproc_fn=feature_preproc_fn)
