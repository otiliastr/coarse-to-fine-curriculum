from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import os
import tensorflow as tf

from ..utils.circular_buffer import CircularBuffer

__author__ = 'Otilia Stretcu'


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer=tf.train.AdamOptimizer,
                 optimizer_params=None,
                 logging_step=10,
                 eval_step=10,
                 clip_gradient_norm=None,
                 num_epochs_window_val=1,
                 max_batches_eval_train=10,
                 val_metric=None,
                 abs_loss_chg_tol=1e-7,
                 rel_loss_chg_tol=1e-5,
                 loss_chg_iter_below_tol=10,
                 max_num_epochs_no_val_improvement=np.inf):
        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.logging_step = logging_step
        self.eval_step = eval_step
        self.clip_gradient_norm = clip_gradient_norm
        self.num_epochs_window_val = num_epochs_window_val
        self.max_elems_val_buffer = max(int(num_epochs_window_val // eval_step), 1)
        self.max_batches_eval_train = max_batches_eval_train
        self.abs_loss_chg_tol = abs_loss_chg_tol
        self.rel_loss_chg_tol = rel_loss_chg_tol
        self.loss_chg_iter_below_tol = loss_chg_iter_below_tol
        self.max_num_epochs_no_val_improvement = max_num_epochs_no_val_improvement

        # Create model.
        self.model = model
        self.encoder = model.encoder()
        self.predictor = None
        self.saver = None
        self.encoder_output_shape = None

        assert val_metric in ['acc', 'loss', None]
        self.val_metric = val_metric if val_metric is not None else 'acc'
        self.higher_better = True if self.val_metric == 'acc' else False
        self.other_metrics = ['confusion']

    def train(self,
              data,
              num_epochs,
              batch_size,
              reset_encoder=False,
              checkpoints_dir=None,
              summary_dir=None):
        """Trains the model on the provided dataset.

        Args:
            data: A Dataset object.
            num_epochs: Maximum number of epochs to train for.
            batch_size: Batch size for training and test.
            reset_encoder: Boolean specifying whether to reset the encoder parameters or use
                the current ones (if the model has been initialized).
            checkpoints_dir: Path to a directory where to save model checkpoints.
            summary_dir: Path to a directory where to save model summaries.

        Returns:

        """
        # Create a TensorFlow summary writer.
        summary_writer = tf.contrib.summary.create_file_writer(summary_dir) if summary_dir else None

        # Create encoder and predictor. The predictor is always reinitialized, but the encoder
        # parameters may be allowed to keep their previous value.
        self.predictor = self.model.predictor(num_outputs=data.num_classes)
        if reset_encoder:
            self.encoder = self.model.encoder()

        # Prepare optimizer.
        optimizer = self.optimizer(**self.optimizer_params)

        # Prepare metrics.
        best_val_acc = -np.inf if self.higher_better else np.inf
        test_acc_at_best_val = -np.inf if self.higher_better else np.inf
        other_metrics_at_best_val = {}
        best_epoch = 0
        val_acc_buffer = CircularBuffer(self.max_elems_val_buffer)

        # Do an initial evaluation.
        self._evaluate_epoch(data, batch_size, -1, -1, -1, -1, {}, val_acc_buffer=val_acc_buffer)

        # Train.
        global_step = tf.Variable(0)
        epoch = 0
        encodings_shape = None
        iter_below_tol = 0
        prev_loss_val = np.inf
        for epoch in range(num_epochs):
            # Train.
            encodings_shape, loss_val = self._train_epoch(data, batch_size, optimizer, global_step)

            # Test.
            if epoch % self.eval_step == 0:
                best_val_acc, test_acc_at_best_val, best_epoch, other_metrics_at_best_val, other_metrics_val = \
                    self._evaluate_epoch(
                        data, batch_size, epoch, best_val_acc, test_acc_at_best_val, best_epoch,
                        other_metrics_at_best_val,
                        checkpoints_dir=checkpoints_dir,
                        summary_writer=summary_writer,
                        val_acc_buffer=val_acc_buffer)

                self._write_train_summary(summary_writer, epoch, loss_val, other_metrics_val)

            has_converged, iter_below_tol = self._check_convergence(
                prev_loss_val, loss_val, iter_below_tol, best_epoch, epoch)
            if has_converged:
                break

        # Do a final evaluation.
        best_val_acc, test_acc_at_best_val, best_epoch, other_metrics_at_best_val, _ = \
            self._evaluate_epoch(
                data, batch_size, epoch, best_val_acc, test_acc_at_best_val,
                best_epoch, other_metrics_at_best_val, checkpoints_dir=checkpoints_dir,
                val_acc_buffer=val_acc_buffer)

        logging.info('Best val %s: %.4f at epoch %d. Test %s at best val point: %.4f',
                     self.val_metric, best_val_acc, best_epoch,
                     self.val_metric, test_acc_at_best_val)
        logging.info(
            'Other metrics at best val: ' +
            ', '.join(['%s = %.4f' % (name, val)
                       for name, val in other_metrics_at_best_val.items()
                       if not isinstance(val, np.ndarray) or len(val)==1]))

        # Save the output shape, needed for saving and restoring vars.
        if encodings_shape is not None:
            self.encoder_output_shape = encodings_shape

        # Restore the checkpoint with the best validation accuracy.
        if checkpoints_dir is not None:
            self.restore(checkpoints_dir)

        return test_acc_at_best_val, other_metrics_at_best_val

    def _train_epoch(self, data, batch_size, optimizer, global_step):
        """Trains the model for one epoch."""
        train_data = create_train_dataset(data, batch_size)
        num_train = len(data.get_indices_train())
        num_visited_samples = 0

        for inputs_batch, targets_batch in train_data:
            inputs_batch = tf.convert_to_tensor(inputs_batch)
            targets_batch = tf.convert_to_tensor(targets_batch)

            iter = global_step.numpy()
            with tf.GradientTape() as tape:
                encodings = self.encoder(inputs_batch, training=True)
                logits = self.predictor(encodings, training=True)
                loss_value = self.loss(logits, targets_batch)
            trainable_variables = self.encoder.trainable_variables + \
                                  self.predictor.trainable_variables
            grads = tape.gradient(loss_value, trainable_variables)
            if self.clip_gradient_norm:
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.clip_gradient_norm)
            optimizer.apply_gradients(zip(grads, trainable_variables), global_step)

            if iter % self.logging_step == 0:
                logging.info("Iteration %10d | loss: %.5f " % (iter, loss_value.numpy()))

            num_visited_samples += inputs_batch.shape[0]
            if num_visited_samples >= num_train:
                break

        return encodings.shape, loss_value

    def _evaluate_epoch(self, data, batch_size, epoch, best_val_acc, test_acc_at_best_val,
                        best_epoch, other_metrics_at_best_val, val_acc_buffer,
                        checkpoints_dir=None, summary_writer=None):
        train_acc, _ = self.test(data, batch_size, indices=data.get_indices_train(),
                                 max_batches=self.max_batches_eval_train)
        val_acc, other_metrics_val = self.test(data, batch_size, indices=data.get_indices_val())
        test_acc, other_metrics = self.test(data, batch_size, indices=data.get_indices_test())

        # Compute a smooth validation metric, by averaging over a rolling window of val metrics.
        val_acc_buffer.add(val_acc)
        val_smooth = val_acc_buffer.average()

        self._write_eval_summary(summary_writer, epoch, train_acc, val_acc, val_smooth, test_acc,
                                 other_metrics)

        # Create an output string for logging metrics.
        log_string = \
            "Epoch %10d | train_%s: %.4f | val_%s: %.4f | test_%s: %.4f" % \
            (epoch, self.val_metric, train_acc, self.val_metric, val_acc,
             self.val_metric, test_acc)
        for name, val in other_metrics.items():
            if not isinstance(val, np.ndarray) or len(val) == 1:
                log_string += " | test_%s: %.4f" % (name, val)
        logging.info(log_string)

        # Check if the best validation metric has improved, and if so save a checkpoint.
        improvement = (self.higher_better and val_smooth > best_val_acc) or \
                      (not self.higher_better and val_smooth < best_val_acc)
        if improvement:
            best_val_acc = val_smooth
            test_acc_at_best_val = test_acc
            best_epoch = epoch
            other_metrics_at_best_val = other_metrics
            if checkpoints_dir is not None:
                self.save(checkpoints_dir)
        return best_val_acc, test_acc_at_best_val, best_epoch, other_metrics_at_best_val, \
               other_metrics_val

    def _check_convergence(self, prev_loss, loss, iter_below_tol, best_epoch, epoch):
        """Checks for convergence."""
        has_converged = False

        # Check if we have reached the desired loss tolerance.
        loss_diff = abs(prev_loss - loss)
        if loss_diff < self.abs_loss_chg_tol or abs(loss_diff / prev_loss) < self.rel_loss_chg_tol:
            iter_below_tol += 1
        else:
            iter_below_tol = 0
        if iter_below_tol >= self.loss_chg_iter_below_tol:
            logging.info('Loss value converged. The loss function has not changed significantly '
                         'for %d iterations.', iter_below_tol)
            has_converged = True

        # Check if there have passed more than the allowed epochs without improvement.
        if epoch - best_epoch > self.max_num_epochs_no_val_improvement:
            logging.info('More than %d epochs without improvement. Stopping.',
                         self.max_num_epochs_no_val_improvement)
            has_converged = True

        return has_converged, iter_below_tol

    def _write_eval_summary(self, summary_writer, step, train_acc, val_acc, val_smooth, test_acc,
                            other_metrics):
        """Writes a tensorboard summary."""
        if summary_writer:
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('train/train_acc', train_acc, step=step)
                tf.contrib.summary.scalar('val/val_acc', val_acc, step=step)
                tf.contrib.summary.scalar('val/val_smooth', val_smooth, step=step)
                tf.contrib.summary.scalar('test/test_acc', test_acc, step=step)
                for metric_name, metric_val in other_metrics.items():
                    if not isinstance(metric_val, (tuple, list, np.ndarray)):
                        tf.contrib.summary.scalar('test/'+metric_name, metric_val, step=step)
                summary_writer.flush()

    def _write_train_summary(self, summary_writer, step, loss, other_metrics_val):
        """Writes a  Tensorboard summary."""
        if summary_writer:
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('train/loss', loss, step=step)
                for metric_name, metric_val in other_metrics_val.items():
                    if not isinstance(metric_val, (tuple, list, np.ndarray)):
                        tf.contrib.summary.scalar('val/'+metric_name, metric_val, step=step)
                summary_writer.flush()

    def loss(self, logits, targets):
        return tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels=targets, logits=logits, reduction="weighted_mean")

    def test(self, data, batch_size, indices, max_batches=None):
        test_data = create_test_dataset(data, batch_size, indices)
        acc = tf.contrib.eager.metrics.Accuracy()
        loss_sum = 0
        confusion = np.zeros((data.num_classes, data.num_classes))
        batch_index = 0
        num_samples = 0
        for inputs_batch, targets_batch in test_data:
            encodings = self.encoder(inputs_batch, training=False)
            logits = self.predictor(encodings, training=False)
            predictions = tf.cast(tf.argmax(logits, axis=1), targets_batch.dtype)
            acc(predictions, targets_batch)
            for t, p in zip(targets_batch, predictions):
                confusion[t, p] += 1
            current_batch_size = inputs_batch.shape[0].value
            loss_sum += self.loss(logits, targets_batch) * current_batch_size
            batch_index += 1
            num_samples += current_batch_size
            if max_batches is not None and batch_index >= max_batches:
                break
        # Row normalize confusion matrix.
        row_sums = confusion.sum(axis=1)
        confusion = confusion / row_sums[:, np.newaxis]
        val_metric = acc.result().numpy()
        # Compute loss.
        loss_value = loss_sum / num_samples
        # Save metrics.
        other_metrics = {'acc': val_metric, 'confusion': confusion, 'loss': loss_value}
        return val_metric, other_metrics

    def predict(self, inputs_batch, normalize=False, ):
        """Predicts logits or probabilities for a provided batch of samples.

        Arguments:
            inputs_batch: A batch of inputs, where the first dimension is batch
                size.
            normalize: A boolean specifying if the output is probabilities
                (if True) or logits (if False).
        Returns:
            A batch of predictions.
        """
        encodings = self.encoder(inputs_batch, training=False)
        predictions = self.predictor(encodings)
        if normalize:
            predictions = tf.nn.softmax(predictions)
        return predictions.numpy()

    def initialize(self, data):
        """Initializes the model, building the encoder and predictor."""
        logging.info('Initializing model...')
        self.predictor = self.model.predictor(num_outputs=data.num_classes)
        # Pass a couple of data samples through the model such that model variables are
        # initialized, according to the shapes of the data.
        indices = data.get_indices_train()[:2]
        inputs = data.get_features(indices)
        encodings = self.encoder(inputs, training=False)
        self.predictor(encodings)
        # Save the output shape, needed for saving and restoring vars.
        self.encoder_output_shape = encodings.shape

    def save(self, checkpoints_dir, suffix='ckpt'):
        """Saves a model checkpoint."""
        if self.predictor is None:
            self.predictor = self.model.predictor()
            with tf.name_scope(self.predictor.name):
                self.predictor.build(self.encoder_output_shape)
        trainable_variables = self.encoder.trainable_variables + \
                              self.predictor.trainable_variables
        trainable_variables = {v.name: v for v in trainable_variables}
        checkpoint = tf.train.Checkpoint(**trainable_variables)
        checkpoint.save(file_prefix=os.path.join(checkpoints_dir, suffix))
        print('Saved checkpoint at: ', os.path.join(checkpoints_dir, suffix))

    def restore(self, checkpoints_dir, data=None):
        """Restores the model variables from a checkpoint."""
        logging.info('Restoring model from %s ...', checkpoints_dir)
        if self.predictor is None:
            assert data is not None, 'If the model is not initialized, we need to pass ' \
                                     'the `data` argument to initialize it.'
            self.initialize(data)
        trainable_variables = self.encoder.trainable_variables + \
                              self.predictor.trainable_variables
        trainable_variables = {v.name: v for v in trainable_variables}
        checkpoint = tf.train.Checkpoint(**trainable_variables)
        status = checkpoint.restore(tf.train.latest_checkpoint(checkpoints_dir))
        status.assert_consumed()
        status.assert_existing_objects_matched()
        logging.info('Variables restored.')


def create_train_dataset(data,
                         batch_size,
                         prefetch_buffer_size=5,
                         shuffle_buffer_size=10000,
                         map_fn=None):
    x_train = data.get_features(data.get_indices_train())
    y_train = data.get_labels(data.get_indices_train())
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if map_fn is not None:
        dataset = dataset.map(map_fn)
    dataset = dataset \
        .repeat() \
        .shuffle(shuffle_buffer_size) \
        .batch(batch_size) \
        .prefetch(prefetch_buffer_size)
    return dataset


def create_test_dataset(data,
                        batch_size,
                        indices,
                        map_fn=None,
                        prefetch_buffer_size=2):
    x_train = data.get_features(indices)
    y_train = data.get_labels(indices)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if map_fn is not None:
        dataset = dataset.map(map_fn)
    dataset = dataset.batch(batch_size).prefetch(prefetch_buffer_size)
    return dataset


def create_predict_dataset(data,
                           batch_size,
                           indices,
                           map_fn=None,
                           prefetch_buffer_size=2):
    x_train = data.get_features(indices)
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    if map_fn is not None:
        dataset = dataset.map(map_fn)
    dataset = dataset.batch(batch_size).prefetch(prefetch_buffer_size)
    return dataset
