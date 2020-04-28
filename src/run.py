import logging
import os
import numpy as np
import tensorflow as tf

from .data.loaders import load_data
from .learning.hierarchy_computation import compute_dist_matrix, compute_hierarchy

from .learning.models import CNN
from .learning.trainer import Trainer
from .utils.printing import print_metrics_dict

__author__ = 'Otilia Stretcu'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# region Flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'shapes', 'Dataset name. '
                    'Options are: `shapes`, `tiny-imagenet` and other tensorflow_datasets.')
flags.DEFINE_string('data_path', None, 'Path to data in case it needs to be loaded from disk.')
flags.DEFINE_string('normalization', None,
                    'Type of data normalization to do at preprocessing time. Valid options are:'
                    ' None, norm.')
flags.DEFINE_string('online_preprocessing', None,
                    'Type of data normalization. Valid options are: None, norm.')
flags.DEFINE_integer('max_samples', None, 'Subsample this number of samples from the dataset.')
flags.DEFINE_float('ratio_val', 0.2, 'Percent of training data to use for validaton.')
flags.DEFINE_float('ratio_test', 0.2,
                   'Percent of training data to use for test, in case a test set is not provided.')
# Learning rate params.
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
# Regularization params.
flags.DEFINE_float('weight_decay', None, 'Weight for L2 loss on parameters of model f.')
flags.DEFINE_float('dropout', None, 'Dropout value.')
# Optimization params.
flags.DEFINE_integer('num_epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('max_num_epochs_auxiliary', None,
                     'Maximum number of epochs for the auxiliary functions. If None, it is set to '
                     'the same value as `num_epochs`.')
flags.DEFINE_integer('num_epochs_window_val', 1,
                     'Number of recent epochs to average over when computing validation accuracy.'
                     'This is only aimed at computing a smoothed validation accuracy for deciding'
                     'on convergence.')
flags.DEFINE_integer('max_num_epochs_no_val_improvement', None,
                     'If more than this number of epochs pass without improving the validation '
                     'metric, we stop.')
flags.DEFINE_integer('batch_size', 512, 'Batch size for agreement model.')
flags.DEFINE_float('clip_gradient_norm', None, 'Gradient clipping by norm.')
# Logging params.
flags.DEFINE_integer('logging_step', 1000,
                     'Print the loss every this number of iterations (not epochs).')
flags.DEFINE_integer('eval_step', 1, 'Evaluate every this number of epochs.')
flags.DEFINE_integer('seed', 123, 'Seed for random number generator.')
flags.DEFINE_bool('load_from_checkpoint', False, 'Whether load a trained model from checkpoint.')
flags.DEFINE_string('checkpoints_dir_original', '',
                    'An optional directory where to load the original model from.')
flags.DEFINE_bool('load_dist_matrix', False,
                  'Whether to load distance matrix, instead of computing it.')
flags.DEFINE_string('path_dist_matrix', None, 'File where to save the distance matrix.')
flags.DEFINE_string('model', 'mlp', 'Model string.')
# endregion Flags.

max_num_epochs_auxiliary = FLAGS.num_epochs if FLAGS.max_num_epochs_auxiliary is None \
    else FLAGS.max_num_epochs_auxiliary

# Initialization.
np.random.seed(FLAGS.seed)
tf.compat.v1.set_random_seed(FLAGS.seed)
np.set_printoptions(linewidth=250)

###############################################################################################
#                                            DATA                                             #
###############################################################################################

# Load training and eval data.
data = load_data(
    FLAGS.dataset,
    ratio_val=FLAGS.ratio_val,
    ratio_test=FLAGS.ratio_test,
    normalization=FLAGS.normalization,
    data_path=FLAGS.data_path,
    max_samples=FLAGS.max_samples)

###############################################################################################
#                                       PREPARE OUTPUTS                                       #
###############################################################################################

# Params.
model_name = FLAGS.dataset + '-seed_' + str(FLAGS.seed)
model_name += '-max_samples_%d' % FLAGS.max_samples if FLAGS.max_samples else ''
model_name += '-ep_%d' % FLAGS.num_epochs
model_name += '-drop_%s' % str(FLAGS.dropout) if FLAGS.dropout else ''
model_name += '-clip_%s' % str(FLAGS.clip_gradient_norm) if FLAGS.clip_gradient_norm else ''
print('Model name: ', model_name)

summary_dir = 'outputs/summaries/' + data.name + '/' + model_name
results_dir = 'outputs/results/' + data.name + '/' + model_name
if FLAGS.load_from_checkpoint and FLAGS.checkpoints_dir_original:
    checkpoints_dir_original = FLAGS.checkpoints_dir_original
else:
    checkpoints_dir_original = 'outputs/checkpoints/' + data.name + '/' + model_name + '-original'
checkpoints_dir = 'outputs/checkpoints/' + data.name + '/' + model_name
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# You save the distance matrix here, or upload your own label distance matrix.
if FLAGS.path_dist_matrix is None:
    path_dist_matrix = './outputs/dist_matrix/'
    if not os.path.exists(path_dist_matrix):
        os.makedirs(path_dist_matrix)
    path_dist_matrix = path_dist_matrix + 'dist_matrix-' + model_name + '.npy'
else:
    path_dist_matrix = FLAGS.path_dist_matrix


###############################################################################################
#                                         MODEL SETUP                                         #
###############################################################################################

# Create prediction model.
f = CNN(
    inputs_shape=data.features_shape,
    num_outputs=data.num_classes,
    activation=tf.keras.layers.LeakyReLU(alpha=0.1),
    reg_weight=FLAGS.weight_decay,
    dropout=FLAGS.dropout,
    name='cnn')

# Create a trainer.
val_metric = 'acc'
higher_is_better = True
max_num_epochs_no_val_improvement = np.inf if FLAGS.max_num_epochs_no_val_improvement is None \
    else FLAGS.max_num_epochs_no_val_improvement
trainer = Trainer(
    f,
    optimizer=tf.train.AdamOptimizer,
    optimizer_params={'learning_rate': FLAGS.lr},
    clip_gradient_norm=FLAGS.clip_gradient_norm,
    logging_step=FLAGS.logging_step,
    eval_step=FLAGS.eval_step,
    num_epochs_window_val=FLAGS.num_epochs_window_val,
    max_num_epochs_no_val_improvement=max_num_epochs_no_val_improvement,
    val_metric='acc')

###############################################################################################
#                                   COMPUTE DISTANCE MATRIX                                   #
###############################################################################################

dist_matrix, test_metric, other_metrics = compute_dist_matrix(
    load_dist_matrix=FLAGS.load_dist_matrix,
    path_dist_matrix=path_dist_matrix,
    trainer=trainer,
    data=data,
    num_epochs=FLAGS.num_epochs,
    batch_size=FLAGS.batch_size,
    summary_dir=summary_dir,
    load_from_checkpoint=FLAGS.load_from_checkpoint,
    checkpoints_dir=checkpoints_dir_original)

###############################################################################################
#                                      COMPUTE HIERARCHY                                      #
###############################################################################################

logging.info('Clustering...')
clusters_per_level = compute_hierarchy(dist_matrix=dist_matrix)

print('Number of hierarchy levels: ', len(clusters_per_level))
print('Clusters per level [class indices]:')
for level, clusters in enumerate(clusters_per_level):
    print('Level %d: ' % level, clusters)
if data.class_names is not None:
    print('Clusters per level [class names]:')
    for level, clusters in enumerate(clusters_per_level):
        clusters = [[data.class_names[i] for i in cluster] for cluster in clusters]
        print('Level %d: ' % level, clusters)

###############################################################################################
#                             TRAIN AT EACH LEVEL OF THE HIERARCHY                            #
###############################################################################################

is_pretrained = False
level = 0
keep_going = True
while keep_going:
    # Obtain the label clusters for the current hierarchy level.
    label_clusters = clusters_per_level[level]

    # Compute a mapping from the previous labels to the new labels. At this level, we will have as
    # many new labels as the number of clusters in label_clusters.
    original_to_new_label = {}
    new_label_to_cluster = {}
    for new_label, label_group in enumerate(label_clusters):
        for original_label in label_group:
            original_to_new_label[original_label] = new_label
        new_label_to_cluster[new_label] = label_group

    num_classes_current = len(label_clusters)

    # Update labels.
    labels = np.asarray([original_to_new_label[l] for l in data.labels])
    data_finer = data.copy(labels=labels, num_classes=num_classes_current)

    # Train model with new labels.
    logging.info('-' * 100)
    logging.info('Train %d classes...', data_finer.num_classes)
    print('Class groups: ', label_clusters)
    if data.class_names is not None:
        groups_names = [[data.class_names[c] for c in group]
                        for group in label_clusters]
        print('Class groups: ', groups_names)
    checkpoints_dir_level = checkpoints_dir+'-level_%d' % (level + 1) if checkpoints_dir else None
    summary_dir_level = summary_dir+'-level_%d' % (level + 1) if summary_dir else None
    trainer.train(
        data_finer,
        num_epochs=max_num_epochs_auxiliary if num_classes_current < data.num_classes else FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
        reset_encoder=not is_pretrained,
        checkpoints_dir=checkpoints_dir_level,
        summary_dir=summary_dir_level)

    is_pretrained = True
    level += 1
    keep_going = num_classes_current < data.num_classes


if not FLAGS.load_dist_matrix:
    logging.info('Printing again the metrics of the baseline model:')
    logging.info('Test %s: %.4f', trainer.val_metric, test_metric)
    print_metrics_dict(other_metrics)

print('\n\nTesting the model trained with curriculum on original data:')
test_metric, other_metrics = trainer.test(data, FLAGS.batch_size, data.get_indices_test())
logging.info('Test %s: %.2f', trainer.val_metric, test_metric)
print_metrics_dict(other_metrics)
