# Coarse-to-Fine Curriculum Learning

This code repository contains an implementation of the work described in:

[O. Stretcu, E.A. Platanios, T. Mitchell, B. Poczos. "Coarse-to-Fine Curriculum Learning." Bridging AI and Cognitive Science (BAICS) Workshop at ICLR 2020](https://baicsworkshop.github.io/pdf/BAICS_26.pdf).

To cite this work please use the following citation:
```bibtex
@inproceedings{coarse-to-fine:2020,
  title={{Coarse-to-Fine Curriculum Learning for Classification}},
  author={Stretcu, Otilia and Platanios, Emmanouil Antonios and Mitchell, Tom and P{\'o}czos, Barnab{\'a}s},
  booktitle={International Conference on Learning Representations (ICLR) Workshop on Bridging AI and Cognitive Science (BAICS)},
  year={2020},
}
```

The code is organized into the following folders:

- **data:** contains classes and methods for loading and
  preprocessing data.
- **learning:**
  - models.py: model architectures implementations.
  - trainder.py: a class in charge of training a provided model.
  - hierarchy_computation.py: code for computing the label hierarchy.
- **utils:** other utility functions, including our
  implementation of the Affinity clustering algorithm.
    
More details can be found in our
[paper](https://baicsworkshop.github.io/pdf/BAICS_26.pdf),
[video & slides](https://baicsworkshop.github.io/program/baics_26.html),

## Requirements

Our code was implemented in Python 3.7 with Tensorflow
1.14.0 using [eager execution](https://www.tensorflow.org/guide/eager).
It also requires the following Python packages:

- matplotlib
- numpy
- tensorflow-datasets
- tqdm
- urllib
- yaml

## How to run

Our code currently supports the datasets included in the
[tensorflow-datasets](https://www.tensorflow.org/datasets/catalog/overview)
package, as well as [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/)
and the synthetic Shapes dataset described in our paper.

##### CIFAR-100, CIFAR-10 and more

To run on tensorflow-datasets such as CIFAR-100, from the
base repository folder run the command:

```bash
python3.7 -m src.run \
  --dataset=cifar100 \
  --normalization=norm
```

##### Shapes

To run on the Shapes dataset, from the base repository
folder run the command:

```bash
python3.7 -m src.run \
  --dataset=shapes \
  --data_path=data/shapes \
  --normalization=norm
```

We provide the Shapes dataset used in our experiments in
the folder `data/shapes`.

##### Tiny ImageNet

To run on the Tiny ImageNet dataset, from the base
repository folder run the command:

```bash
python3.7 -m src.run \
  --dataset=tiny_imagenet \
  --data_path=data/tiny_imagent \
  --normalization=norm
```

This will automatically download the Tiny ImageNet dataset
from the official website to the folder specified by the
`--data_path` flag.


##### Advice for running the code

Note that our code offers a range of commandline arguments,
which are described in run.py. An important argument is
`normalization` which specifies what type of preprocessing
to perform on the data (e.g, in the examples above
`--normalization=norm` specifies that inputs should be
converted from integer pixels in [0, 255] to float numbers
in [0, 1]).

Another useful argument is `max_num_epochs_auxiliary` which
can limit the maximum number of epochs for each of the
auxiliary functions (when training on coarser labels). In
our experiments we set this to a number lower than
`num_epochs`.

We recommend running on a GPU. With CUDA, this can be done
by prepending `CUDA_VISIBLE_DEVICES=<your-gpu-number>` in
front of the run command.

## Visualizing the results

To visualize the results in Tensorboard, use the following
command, adjusting the dataset name accordingly:
`tensorboard --logdir=outputs/summaries/cifar100`

An example of such visualization on the CIFAR-100 dataset
is the following:

![Tensorboard plot](cifar100_plot.png?raw=true "CIFAR-100 results")

The command we ran for this experiment was:

```bash
python3.7 -m src.run \
  --dataset=cifar100 \
  --normalization=norm \
  --num_epochs=300 \
  --max_num_epochs_auxiliary=30 \
  --max_num_epochs_no_val_improvement=50
```

The image above shows the accuracy per epoch on the test set for the baseline 
model trained without curriculum (the model with the suffix "original" [green]),
as well as the accuracy of the curriculum models trained at each level of the 
label hierarchy (identified through the suffixes "level_1" [grey], "level_2"
[orange], etc.). The model trained at the highest level ("level_4" in this case
[red]) is trained on the original label space, and is thus comparable to the 
"original" [green] model.
