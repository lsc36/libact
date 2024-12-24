# libact: Pool-based Active Learning in Python

authors: [Yao-Yuan Yang](http://yyyang.me), Shao-Chuan Lee, Yu-An Chung, Tung-En Wu, Si-An Chen, [Hsuan-Tien Lin](http://www.csie.ntu.edu.tw/~htlin)

[![Build Status](https://travis-ci.org/ntucllab/libact.svg)](https://travis-ci.org/ntucllab/libact)
[![Documentation Status](https://readthedocs.org/projects/libact/badge/?version=latest)](http://libact.readthedocs.org/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/libact.svg)](https://badge.fury.io/py/libact)
[![codecov.io](https://codecov.io/github/ntucllab/libact/coverage.svg?branch=master)](https://codecov.io/github/ntucllab/libact?branch=master)

# Introduction

`libact` is a Python package designed to make active learning easier for
real-world users. The package not only implements several popular active learning strategies, but also features the [active-learning-by-learning](http://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf)
meta-algorithm that assists the users to automatically select the best strategy
on the fly. Furthermore, the package provides a unified interface for implementing more strategies, models and application-specific labelers. The package is open-source along with issue trackers on github, and can be easily installed from Python Package Index repository.

# Documentation

The technical report associated with the package is on [arXiv](https://arxiv.org/abs/1710.00379), and the documentation for the latest release is available on [readthedocs](http://libact.readthedocs.org/en/latest/).
Comments and questions on the package is welcomed at `libact-users@googlegroups.com`. All contributions to the documentation are greatly appreciated!

# Basic Dependencies

* Python 2.7, 3.3, 3.4, 3.5, 3.6, 3.10

* Debian (>= 7) / Ubuntu (>= 14.04)
```
sudo apt-get install build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev
```

* Python dependencies
```
pip install -r requirements.txt
```

* Arch
```
sudo pacman -S lapacke
```

* macOS
```
brew install openblas
```

# Installation

After resolving the dependencies, you may install the package via pip (for all users):
```
sudo pip install libact
```

or pip install in home directory:
```
pip install --user libact
```

or pip install from github repository for latest source:
```
pip install git+https://github.com/ntucllab/libact.git
```

To build and install from souce in your home directory:
```
python setup.py install --user
```

To build and install from souce for all users on Unix/Linux:

**(This is the recommended method for Python 3.10 users)**
```
pip install -e .
```

## Installation Options

- `LIBACT_BUILD_HINTSVM`: set this variable to 1 if you would like to build
  hintsvm c-extension. If set to 0, you will not be able to use the HintSVM
  query strategy. Default=1.
- `LIBACT_BUILD_VARIANCE_REDUCTION`: set this variable to 1 if you would like to
  build variance reduction c-extension. If set to 0, you will not be able to use
  the VarianceReduction query strategy. Default=1.

Example:
```
LIBACT_BUILD_HINTSVM=1 pip install git+https://github.com/ntucllab/libact.git
```

# Usage

The main usage of `libact` is as follows:

```python
qs = UncertaintySampling(trn_ds, method='lc') # query strategy instance

ask_id = qs.make_query() # let the specified query strategy suggest a data to query
X, y = zip(*trn_ds.data)
lb = lbr.label(X[ask_id]) # query the label of unlabeled data from labeler instance
trn_ds.update(ask_id, lb) # update the dataset with newly queried data
```

Some examples are available under the `examples` directory. Before running, use
`examples/get_dataset.py` to retrieve the dataset used by the examples.

Available examples:

  - [plot](examples/plot.py) : This example performs basic usage of libact. It splits
    a fully-labeled dataset and remove some label from dataset to simulate
    the pool-based active learning scenario. Each query of an unlabeled dataset is then equivalent to revealing one labeled example in the original data set.
  - [label_digits](examples/label_digits.py) : This example shows how to use libact in the case
    that you want a human to label the selected sample for your algorithm.
  - [albl_plot](examples/albl_plot.py): This example compares the performance of ALBL
    with other active learning algorithms.
  - [multilabel_plot](examples/multilabel_plot.py): This example compares the performance of
    algorithms under multilabel setting.
  - [alce_plot](examples/alce_plot.py): This example compares the performance of
    algorithms under cost-sensitive multi-class setting.

# Running tests

To run the test suite:

```
python setup.py test
```

To run pylint, install pylint through ```pip install pylint``` and run the following command in root directory:

```
pylint libact
```

To measure the test code coverage, install coverage through ```pip install coverage``` and run the following commands in root directory:

```
coverage run --source libact --omit */tests/* setup.py test
coverage report
```

# Citing
If you find this package useful, please cite the original works (see Reference of each strategy) as well as the following

```
@techreport{YY2017,
  author = {Yao-Yuan Yang and Shao-Chuan Lee and Yu-An Chung and Tung-En Wu and Si-An Chen and Hsuan-Tien Lin},
  title = {libact: Pool-based Active Learning in Python},
  institution = {National Taiwan University},
  url = {https://github.com/ntucllab/libact},
  note = {available as arXiv preprint \url{https://arxiv.org/abs/1710.00379}},
  month = oct,
  year = 2017
}
```


# Acknowledgments

The authors thank Chih-Wei Chang and other members of the [Computational Learning Lab](https://learner.csie.ntu.edu.tw/) at National Taiwan University for valuable discussions and various contributions to making this package better.
