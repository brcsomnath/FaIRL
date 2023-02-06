# Fairness-aware Incremental Representation Learning (FaIRL)

We present the implementation of the AAAI 2023 paper : 

> [**Sustaining Fairness via Incremental Learning**](https://arxiv.org/pdf/2208.12212.pdf),<br/>
[Somnath Basu Roy Chowdhury](https://www.cs.unc.edu/~somnath/) and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/), <br>UNC Chapel Hill


## Dataset

We prepare the Biased MNIST dataset using `src/mnist_data_create.py`.

Data for the Biographies dataset  is obtained from [https://github.com/microsoft/biosbias](https://github.com/microsoft/biosbias).

To access the exact data in our experiments: Biased MNIST (is available [here](https://www.cs.unc.edu/~somnath/FaIRL/data/)) and Biographies dataset (available [here](https://github.com/brcsomnath/FaRM)).

## Setting up the environment

* __Python version:__ `python 3.8.5`

* __Dependencies:__ To install the dependencies using conda, please follow the steps below:

		conda create -n fairl python=3.8.5
        source activate fairl
		pip install -r requirements.txt 

## Running Experiments

Running the main experiments on Biographies dataset.

```
cd src/fairl/
python bios.py \
        --device cuda:0  \
        --num_target_class 28 \
        --num_protected_class 2 \
        --exemplar_selection prototype
```

Running the main experiments in various configurations of the Biased MNIST dataset.

```
cd src/fairl/
python mnist.py \
        --device cuda:0  \
        --num_target_class 10 \
        --num_protected_class 10 \
        --dataset .8 \
        --exemplar_selection prototype
```

## Reference


```
@article{fairl,
  title = {Sustaining Fairness via Incremental Learning},
  author = {Basu Roy Chowdhury, Somnath and Chaturvedi, Snigdha},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {37},
  number = {1},
  year = {2023}
}
```