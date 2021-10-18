# PerfSAGE: Performance Modeling of Neural Networks using Graph Neural Network

### Author: Yuji Chai, Supervisor: Paul Whatmough
## Overview

This directory contains code for PerfSAGE. PerfSAGE is a performance model of neural networks using graph neural network. It is built for predicting latency of any given neural network running on a hardware platform. Unlike previous methods, such as CNN-based modeling methods, it does not require the target neural network to be constrained in a specific format or structure. And a single model could be used for various neural network models, using different backend and used for different applications. Unlike other previously proposed GNN-based models, it directly use saved TensorFlow model and provide quantitative results for latency prediction. 

Key folders and files are explained in the following list:
* `./data_preprocess`: This folder contains all the scripts and docker files for extracting data from the input datasets. More detailed explanation in provided in the following section. 
* `./perfsage`: This folder contains all the prediction model source code of PerfSAGE. More detailed explanation in provided in the following section.
* `./perfsage_supervised.sh`: This is the bash script to train a PerfSAGE model from a given dataset.
* `./perfsage_extrapolate.sh`: This is a bash script to load a trained PerfSAGE model and test its prediction capability on a given dataset. This could be used for a dataset, which have not been seen during the train process.
* `./data`: This folder stores converted datasets. It is not tracked by git, due to its large size. To use PerfSAGE, it needs to be created locally. 
* `./saves`: This folder stores trained model and visualization plots for its results. It is not tracked by git, due to its large size. To use PerfSAGE, it needs to be created locally. 

## Preprocessor
In this section, we will explain how to use scripts in `./data_preprocess` folder to convert tflite into usable data inputs for the prediction model.

### Docker

If you do not have [docker](https://docs.docker.com/) installed, you will need to do so. (Just click on the preceding link, the installation is pretty painless).  

To run and attach to the docker, use the following command in `./data_preprocess` folder:

	$ bash docker_script.sh

*Note:* This docker file uses the docker image `rmatas/bbb:tf2-nightly-6.20`, please build this docker before running the previous script.

In the docker, run the following script to install additional python packages:

	$ cd /workspace/PerfSAGE/data_preprocess/
	$ bash container_install.sh

### Generating Vela Results

To generate all the vela results from input tflites:
	
	$ python vela_generator.py

### Converting Input Models

With the Vela results generated, convert the input dataset into usable format:
	
	$ python preprocess.py

## Prediction Model
In this section, we will explain how to use scripts in `./` to train and evaluate PerfSAGE models.

### Docker
To build the docker, run the following command:
	
	$ bash docker_script.sh create

To run and attach to the docker, run the following command:

	$ bash docker_script.sh run

### Train a Model

To train a model given a dataset, run the following command:

	$ bash perfsage_supervised.sh

*Note:* The `--train_prefix` flag will be the path to the converted dataset. 

### Evaluate a Model

To evaluate a model's performance given a dataset, run the following command:

	$ bash perfsage_extrapolate.sh

*Note:* The `--train_prefix` flag will be the path prefix to the dataset that the model was trained on. The `--load_model_step` defines step version of the saved model. The `--extrapolate_prefix` indicates which converted dataset you want to evaluate the model on.

## Acknowledgements

The implementation of this project was based on the [GraphSAGE](http://snap.stanford.edu/graphsage/). Their results were published in [paper](https://arxiv.org/pdf/1706.02216.pdf). Their model serves as the backbone for PerfSAGE.

Special thanks to all team members, who contribute to this project, in ARM Research machine learning team. 

## Related Works 

* https://arxiv.org/abs/2008.01040
* http://proceedings.mlr.press/v97/mendis19a.html
* https://ieeexplore.ieee.org/document/8091247?reload=true
* https://arxiv.org/abs/2001.08743v1
* https://arxiv.org/abs/2003.10536v1
* https://iclr.cc/virtual_2020/poster_rkxDoJBYPB.html
* https://papers.nips.cc/paper/2019/hash/71560ce98c8250ce57a6a970c9991a5f-Abstract.html
* https://nips.cc/virtual/2020/public/poster_9f29450d2eb58feb555078bdefe28aa5.html
* https://paperswithcode.com/paper/neural-predictor-for-neural-architecture
* https://arxiv.org/abs/2003.12857
* https://arxiv.org/pdf/1409.4011.pdf
* https://arxiv.org/pdf/1812.00332.pdf