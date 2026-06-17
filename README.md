# CDA-case1

This is the repository for case 1 project for [02582 Computational Data Analysis](https://kurser.dtu.dk/course/02582) course from [DTU](https://www.dtu.dk/).

The project goal was to build a predictive model for a response vector Y based on a 100-dimensional feature matrix X (100x1 Y responses and 100x100 feature matrix was given for this purpose). No domain information was given and data analyses was inconclusive. For this reason hyperparameter sweep with robust cross-validation was performed to find best configuration of pre-processing and model choice. The pre-processing and model training is handled  in [scikit-learn](https://scikit-learn.org/stable/) pipeline format. [Hydra](https://hydra.cc/) is used for hyperparameter sweep configuration. The experiment tracking and logging is integrated via [Weights&Biases](https://wandb.ai/site/).

The data analyses for this project is located in the "/notebook/" folder. It ws conducted thourghly, but unfortunately due to the lack of the domain knowledge (no context was given to the data) it was inconclusive.

The cross-validation pipeline is located in the "/src/case1/" folder. The  sweep configuration is located in "/configs/" folder.

[Uv](https://docs.astral.sh/uv/) is used as virtual environment (libraries/packages) manager. The project was tested on a few Windows machines, and dependencies worked as expected. Only "uv sync" (see Installation) command was needed to construct needed virtual environment for repository run.

## Coookiecutter info

This structure was created by modifying this [cookiecutter template](https://github.com/InfiniteLobster/cookiecutter-matlab-project).

## Installation
To use this GitHub repository it first needs to be cloned locally as follows (standard code for git repository cloning):
```bash
git clone https://github.com/InfiniteLobster/CDA-case1
```
After the cloning, the virtual environment needs to be configured. In this project [uv](https://docs.astral.sh/uv/) is used for this purpose. With it installed on the machine following code needs to be executed for this purpose  (standard code for uv):
```bash
uv sync
```
## Requirements
* [git](https://git-scm.com/)
* [uv](https://docs.astral.sh/uv/)
* python version: 3.12
## Cross-validation python file description
* data.py - in this file functions related to the dataset handling are located.
* evaluate.py - placeholder file for the model evaluation iterface. Phased-out due to lack of need in the context of the project. It is left as a potential place for further expansion.
* model.py - in this file model picking and configuration, depending on given parameters (configuration information handled by [hydra](https://hydra.cc/)) is handled. It is composed of single funcion, but it is put in separate file for the project code logic clarity. This function creates output for model part of the pipeline created by code in pipeline.py. In practice this code is used in the pipeline.py (which is used in train.py).
* pipeline.py - in this file the functions handling [scikit-learn](https://scikit-learn.org/stable/) pipeline creation is located. It relies on the pre-processing (preprocessing.py) and model (model.py) handling functions from other files. In practice this code is used in the train.py.
* predict.py - this is file used for the predictions on the dataset by trained model. It is not function file as previous, but one to be used directly.
* preprocessing.py - in this file pre-processing method picking depending on given parameters (configuration information handled by [hydra](https://hydra.cc/)) is handled. There is couple of functions to do that in this file. These functions creates output for pre-processing part of the pipeline created by code in pipeline.py. In practice this code is used in the pipeline.py (which is used in train.py).
* train.py - this is the file used for the hyperparameter sweep with cross-validation. It is the main file of the project and repository. It performs cross-validation based on the given configuration (via hydra) and logs all the results (to the [Weights&Biases](https://wandb.ai/site/)).
## Authors
* s253711 - [InfiniteLobster](https://github.com/InfiniteLobster/)
* s260067 - [yukthadabke](https://github.com/yukthadabke)
## Version

1.0.0 (Created: 2026-02-20)

## License
MIT License

