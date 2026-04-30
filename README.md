# CDA-case1

This is the repository for case 1 project for [02582 Computational Data Analysis](https://kurser.dtu.dk/course/02582) course from [DTU](https://www.dtu.dk/).

The project goal was to build a predictive model for a response vector Y based on a 100-dimensional feature matrix X (100x1 Y responses and 100x100 feature matrix was given for this purpose). No domain information was given and data analyses was inconclusive. For this reason robust cross-validation was performed to find best configuration of pre-processing and model choice. The pre-processing and model training is handled in Scikit-learn pipeline format. Hydra is used for hyperparameter sweep configuration. The experiment tracking and logging is integrated via Weight&Biases.

[Uv](https://docs.astral.sh/uv/) is used as virtual environment (libraries/packages) manager. The project was tested on a few Windows machines, and dependencies worked as expected. Only "uv sync" (see Installation) command was needed to construct needed virtual environment for repository run.

## Coookiecutter info

This structure was created by modifying this [cookiecutter template](https://github.com/InfiniteLobster/cookiecutter-matlab-project)

## Installation
To use this GitHub repository it first needs to be cloned locally as follows (standard code for git repository cloning):
```bash
git clone https://github.com/InfiniteLobster/CDA-case2
```
After the cloning, the virtual environment needs to be configured. In this project [uv](https://docs.astral.sh/uv/) is used for this purpose. With it installed on the machine following code needs to be executed for this purpose  (standard code for uv):
```bash
uv sync
```
## Requirements
* [git](https://git-scm.com/)
* [uv](https://docs.astral.sh/uv/)
* python version: 3.12
## Authors
* s253711 - [InfiniteLobster](https://github.com/InfiniteLobster/)
* s260067 - [yukthadabke](https://github.com/yukthadabke)
## Version

1.0.0 (Created: 2026-02-20)

## License

MIT License

