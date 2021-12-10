# 🤬 A Comparative Study on NLP algorithms for Hate Speech Detection

**Context**: Final Project for CS7643/CS7650 courses at Georgia Tech, Fall 2021

**Authors**: Aymane Abdali (CS7643/CS7650), Maya Boumerdassi (CS7643), Mohamed Amine Dassouli (CS7643/CS7650), Jonathan Hou (CS7643), Richard Huang (CS7643/CS7650)

## Installation phase

Please refer to [install.md](docs/install.md).

## Dataset: Offensive Language Identification Dataset - OLID 

[Dataset Paper](https://arxiv.org/abs/1902.09666) |
[Dataset Link1](https://scholar.harvard.edu/malmasi/olid) |
[Dataset Link2](https://sites.google.com/site/offensevalsharedtask/offenseval2019)

For this study, we use the sub-task A of the OLID Dataset. This dataset contains English tweets annotated using a three-level annotation hierarchy and was used in the OffensEval challenge in 2019. 

Preprocessing functions can be found in [preprocess_utils.py](src/utils/preprocess_utils.py).

## Training phase

Please refer to [training.md](docs/training.md).

## Trained models

All of our best trained models can be found [here](https://1drv.ms/u/s!Ak4YJhU8zi9qrzdQT5BFOXCfVQ3A?e=xJPiJm).

## Word importance with Captum

This is the qualitative part of our study. We use [Captum](https://captum.ai/) to interpret our trained models. Here are some examples from our best trained model `DistillBert_2021-12-08_16-39-08_trained_testAcc=0.7960.pth` on the test set:

#### True Positive

![DistillBert_TP](docs/assets/DistillBert_TP.png)

#### False Positive

![DistillBert_FP](docs/assets/DistillBert_FP.png)

Please refer to [interpret.md](docs/interpret.md) for more details.
