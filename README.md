# A survey on Hate Speech Detection using NLP algorithms
     Final Project CS7643/CS7650 at Georgia Tech, Fall 2021

Authors: Aymane Abdali (CS7643/CS7650), Maya Boumerdassi (CS7643), Mohammed Amine Dassouli (CS7643/CS7650), Jonathan Hou (CS7643), Richard Huang (CS7643/CS7650)

## Installation

Tested with Python 3.8.12

Install PyTorch first.  
Please refer to [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

After PyTorch has finished installing, you can install from source by cloning the repository and run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Training

Start training:

```bash
python main.py
```


## Gridsearch training

You can modify the gridsearch parameters in *gridsearch_config.yaml*

Gridsearch:

```bash
python gridsearch.py
```
