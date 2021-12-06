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

Figures and saved models are saved after training in their respective folder.

Start training:

```bash
python main.py
```


## Gridsearch training

Gridsearch csv files are saved in gridsearch_results folder.

You can modify the gridsearch parameters in [gridsearch_config.yml](gridsearch_config.yml).

Start gridsearch:

```bash
python gridsearch.py
```

| Hyperparameters      | Possible values |
| ----------- | ----------- |
| model_type  | ['BasicLSTM', 'BiLSTM', <br />'HybridCNNLSTM', 'HybridLSTMCNN', <br />'DistillBert', 'DistillBertEmotion']       |
| optimizer_type   | ['adam', 'adamw', 'sgd']        |
| loss_criterion   | ['bceloss', 'bcelosswithlogits', 'crossentropy']        |
| lr   | [*float*]        |
| epochs   | [*int*]        |
| batch_size   | [*int*]        |
| patience_es   | [*int*]        |
| scheduler_type   | ['', reduce_lr_on_plateau', <br />'linear_schedule_with_warmup']        |
| patience_lr   | [*int*]        |
| save_condition   | ['loss', 'acc']        |


## Retrieve stats for visualisation

Stats csv files are saved in stats_results folder.

Examples:

- For BiLSTM:
    ```bash
    python test_save_stats.py --model BiLSTM --saved_model_path saved_models/BiLSTM_2021-12-03_23-58-08_trained_testAcc=0.5561.pth --loss_criterion bcelosswithlogits --stats_label 1
    ```

- For DistillBert:
    ```bash
    python test_save_stats.py --model DistillBert --saved_model_path saved_models/DistillBert_2021-12-05_17-03-43_trained_testAcc=0.6058.pth --loss_criterion crossentropy --stats_label 1
    ```
