# Training instructions

[Main README](../README.md)

All commands should be run from root directory.

## Training

Figures and saved models are saved after training in their respective folder.

```bash
### Start training | Default to BasicLSTM
python -m src.training.main
```


## Gridsearch training

Gridsearch csv files are saved in `gridsearch-results/` folder.

You can modify the gridsearch parameters in [gridsearch_config.yml](../gridsearch_config.yml) before running this command:

```bash
### Start gridsearch
python -m src.training.gridsearch
```

| Hyperparameters      | Possible values |
| ----------- | ----------- |
| model_type  | ['BasicLSTM', 'BiLSTM', 'HybridCNNLSTM', 'HybridLSTMCNN', <br />'DistillBert', 'DistillBertEmotion', 'PyramidCNN', 'TinyBert']       |
| optimizer_type   | ['adam', 'adamw', 'sgd']        |
| loss_criterion   | ['bceloss', 'bcelosswithlogits', 'crossentropy']        |
| lr   | [*float*]        |
| epochs   | [*int*]        |
| batch_size   | [*int*]        |
| patience_es   | [*int*]        |
| scheduler_type   | ['', reduce_lr_on_plateau', <br />'linear_schedule_with_warmup']        |
| patience_lr   | [*int*]        |
| save_condition   | ['loss', 'acc']        |
| fix_length   | [*null* or *int*]        |


## Testing

To print the model size parameters, loss and accuracy on the test set, run this command:

```bash
### Example on our best BasicLSTM trained model
python -m src.evaluation.test_save_stats --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2021-12-08_01-04-25_trained_testAcc=0.7107.pth --loss_criterion bcelosswithlogits --only_test 1
```
