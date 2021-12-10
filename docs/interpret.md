# XAI Interpretability instructions

[Main README](../README.md)

All commands should be used from root directory.

## Retrieve stats for visualisation

Stats csv files are saved in `stats-results/` folder, run this command before running the notebooks:

```bash
### Example on our best BasicLSTM trained model
python -m src.evaluation.test_save_stats --model BasicLSTM --saved_model_path saved-models/BasicLSTM_2021-12-08_01-04-25_trained_testAcc=0.7107.pth --loss_criterion bcelosswithlogits --only_test 0 --stats_label 1
```

## Word importance visualisations with Captum:

We provide two notebooks to visualize which parts of the input sentence are used for an inference of a trained model.

In the current state, we use Integrated Gradients from [Captum](https://captum.ai/) library to obtain the attribution scores for each word in a given sentence. 

- For CNN/RNN-based models, please use this [notebook](../src/evaluation/explainability_visualization.ipynb) (Example on our best BasicLSTM trained model).

- For BERT-based models, please use this [notebook](../src/evaluation/explainability_visualization_bert.ipynb) (Example on our best DistillBert trained model).
