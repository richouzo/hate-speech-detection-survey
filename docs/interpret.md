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

- For CNN/RNN-based models, please use this [XAI LSTM notebook](../src/evaluation/explainability_visualization.ipynb) (Example on our best BasicLSTM trained model).

- For BERT-based models, please use this [XAI Bert notebook](../src/evaluation/explainability_visualization_bert.ipynb) (Example on our best DistillBert trained model).

## Details on XAI Bert notebook

Here is the Confusion Matrix of our best trained model `DistillBert_2021-12-08_16-39-08_trained_testAcc=0.7960.pth` used in the XAI Bert notebook:

![DistillBert_CM](../assets/DistillBert_CM_2021-12-08_16-39-08_testAcc=0.7960.png)

#### True Positive

![DistillBert_TP](docs/assets/DistillBert_TP.png)

#### False Positive

![DistillBert_FP](docs/assets/DistillBert_FP.png)

#### True Negative

![DistillBert_TN](docs/assets/DistillBert_TN.png)

#### False Negative

![DistillBert_FN](docs/assets/DistillBert_FN.png)
