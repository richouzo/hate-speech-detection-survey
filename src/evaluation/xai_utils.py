import tqdm
import argparse
import numpy as np
import datetime
import time

import spacy
import pandas as pd
from sklearn.metrics import f1_score

from torch import optim
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils.preprocess_utils import *
from src.training.train_utils import train_model, test_model
from src.models import BasicLSTM, BiLSTM
from src.evaluation.test_save_stats import *

from src.utils.utils import *

import captum
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from typing import Any, Iterable, List, Tuple, Union
from IPython.core.display import HTML, display


"""
CNN, LSTM models' XAI methods
"""

def model_explainability(interpret_sentence, lig, model, vocab_stoi, vocab_itos, 
                         df, max_samples, field, device, class_names=["Neutral","Hate"]):
    """
    Computing words importance for each sample in df
    """
    print("\n\n**MODEL EXPLAINABILITY**\n")
    print("Computing words importance for each sample... ")

    pad_ind = field.vocab.stoi[field.pad_token]-1 #vocab_stoi[field.pad_token]
    token_reference = TokenReferenceBase(reference_token_idx=pad_ind)

    # accumalate couple samples in this array for visualization purposes
    vis_data_records_ig = []
    
    for idx in range(max_samples):
        sentence = df.iloc[idx].text
        label = df.iloc[idx].true_label
        original_idx = df.iloc[idx].original_index
        input_tokens = sentence_to_input_tokens(sentence, vocab_stoi)
        with torch.set_grad_enabled(True):
            interpret_sentence(model, field, pad_ind=pad_ind, input_data=input_tokens, sentence=sentence, vocab_stoi=vocab_stoi, \
                               vocab_itos=vocab_itos, device=device, original_idx=original_idx, 
                               vis_data_records_ig=vis_data_records_ig,\
                               token_reference=token_reference, lig=lig, min_len=7, label=label, \
                               class_names=class_names)
    
    print("Computations completed.")
    return vis_data_records_ig

def sentence_to_input_tokens(sentence, vocab_stoi):
    input_tokens = []
    for word in sentence.split(" "):
        token = vocab_stoi[word]
        input_tokens.append(token)
    input_tokens= torch.tensor(input_tokens).unsqueeze(0).permute(1, 0)
    return input_tokens

def dataset_visualization(interpret_sentence, lig, visualize_text, model, vocab_stoi, vocab_itos, df,\
                           field, device, max_samples=10,partial_vis=False,class_names=["Neutral","Hate"]):
    n = len(df)
    if partial_vis:
        n = min(n,max_samples)
    
    vis_data_record = model_explainability(interpret_sentence, lig, model, vocab_stoi, vocab_itos, df, n,\
                                           field, device, class_names=class_names)
    print("\n\n**LOADING VISUALIZATION**\n")
    visualize_text(vis_data_record)

"""
BERT's XAI methods
"""

def model_explainability_bert(interpret_sentence, lig, model, df, max_samples, device):
    """
    Computing words importance for each sample in df
    """
    print("\n\n**MODEL EXPLAINABILITY**\n")
    print("Computing words importance for each sample... ")

    # accumalate couple samples in this array for visualization purposes
    vis_data_records_ig = []

    for idx in range(max_samples):
        sentence = df.iloc[idx].text
        label = df.iloc[idx].true_label
        original_idx = df.iloc[idx].original_index
        with torch.set_grad_enabled(False):
            interpret_sentence(model, sentence, label, original_idx, vis_data_records_ig, device)

    print("Computations completed.")
    return vis_data_records_ig

def dataset_visualization_bert(interpret_sentence, lig, visualize_text, model, df, 
                               device, max_samples=10,partial_vis=False):
    n = len(df)
    if partial_vis:
        n = min(n,max_samples)

    vis_data_record = model_explainability_bert(interpret_sentence, lig, model, df, n, device)

    print("\n\n**LOADING VISUALIZATION**\n")
    visualize_text(vis_data_record)

"""
Visualization methods
"""

def format_classname(classname):

    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)

def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_special_tokens(word)
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)

def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        #(52, 85%, 69%);
        # red
        hue = 0
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        #yellow
        hue = 52
        sat = 85
        lig = 100 - int(-40 * attr)
        
#     attr = max(-1, min(1, attr))
#     hue = 0
#     sat = 75
#     lig = np.clip(100 - int(50 * attr), 0, 100)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)

class VisualizationDataRecordCustom:
    """
    A data record for storing attribution relevant information
    """
    def __init__(
        self,
        word_attributions,
        pred_prob,
        pred_class,
        true_class,
        attr_class,
        attr_score,
        raw_input_ids,
        convergence_score,
        original_idx,
    ) -> None:
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class
        self.true_class = true_class
        self.attr_class = attr_class
        self.attr_score = attr_score
        self.raw_input_ids = raw_input_ids
        self.convergence_score = convergence_score
        self.original_idx = original_idx

def visualize_text(
    datarecords, legend: bool = True
) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
    HAS_IPYTHON = True
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>Original Index</th>"
        "<th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
       
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname(datarecord.original_idx),
                    format_classname(datarecord.true_class),
                    format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    format_classname("{0:.2f}".format(datarecord.attr_score)),
                    format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1,1], ["Neutral", "Hate"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html = HTML("".join(dom))
    display(html)

    return html

