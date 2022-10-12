
!pip install --quiet tensorflow-text
!pip install --quiet tokenizers

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import re
import os 
import sys

import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers,Input,Model
from tensorflow.keras.layers import * 
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow_hub as hub 
from tokenizers import BertWordPieceTokenizer
import tensorflow_text as text 

from model import squad_model
from helper_functions import encoder, padder


bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")


input_len = 384

context = 'Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi\'s Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.'
question = 'What venue did Super Bowl 50 take place in?'

def prediction(context, question):
    tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)
    bert_squad = squad_model()
    bert_squad.load_weights('drive/MyDrive/Colab Notebooks/squad/bert_squad_weights.h5')
    ids, type_ids, attention_mask,ids_len = encoder(context,question)
    pad_ids = padder([ids],input_len)
    pad_type_ids = padder([type_ids],input_len)
    pad_attention_mask = padder([attention_mask],input_len)
    preds_start, preds_end = bert_squad.predict([
            (pad_ids), 
            (pad_attention_mask), 
            (pad_type_ids)
            ])
    
    return (' '.join(tokenizer.encode(context).tokens[np.argmax(preds_start):np.argmax(preds_end)]))

print('answer : ',prediction(context, question));
