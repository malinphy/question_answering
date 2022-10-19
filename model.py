#### question answering with BERT on squad v1.1 dataset

#!pip install --quiet tensorflow-text
#!pip install --quiet tokenizers

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers,Input,Model
from tensorflow.keras.layers import * 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_hub as hub 
from tokenizers import BertWordPieceTokenizer
import tensorflow_text as text 


def squad_model():
    input_len = 384
    input_word_ids = Input(shape=(input_len,), dtype=tf.int32, name='input_word_ids')
    input_mask = Input(shape=(input_len,), dtype=tf.int32, name='input_mask')
    input_type_ids = Input(shape=(input_len,), dtype=tf.int32, name='input_type_ids')

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])

    start_flatten = Flatten()(sequence_output)
    start_final = Dense(input_len, activation = 'softmax')(start_flatten)

    end_flatten = Flatten()(sequence_output)
    end_final = Dense(input_len, activation = 'softmax')(start_flatten)


    return  Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[start_final, end_final])