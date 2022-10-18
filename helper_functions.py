#### squad helper functions

import tensorflow as tf 
import tensorflow_hub as hub 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizers import BertWordPieceTokenizer


bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)


def encoder(x,y):
    tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)
    ids = []
    type_ids = []
    attention_mask = []
    ids_len = []
    # for i in range(len(x)):
    var1 = tokenizer.encode(x,y)
    ids=(var1.ids)
    type_ids = (var1.type_ids)
    attention_mask = (var1.attention_mask)
    ids_len = (len(var1.ids))
    return ids, type_ids, attention_mask,ids_len

def padder(x,pad_len):
    padded_var = pad_sequences(
    x,
    maxlen=pad_len,
    dtype='int32',
    padding='post',
    truncating='post',
    value=0.0
    )
    return padded_var
