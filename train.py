import os 
import json
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

# from data_loader import df_maker,answer_extractor

class str_maker :

    def __init__(self, data):
        self.data = data 

    def splitter(self):
        bos = []
        for i in self.data:
            bos.append(' '.join(i.split()))

        return bos



class df_maker :
    def __init__(self,json_path):
        self.json_path = json_path
        # self.test_path = test_path

    def squad_json_to_dataframe(self, record_path=['data','paragraphs','qas','answers']):
        """
        input_file_path: path to the squad json file.
        record_path: path to deepest level in json file default value is
        ['data','paragraphs','qas','answers']
        """
        file = json.loads(open(self.json_path).read())
        # parsing different level's in the json file
        js = pd.json_normalize(file, record_path)
        m = pd.json_normalize(file, record_path[:-1])
        r = pd.json_normalize(file,record_path[:-2])
        # combining it into single dataframe
        idx = np.repeat(r['context'].values, r.qas.str.len())
        m['context'] = idx
        data = m[['id','question','context','answers']].set_index('id').reset_index()
        data['c_id'] = data['context'].factorize()[0]
        return data 

    # def answer_extractor(x):

#train_data = df_maker(train_path).squad_json_to_dataframe()
#test_data = df_maker(test_path).squad_json_to_dataframe()


class answer_extractor :
    def __init__(self,x):
        self.x = x
    def answer_extractor_2(self):
        answer_str= []
        answer_str_pos = []
        
        for i in range(len(self.x)):
        
            var1 = self.x[i]
            var2 = re.sub('\[{|\'}]','',str(var1))
            var3 = var2.split(':')[-1].strip()
            answer_str.append(re.sub('\'','',var3))

            answer_str_pos.append(int(re.sub(',','',str(var1).split(' ')[1])))

        return(answer_str,answer_str_pos)


def answer_cleaner(answer):
    an_new = []
    for i in range(len(answer)):

        an_new.append(re.sub(r'"|}]','',answer[i]))
    return an_new

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)


input_len = 384

train_path = 'drive/MyDrive/Colab Notebooks/datasets/squad/train-v1.1.json'
test_path = 'drive/MyDrive/Colab Notebooks/datasets/squad/dev-v1.1.json'

# train_path = 'train-v1.1.json'
# test_path = 'dev-v1.1.json'

train_data = df_maker(train_path).squad_json_to_dataframe()
test_data = df_maker(test_path).squad_json_to_dataframe()

train_answer_text,train_answer_pos = answer_extractor(train_data['answers']).answer_extractor_2()
test_answer_text,test_answer_pos = answer_extractor(test_data['answers']).answer_extractor_2()

train_data = train_data.drop(columns = ['id', 'c_id'])
train_data['answers'] = train_answer_text
train_data['starting_idx'] = train_answer_pos

test_data = test_data.drop(columns = ['id', 'c_id'])
test_data['answers'] = test_answer_text
test_data['starting_idx'] = test_answer_pos
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

train_data['question'][0]

train_answer_str = str_maker(train_data['answers']).splitter()
train_question_str = str_maker(train_data['question']).splitter()
train_context_str = str_maker(train_data['context']).splitter()

test_answer_str = str_maker(test_data['answers']).splitter()
test_question_str = str_maker(test_data['question']).splitter()
test_context_str = str_maker(test_data['context']).splitter()

train_answer_str  = answer_cleaner(train_answer_str)
test_answer_str  = answer_cleaner(test_answer_str)

# train_answer_str

# answer_cleaner(train_answer_str)

# train_answer_str

def token_positioner(x):
    first_pos = []
    last_pos = []
    for i in range(len(x)):
        context_ids = tokenizer.encode(x['context'][i]).ids
        answer_ids = tokenizer.encode(x['answers'][i]).ids[1:-1]
        pos = (np.where(np.in1d(context_ids,answer_ids)== True)[0])
        # print(i)
        if len(pos) != 0 :
            first_pos.append(pos[0])
        # last_pos_train.append(pos[-1])
            last_pos.append(pos[0]+len(answer_ids))

        else :
            first_pos.append(-1)
            last_pos.append(-1)
        # pos.append

    return first_pos, last_pos

first_pos_train,last_pos_train = token_positioner(train_data)
first_pos_test,last_pos_test = token_positioner(test_data)

dropper = np.where(np.array(first_pos_train) == -1)[0]
dropper2 = np.where(np.array(last_pos_train) > input_len-1)[0]
train_data  = train_data.drop(dropper)
train_data  = train_data.drop(dropper2).reset_index(drop = True)

train_answer_str = str_maker(train_data['answers']).splitter()
train_question_str = str_maker(train_data['question']).splitter()
train_context_str = str_maker(train_data['context']).splitter()

test_answer_str = str_maker(test_data['answers']).splitter()
test_question_str = str_maker(test_data['question']).splitter()
test_context_str = str_maker(test_data['context']).splitter()

first_pos_train,last_pos_train = token_positioner(train_data)
first_pos_test,last_pos_test = token_positioner(test_data)

train_data['first_pos'] = first_pos_train
train_data['last_pos'] = last_pos_train

test_data['first_pos'] = first_pos_test
test_data['last_pos'] = last_pos_test

def encoder(x,y):
    tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)
    ids = []
    type_ids = []
    # tokens = []
    # offsets = []
    attention_mask = []
    # special_tokens_mask =[]
    ids_len = []
    for i in range(len(x)):
        var1 = tokenizer.encode(x[i],y[i])
        ids.append(var1.ids)
        type_ids.append(var1.type_ids)
        # tokens.append(var1.tokens)
        # offsets.append(var1.offsets)
        attention_mask.append(var1.attention_mask)
        # special_tokens_mask.append(var1.special_tokens_mask)
        ids_len.append(len(var1.ids))
    return ids, type_ids, attention_mask,ids_len

train_ids, train_type_ids, train_attention_mask, train_ids_len = encoder(train_context_str,train_question_str)
test_ids, test_type_ids , test_attention_mask,  test_ids_len = encoder(test_context_str,test_question_str)

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

pad_ids_train = padder(train_ids,input_len)
pad_type_ids_train = padder(train_type_ids,input_len)
pad_attention_mask_train = padder(train_attention_mask,input_len)
# pad_special_tokens_mask_train = padder(train_special_tokens_mask, input_len)

pad_ids_test = padder(test_ids,input_len)
pad_type_ids_test = padder(test_type_ids,input_len)
pad_attention_mask_test = padder(test_attention_mask,input_len)
# pad_special_tokens_mask_test = padder(test_special_tokens_mask, input_len)

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
def squad_model():
    input_word_ids = Input(shape=(input_len,), dtype=tf.int32, name='input_word_ids')
    input_mask = Input(shape=(input_len,), dtype=tf.int32, name='input_mask')
    input_type_ids = Input(shape=(input_len,), dtype=tf.int32, name='input_type_ids')

    # bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])

    start_flatten = Flatten()(sequence_output)
    start_final = Dense(input_len, activation = 'softmax')(start_flatten)

    end_flatten = Flatten()(sequence_output)
    end_final = Dense(input_len, activation = 'softmax')(start_flatten)


    return  Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[start_final, end_final])

bert_squad = squad_model()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
bert_squad.compile(optimizer=optimizer, loss=[loss, loss],
              metrics = ['accuracy'])

# history = bert_squad.fit([
#             (pad_ids_train), 
#             (pad_attention_mask_train), 
#             (pad_type_ids_train)
#             ],
#             ([
#             np.array(first_pos_train),
#             np.array(last_pos_train)
#             ]),
#             epochs = 2,
#             batch_size = 4,
#             )

# bert_squad.save_weights('drive/MyDrive/Colab Notebooks/squad/bert_squad_weights_2epochs.h5')

bert_squad.load_weights('drive/MyDrive/Colab Notebooks/squad/bert_squad_weights.h5')
