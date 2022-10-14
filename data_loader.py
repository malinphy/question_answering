import numpy as np 
import pandas as pd 
import json
import re

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
