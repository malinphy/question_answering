# Sentiment Classification

Question answering with BERT model on SQUAD v1.1.

Data
----
SQUAD v1.1. :https://deepai.org/dataset/squad

File Description :
----
- data_loader.py : reading data from json file and converting into pandas dataframe
- helper_funtions.py : encodeing and padding functions
- model.py : nueral network model
- squad_prediction_fast.py : prediction file for deployment purpose 
- train.py : training file 
- requirements.txt : dependencies 


Usage :
----
if necessary download repo and create an virtual env using following commands 
```
conda create --name exp_env
conda activate exp_env
```
find the folder directory in exp_env
```
pip install -r requirements.txt 
```
run ***train.py*** file 
<br/>
train.py file will generate model_weights and squad_prediction.py will use the generated weights for prediction. Since size of the model_weights file is around 1.3 gb. I cannot share it on github.
pr
for deployment purpose prediction file created seperately as ***squad_prediction.py***

```
CONTEXT:
Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.

Examples :
---
QUESTION: Which NFL team represented the AFC at Super Bowl 50?  
ANSWER:Denver Broncos  
PREDICTED ANSWER: denver broncos

QUESTION : What day was the game played on?
ANSWER : February 7, 2016
PREDICTED ANSWER: february 7 , 2016

QUESTION : What venue did Super Bowl 50 take place in?
ANSWER : Levis Stadium in the San Francisco Bay Area at Santa Clara
PREDICTED ANSWER: levi ' s stadium
```
