from ToolBox.lib import testing
from ToolBox.lib import cleaning
import pandas as pd

data = pd.read_csv('raw_data/data', sep=",", header=None)
data.columns = ['text']
data["clean_text"] = data['text'].apply(cleaning)
data["clean_text"]= data['clean_text'].astype('str')
def tester():
    text_test =['I Love football but wht I love most is baskettball where people have fun and jump']
    assert testing(data.clean_text, text_test) == ('topic 1 :', 0.701596149051779)


