from ToolBox.lib import testing
import pandas as pd
data = pd.read_csv('raw_data/data', sep=",", header=None)
data.columns = ['text']
def tester():
    text_test=['I Love football but wht I love most is baskettball where people have fun and jump']

    assert (testing(data, text_test) == )
