import pandas as pd

def get_dataset():
    df = pd.read_csv('./static/data/dataset.csv')
    return df