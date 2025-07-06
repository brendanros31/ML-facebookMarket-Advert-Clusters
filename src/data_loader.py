import pandas as pd


# Loading data
def load_data(path):
    df = pd.read_csv(path)
    return df
