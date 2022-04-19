import pandas as pd
def dummies(column,data):
    return pd.get_dummies(data[column])