import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_path, test_size=0.2, random_state=42):
    ds = pd.read_csv(data_path)
    train_ds, eval_ds = train_test_split(ds, test_size=test_size, random_state=random_state)
    return train_ds, eval_ds
