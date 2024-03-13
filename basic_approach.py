import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer, util

base = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
dataset = pd.read_csv("./80_percent_split.csv")

def calc_sim(cell_1, cell_2):
    return util.dot_score(base.encode(cell_1), base.encode(cell_2))

dataset['sim_score'] = dataset.apply(lambda row: calc_sim(row['desired_answer'], row['student_answer']), axis=1)
dataset['sim_score'] = dataset['sim_score'].apply(lambda x: x.item())
MAX_VAL = dataset['score_avg'].max()
dataset['predicted_score'] = dataset.apply(lambda row: row['sim_score'] * MAX_VAL, axis=1)
dataset.to_csv('processed.csv', index=False)

mae = np.mean(np.abs(dataset['predicted_score'] - dataset['score_avg']))
rmse = np.sqrt(np.mean((dataset['predicted_score'] - dataset['score_avg'])**2))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Square Error (RMSE):", rmse)
