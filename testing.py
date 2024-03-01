import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
base = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

def calc_sim(cell_1, cell_2):
    return util.dot_score(base.encode(cell_1), base.encode(cell_2))
    
def get_output(cell):
    return model(cell)

data = pd.read_csv('20_percent.csv')
data['sim_score'] = data.apply(lambda row: calc_sim(row['desired_answer'], row['student_answer']), axis = 1)

model = LinearRegressionModel(input_size=1, output_size=1)
model.load_state_dict(torch.load('lin_reg.pth'))
model.eval()

data['ouputs'] = data.apply(lambda row: get_output(row['sim_score']), axis = 1)

from sklearn.metrics import mean_squared_error

predicted_values = data['outputs']
true_values = data['score_avg']

rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
