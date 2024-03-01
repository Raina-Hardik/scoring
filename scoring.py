import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, util

base = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
dataset = pd.read_csv("./80_percent_split.csv")

def calc_sim(cell_1, cell_2):
    return util.dot_score(base.encode(cell_1), base.encode(cell_2))

dataset['sim_score'] = dataset.apply(lambda row: calc_sim(row['desired_answer'], row['student_answer']), axis = 1)
dataset.to_csv('processed.csv', index = False)

X = dataset['sim_score'].values.reshape(-1, 1)
Y = dataset['score_avg'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

class LinReg(nn.Module):
    def __init__(self):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        return self.linear

model = LinReg()
crit = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 2500

for epoch in range(epochs):
    y_pred = model(x_train)

    loss = crit(y_pred, y_train)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}; Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'lin_reg.pth')

# test = model(torch.tensor(X_test, dtype=torch.float32))
