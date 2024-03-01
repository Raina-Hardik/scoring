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
y = dataset['score_avg'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.from_numpy(X_train, dtype=torch.float32)
y_train_tensor = torch.from_numpy(y_train, dtype=torch.float32)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 2500
for epoch in range(epochs):
    y_pred = model(X_train_tensor)

    loss = criterion(y_pred, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_pred_test = model(X_test_tensor)

mse = criterion(y_pred_test, torch.tensor(y_test, dtype=torch.float32))
print(f'\nMean Squared Error on Test Set: {mse.item():.4f}')

torch.save(model.state_dict(), 'lin_reg.pth')

# test = model(torch.tensor(X_test, dtype=torch.float32))
