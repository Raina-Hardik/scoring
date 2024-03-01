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

print("Sim Score Done!")

from sklearn.preprocessing import StandardScaler

input_var = 'sim_score'
output_var = 'score_avg'

X = dataset[[input_var]].values
y = dataset[output_var].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset Split")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

input_size = X_train.shape[1]
output_size = 1
model = LinearRegressionModel(input_size, output_size)

print("Model Instantiated")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Training Start")

num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

X_test_tensor = torch.FloatTensor(X_test)
predictions_tensor = model(X_test_tensor)

predictions = predictions_tensor.detach().numpy()

torch.save(model.state_dict(), 'lin_reg.pth')

# test = model(torch.tensor(X_test, dtype=torch.float32))
