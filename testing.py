import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
base = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Load the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(input_size=1, output_size=1)
model.load_state_dict(torch.load('lin_reg.pth'))
model.eval()

# Function to calculate similarity score
def calc_sim(sentence1, sentence2):
    return util.dot_score(base.encode(sentence1), base.encode(sentence2))

# Get user input for two sentences
sentence1 = input("Enter the first sentence: ")
sentence2 = input("Enter the second sentence: ")

# Calculate similarity score
similarity_score = calc_sim(sentence1, sentence2)

# Standardize the similarity score using the previously used scaler
scaler = StandardScaler()
similarity_score_standardized = scaler.fit_transform([[similarity_score]])[0][0]

# Convert the standardized score to a PyTorch tensor
similarity_score_tensor = torch.FloatTensor([[similarity_score_standardized]])

# Make a prediction using the linear regression model
prediction_tensor = model(similarity_score_tensor)
predicted_score = prediction_tensor.item()

print(f"Similarity Score: {similarity_score}")
print(f"Predicted Score: {predicted_score}")
