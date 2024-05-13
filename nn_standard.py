import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_layer_size=100, num_hidden_layers=2, dropout_p=0):
        super(NeuralNetwork, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

def train_nn_model(model, x, y, learning_rate=0.01, epochs=4000):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

def get_dist(model, x_test, num_samples=10000):
    # Prediction
    model.train()  # Set to training mode to enable dropout during inference
    with torch.no_grad():
        outputs = torch.cat([model(x_test) for _ in range(num_samples)], dim=1)
        mean_prediction = outputs.mean(dim=1).numpy()
        std_prediction = outputs.std(dim=1).numpy()
    return mean_prediction, std_prediction