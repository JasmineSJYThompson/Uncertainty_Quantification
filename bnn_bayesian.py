import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import numpy as np

def bnn_model(mu=0, sigma=0.1, dropout_prob=0, width=100, depth=2):
    # Creates the middle hidden layers for the model
    layers = [bnn.BayesLinear(prior_mu=mu, prior_sigma=sigma, in_features=1, out_features=width),
              nn.Tanh(),
              nn.Dropout(p=dropout_prob)]
    # Adds the layers for the depth that we want
    for i in range(depth - 1):
        layers += [bnn.BayesLinear(prior_mu=mu, prior_sigma=sigma, in_features=width, out_features=width),
                   nn.Tanh(), nn.Dropout(p=dropout_prob),]
    
    layers += [bnn.BayesLinear(prior_mu=mu, prior_sigma=sigma, in_features=width, out_features=1)]
    model_nn = nn.Sequential(*layers)
    return model_nn

def train_bnn_model(model_nn, x, y, kl_weight=0.01, learning_rate=0.01, epochs=4000):
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    # kl weight started as 0.01
    kl_weight = kl_weight
    optimizer_nn = optim.Adam(model_nn.parameters(), lr=learning_rate)
    for step in range(epochs):
        pre = model_nn(torch.unsqueeze(x, dim=1))
        mse = mse_loss(pre, torch.unsqueeze(y, dim=1))
        kl = kl_loss(model_nn)
        cost = mse + kl_weight*kl
        optimizer_nn.zero_grad()
        cost.backward()
        optimizer_nn.step()
        if step%500 == 0:
            print(step, "/", epochs)

def get_dist(model, x_test, num_samples=10000):
    models_result = np.array([model(x_test).data.numpy() for k in range(num_samples)])
    models_result = models_result[:,:,0]    
    models_result = models_result.T
    mean_values = np.array([models_result[i].mean() for i in range(len(models_result))])
    std_values = np.array([models_result[i].std() for i in range(len(models_result))])
    return mean_values, std_values