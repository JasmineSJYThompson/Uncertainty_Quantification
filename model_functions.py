import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

def bnn_model(mu=0, sigma=0.1, width=100):
    model_nn = nn.Sequential(
        bnn.BayesLinear(prior_mu=mu, prior_sigma=sigma, in_features=1, out_features=width),
        nn.ReLU(),
        nn.Dropout(p=dropout_prob),
        bnn.BayesLinear(prior_mu=mu, prior_sigma=sigma, in_features=width, out_features=width),
        nn.Tanh(),
        nn.Dropout(p=dropout_prob),
        bnn.BayesLinear(prior_mu=mu, prior_sigma=sigma, in_features=width, out_features=1),
    )
    return model_nn