import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import Predictive, MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

class BNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=10, n_hid_layers=5):
        super().__init__()

        self.activation = nn.Tanh() # Chosen activation function

        # Creates a list of the consecutive layer sizes
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        # Creates a list of pyro modulkes the appropriate input and output size
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        # We add all these layers into one collective pyro module list
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        # Sets the weights and the bias for all the appropriate layers
        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0., 0.1).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., 0.1).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.activation(self.layers[0](x)) # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x)) # hidden --> hidden
        mu = self.layers[-1](x).squeeze() # hidden --> output
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1)) # infer the response noise

        # Run a for loop over x.shape[0] so over every value
        with pyro.plate("data", x.shape[0]):
            # Sample the observation given the mu we received and the sigma we are using
            # we get some observation from the mu of what we get from combining everything plus some sigma from the gamma we gave ourselves
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu

def train_pyro_model(model, x, y, num_samples=50):
    # Defines the NUTS kernel to help with MCMC
    nuts_kernel = NUTS(model, jit_compile=False)
    # Define the MCMC sampler
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=num_samples)
    # Run MCMC training on the data we have
    mcmc.run(x, y)
    return mcmc

def train_pyro_mcmc_model(model, x, y, num_epochs=200):
    # Defines the guide for the distribution of the parameters in the model
    mean_field_guide = AutoDiagonalNormal(model)
    # Sets the optimizer as the Adam optimizer with a normal learning rate
    optimizer = pyro.optim.Adam({"lr": 0.01})

    # Defines the stochastic variational interference with the type of guide used,
    # the optimizer and the loss defined
    svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())

    # Clears what we already have stored to save memory
    pyro.clear_param_store()

    # We do a stochastic variational interference training step for each of the epochs
    for epoch in range(num_epochs):
        loss = svi.step(x, y)
    return model, mean_field_guide

def get_dist(model, mcmc, x_test):
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    preds = predictive(x_test)
    mean_values = preds['obs'].T.detach().numpy().mean(axis=1)
    std_values = preds['obs'].T.detach().numpy().std(axis=1)
    return mean_values, std_values

def get_dist_variational(model, x_test, mean_field_guide, num_samples=50):
    predictive = Predictive(model=model, guide=mean_field_guide, num_samples=num_samples)
    preds = predictive(x_test)

    mean_values = preds['obs'].T.detach().numpy().mean(axis=1)
    std_values = preds['obs'].T.detach().numpy().std(axis=1)
    return mean_values, std_values