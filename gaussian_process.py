import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super(ExactGPModel, self).__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_model(model, x, y, likelihood, learning_rate=0.01, epochs=100):
    model.train()
    likelihood.train()
    optimizer_gp = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for epoch in range(epochs):
        optimizer_gp.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        optimizer_gp.step()

def get_dist(model, x_test, likelihood):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred_gp = likelihood(model(x_test))
    return y_pred_gp.mean, y_pred_gp.variance