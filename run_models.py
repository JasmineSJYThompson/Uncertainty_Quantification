import time
import gpytorch
import make_data
import eval_data
import bnn_bayesian
import pyro_bayesian
import nn_standard
import gaussian_process

def run_model(x, y, x_test, y_test, y_perfect, model_type="bnn_variational", function_type=None,
              dropout_prob=0.1, epochs=100, samples=100, mu=0, sigma=0.1, width=10, depth=2, reverse_plot=False, figsize=(10, 8)):
    filename = (f"./Graphs/{model_type}_{function_type}_{width}_{depth}_{dropout_prob}_{epochs}_{samples}_{mu}_{sigma}.png")
    title = f"{" ".join(model_type.split("_")).title()} {" ".join(function_type.split("_"))}, Width: {width} Depth: {depth} Dropout: {dropout_prob} Epochs: {epochs} Samples: {samples}\n"
    start = time.perf_counter()
    if model_type=="bnn_variational":
        model = bnn_bayesian.bnn_model(mu=mu, sigma=sigma, dropout_prob=dropout_prob, width=width, depth=depth)
        bnn_bayesian.train_bnn_model(model, x, y, epochs=epochs)
        mean_values, std_values = bnn_bayesian.get_dist(model, x_test, num_samples=samples)
    elif model_type=="bnn_MCMC":
        model = pyro_bayesian.BNN(hid_dim=width, n_hid_layers=depth)
        mcmc = pyro_bayesian.train_pyro_model(model, x, y, num_samples=epochs)
        mean_values, std_values = pyro_bayesian.get_dist(model, mcmc, x_test)
    elif model_type=="bnn_variational_pyro":
        model = pyro_bayesian.BNN(hid_dim=width, n_hid_layers=depth)
        model, mean_field_guide = pyro_bayesian.train_pyro_mcmc_model(model, x, y, num_epochs=epochs)
        mean_values, std_values = pyro_bayesian.get_dist_variational(model, x_test, mean_field_guide, num_samples=samples)
    elif model_type=="standard_nn":
        model = nn_standard.NeuralNetwork(dropout_p=dropout_prob, num_hidden_layers=depth, hidden_layer_size=width)
        nn_standard.train_nn_model(model, x, y, epochs=epochs)
        mean_values, std_values = nn_standard.get_dist(model, x_test, num_samples=samples)
    elif model_type=="gaussian_process":
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = gaussian_process.ExactGPModel(x, y, likelihood)
        # Epochs should be 100 as this model takes much longer and is very accurate
        gaussian_process.train_gp_model(model, x, y, likelihood, epochs=epochs)
        mean_values, std_values = gaussian_process.get_dist(model, x_test, likelihood)
        mean_values = mean_values.numpy()
        std_values = std_values.numpy()
    training_time = time.perf_counter() - start
    eval_data.plot_model(x, y, x_test, y_perfect, mean_values, std_values, title=title, filename=filename, reverse_plot=reverse_plot, figsize=figsize)
    mse = eval_data.mean_squared_error(y_perfect, mean_values)
    coverage = eval_data.coverage_95(y_test, mean_values, std_values)
    general_info = title + f"Training time: {training_time} MSE: {mse} Coverage: {coverage}\n"
    with open("write_up.txt", "a") as writefile:
        writefile.write(general_info)