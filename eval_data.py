import numpy as np
import matplotlib.pyplot as plt

def plot_data(x, y, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    plt.scatter(x, y, s=5)
    plt.show()

def plot_model(x, y, x_test, y_perfect, mean_values, std_values, title=None, filename=None, reverse_plot=False, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=0.8, label="Original data", color="#00dca6ff")
    plt.fill_between(x_test.data.numpy().T[0],mean_values-2.0*std_values,mean_values+2.0*std_values,alpha=0.2,color='#003c3cff',
                     label='95% confidence interval')
    if reverse_plot == False:
        plt.plot(x_test.data.numpy(),y_perfect.data.numpy(),'.',color="#FF00ACFF", lw=2,label='Original function')
        plt.plot(x_test.data.numpy(),mean_values,color='#003c3cff',lw=2,label='Predicted Mean Model')
    else:
        plt.plot(x_test.data.numpy(),mean_values,color='#003c3cff',lw=2,label='Predicted Mean Model')
        plt.plot(x_test.data.numpy(),y_perfect.data.numpy(),'.',color="#FF00ACFF", lw=2,label='Original function')
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def mean_squared_error(y_test, mean_values):
    return np.sqrt(np.sum((np.array(y_test) - mean_values.reshape(-1, 1))**2))

def coverage_95(y_test, mean_values, std_values):
    upper = mean_values.reshape(-1, 1)+2*std_values.reshape(-1, 1)
    lower = mean_values.reshape(-1, 1)-2*std_values.reshape(-1, 1)
    return np.mean((lower<=np.array(y_test)) & ((np.array(y_test)<=upper)))