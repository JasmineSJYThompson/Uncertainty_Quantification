import torch

def make_initial_data(get_target_function, start=-5, end=5, noise_coeff=1):
    torch.manual_seed(42)
    
    x = torch.linspace(start, end, 10000)
    y = get_target_function(x) + noise_coeff*torch.normal(0, 1, size=x.size())
    return x, y

def make_test_data(get_target_function, start=-5, end=5, noise_coeff=1):
    torch.manual_seed(66)
    x_test = torch.linspace(start, end, 10000)
    y_test = get_target_function(x_test) + noise_coeff*torch.normal(0, 1, size=x_test.size())
    y_perfect = get_target_function(x_test)
    
    x_test = torch.unsqueeze(x_test, dim=1)
    y_test = torch.unsqueeze(y_test, dim=1)
    y_perfect = torch.unsqueeze(y_perfect, dim=1)
    return x_test, y_test, y_perfect