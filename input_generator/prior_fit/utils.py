import torch


def neg_log_likelihood(y, yhat):
    """
    Convert dG to probability and use KL divergence to get difference between
    predicted and actual
    """
    L = torch.sum(torch.exp(-y) * torch.log(torch.exp(-yhat)))
    return -L


def compute_aic(real_distibution, ref_distribution, free_parameters):
    """Method for computing the AIC"""
    aic = (
        2
        * neg_log_likelihood(
            ref_distribution,
            real_distibution,
        )
        + 2 * free_parameters
    )
    return aic
