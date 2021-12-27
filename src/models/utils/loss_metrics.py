import numpy as np
import torch


def gaussian_nll(x, x_hat, scale=1):
    predicted_dist = torch.distributions.Normal(x_hat, scale)
    batch_size = x.shape[0]
    return -predicted_dist.log_prob(x).view(batch_size, -1).sum(dim=1)


def kl_divergence(mu, log_var):
    return torch.mean(0.5 * torch.sum(log_var.exp() - log_var + mu ** 2 - 1, dim=1), dim=0)


def decomposed_kl_divergence(z, mu, logvar):
    """
    (Borrowed from https://github.com/themattinthehatt/behavenet)

    Decompose KL-divergence term in VAE loss.
    Decomposes the KL divergence loss term of the variational autoencoder into three terms:
    1. index code mutual information
    2. total correlation
    3. dimension-wise KL
    None of these terms can be computed exactly when using stochastic gradient descent. This
    function instead computes approximations as detailed in https://arxiv.org/pdf/1802.04942.pdf.
    Parameters
    ----------
    z : :obj:`torch.Tensor`
        sample of shape (n_frames, n_dims)
    mu : :obj:`torch.Tensor`
        mean parameter of shape (n_frames, n_dims)
    logvar : :obj:`torch.Tensor`
        log variance parameter of shape (n_frames, n_dims)
    Returns
    -------
    :obj:`tuple`
        - index code mutual information (:obj:`torch.Tensor`)
        - total correlation (:obj:`torch.Tensor`)
        - dimension-wise KL (:obj:`torch.Tensor`)
    """

    # Compute log(q(z(x_j)|x_i)) for every sample/dimension in the batch, which is a tensor of
    # shape (n_frames, n_dims). In the following comments, (n_frames, n_frames, n_dims) are indexed
    # by [j, i, l].
    #
    # Note that the insertion of `None` expands dims to use torch's broadcasting feature
    # z[:, None]: (n_frames, 1, n_dims)
    # mu[None, :]: (1, n_frames, n_dims)
    # logvar[None, :]: (1, n_frames, n_dims)
    log_qz_prob = _gaussian_log_density_unsummed(z[:, None], mu[None, :], logvar[None, :])

    # Compute log(q(z(x_j))) as
    # log(sum_i(q(z(x_j)|x_i))) + constant
    # = log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant
    # = log(sum_i(exp(sum_l log q(z(x_j)_l|x_i))) + constant (assumes q is factorized)
    log_qz = torch.logsumexp(
        torch.sum(log_qz_prob, dim=2, keepdim=False),  # sum over gaussian dims
        dim=1,  # logsumexp over batch
        keepdim=False)

    # Compute log prod_l q(z(x_j)_l | x_j)
    # = sum_l log q(z(x_j)_l | x_j)
    log_qz_ = torch.diag(torch.sum(log_qz_prob, dim=2, keepdim=False))  # sum over gaussian dims

    # Compute log prod_l p(z(x_j)_l)
    # = sum_l(log(sum_i(q(z(x_j)_l|x_i))) + constant
    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1, keepdim=False),  # logsumexp over batch
        dim=1,  # sum over gaussian dims
        keepdim=False)

    # Compute sum_l log p(z(x_j)_l)
    log_pz_prob = _gaussian_log_density_unsummed_std_normal(z)
    log_pz_product = torch.sum(log_pz_prob, dim=1, keepdim=False)  # sum over gaussian dims

    idx_code_mi = torch.mean(log_qz_ - log_qz)
    total_corr = torch.mean(log_qz - log_qz_product)
    dim_wise_kl = torch.mean(log_qz_product - log_pz_product)

    return idx_code_mi, total_corr, dim_wise_kl


def _gaussian_log_density_unsummed(z, mu, logvar):
    """First step of Gaussian log-density computation, without summing over dimensions.
    Assumes a diagonal noise covariance matrix.
    """
    diff_sq = (z - mu) ** 2
    inv_var = torch.exp(-logvar)
    return - 0.5 * (inv_var * diff_sq + logvar + np.log(2 * np.pi))


def _gaussian_log_density_unsummed_std_normal(z):
    """First step of Gaussian log-density computation, without summing over dimensions.
    Assumes a diagonal noise covariance matrix.
    """
    diff_sq = z ** 2
    return - 0.5 * (diff_sq + np.log(2 * np.pi))
