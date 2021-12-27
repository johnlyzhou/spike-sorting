import numpy as np
import scipy.stats as stats
from tqdm import tqdm


def featurization_dataset(templates, positions_templates, channels_pos, a, loc, scale, size, n_samples=10000):
    """
    Produces a dataset for feature learning. For each sample, randomly sample a template and position parameters
    x, y, z, and alpha, then use the point cloud model from Boussard et al. 2021 to project that template to a new
    location and estimate its new PTP amplitude on each recording channel.
    """
    gamma = stats.gamma
    # y = gamma.rvs(a, loc, scale, size=size)
    alpha = gamma.rvs(a, loc, scale, size=n_samples)
    y = np.random.uniform(0, 150, n_samples)
    x_z = np.zeros((2, n_samples))
    x_z[0, :] = np.random.uniform(-150, 182, n_samples)
    chan_pos_mean = channels_pos[:, 1].mean()
    x_z[1, :] = np.random.normal(chan_pos_mean, 25, n_samples)

    idxbool = (y ** 2 + (x_z[0, :] - 16) ** 2 > 150 ** 2)
    num_outside = idxbool.sum()
    while num_outside > 0:
        y[idxbool] = np.random.uniform(0, 150, num_outside)
        x_z[0, idxbool] = np.random.uniform(-150, 182, num_outside)
        idxbool = (y ** 2 + (x_z[0, :] - 16) ** 2 > 150 ** 2)
        num_outside = idxbool.sum()

    relocated_positions = np.zeros((4, n_samples))
    relocated_positions[:2] = x_z
    relocated_positions[2] = y
    relocated_positions[3] = alpha
    idx_units = np.zeros(n_samples)
    new_templates = np.zeros((n_samples, templates.shape[1], templates.shape[2]))

    for i in tqdm(range(n_samples)):
        idx_temp = np.random.choice(np.arange(positions_templates.shape[0]))
        idx_units[i] = idx_temp
        # scaling_factor = np.random.uniform(0.8, 1.5)
        for j in range(channels_pos.shape[0]):
            predicted_ptp = positions_templates[idx_temp, 3] / ((([positions_templates[idx_temp, 0],
                                                                   positions_templates[idx_temp, 1]] -
                                                                  channels_pos[j]) ** 2).sum() +
                                                                positions_templates[idx_temp, 2] ** 2) ** 0.5
            new_predicted_ptp = alpha[i] / (((x_z[:, i] - channels_pos[j]) ** 2).sum() + y[i] ** 2) ** 0.5
            new_templates[i, :, j] = templates[idx_temp, :, j] * new_predicted_ptp / predicted_ptp
    return new_templates, new_templates.ptp(1), relocated_positions, idx_units


def positional_invariance_dataset(templates, positions_templates, channels_pos, a, loc, scale, perturb_param="x",
                                  n_samples=100):
    """
    Produces a dataset to find positional invariance among learned features. Repeatedly sample the specified
    position parameter that we want to perturb (exactly the same as for the featurization dataset), but samples all
    other parameters (4 of 5: x, y, z, alpha, template) only once and holds them constant for all samples.
    """
    gamma = stats.gamma
    if perturb_param != "alpha":
        alpha_const = gamma.rvs(a, loc, scale)
        alpha = np.full(n_samples, alpha_const)
    else:
        alpha = gamma.rvs(a, loc, scale, size=n_samples)

    if perturb_param != "y":
        y_const = np.random.uniform(0, 150)
        y = np.full(n_samples, y_const)
    else:
        y = np.random.uniform(0, 150, n_samples)

    x_z = np.zeros((2, n_samples))
    if perturb_param != "x":
        x_const = np.random.uniform(-150, 182)
        x_z[0, :] = np.full(n_samples, x_const)
    else:
        x_z[0, :] = np.random.uniform(-150, 182, n_samples)

    chan_pos_mean = channels_pos[:, 1].mean()
    if perturb_param != "z":
        z_const = np.random.normal(chan_pos_mean, 25)
        x_z[1, :] = np.full(n_samples, z_const)
    else:
        x_z[1, :] = np.random.normal(chan_pos_mean, 25, n_samples)

    idxbool = (y ** 2 + (x_z[0, :] - 16) ** 2 > 150 ** 2)
    num_outside = idxbool.sum()
    while num_outside > 0:
        if perturb_param != "y":
            y_const = np.random.uniform(0, 150)
            y = np.full(n_samples, y_const)
        else:
            y[idxbool] = np.random.uniform(0, 150, num_outside)
        if perturb_param != "x":
            x_const = np.random.uniform(-150, 182)
            x_z[0, :] = np.full(n_samples, x_const)
        else:
            x_z[0, idxbool] = np.random.uniform(-150, 182, num_outside)
        idxbool = (y ** 2 + (x_z[0, :] - 16) ** 2 > 150 ** 2)
        num_outside = idxbool.sum()

    relocated_positions = np.zeros((4, n_samples))
    relocated_positions[:2] = x_z
    relocated_positions[2] = y
    relocated_positions[3] = alpha
    idx_units = np.zeros(n_samples)
    new_templates = np.zeros((n_samples, templates.shape[1], templates.shape[2]))

    idx_temp = np.random.choice(np.arange(positions_templates.shape[0]))
    for i in tqdm(range(n_samples)):
        idx_units[i] = idx_temp
        for j in range(channels_pos.shape[0]):
            predicted_ptp = positions_templates[idx_temp, 3] / ((([positions_templates[idx_temp, 0],
                                                                   positions_templates[idx_temp, 1]] -
                                                                  channels_pos[j]) ** 2).sum() +
                                                                positions_templates[idx_temp, 2] ** 2) ** 0.5
            new_predicted_ptp = alpha[i] / (((x_z[:, i] - channels_pos[j]) ** 2).sum() + y[i] ** 2) ** 0.5
            new_templates[i, :, j] = templates[idx_temp, :, j] * new_predicted_ptp / predicted_ptp
    return new_templates, new_templates.ptp(1), relocated_positions, idx_units


def clustering_dataset(templates, positions_templates, channels_pos, a, loc, scale, size, n_clusters=20,
                       num_samples_per_cluster=100):
    """
    Produces a dataset for feature evaluation on evaluation performance. For each cluster, randomly sample a
    template and set of mean position parameters x, y, z, and alpha. Then add some Gaussian noise to each positional
    parameter (variance arbitrarily selected as of now based on evaluation performance) and generate resulting
    waveforms.

    Note: unlike the featurization and positional invariance datasets, we do not (yet) conduct an "outside"
    position check after adding drift.
    """
    gamma = stats.gamma

    # For each cluster, randomly select a template and mean position
    mean_alpha = gamma.rvs(a, loc, scale, size=n_clusters)
    mean_x_z = np.zeros((2, n_clusters))
    mean_x_z[0, :] = np.random.uniform(-150, 182, n_clusters)
    chan_pos_mean = channels_pos[:, 1].mean()
    mean_x_z[1, :] = np.random.normal(chan_pos_mean, 25, n_clusters)
    mean_y = np.random.uniform(0, 150, n_clusters)

    idxbool = (mean_y ** 2 + (mean_x_z[0, :] - 16) ** 2 > 150 ** 2)
    num_outside = idxbool.sum()
    while num_outside > 0:
        mean_y[idxbool] = np.random.uniform(0, 150, num_outside)
        mean_x_z[0, idxbool] = np.random.uniform(-150, 182, num_outside)
        idxbool = (mean_y ** 2 + (mean_x_z[0, :] - 16) ** 2 > 150 ** 2)
        num_outside = idxbool.sum()

    # Now repeat each mean by the number of samples desired per cluster
    mean_alpha = np.repeat(mean_alpha, num_samples_per_cluster)
    mean_x_z = np.repeat(mean_x_z, num_samples_per_cluster, axis=1)
    mean_y = np.repeat(mean_y, num_samples_per_cluster)

    # Apply "drift" for every sample
    alpha = np.random.normal(mean_alpha, 200)
    x_z = np.random.normal(mean_x_z, 20)
    y = np.random.normal(mean_y, 20)

    new_templates = np.zeros((n_clusters * num_samples_per_cluster, templates.shape[1], templates.shape[2]))
    relocated_positions = np.zeros((4, n_clusters * num_samples_per_cluster))
    relocated_positions[:2] = x_z
    relocated_positions[2] = y
    relocated_positions[3] = alpha
    idx_units = np.zeros(num_samples_per_cluster * n_clusters)
    idx_temps = np.random.choice(np.arange(positions_templates.shape[0]), size=n_clusters, replace=False)

    i = 0
    for k in range(n_clusters):
        idx_temp = idx_temps[k]
        for j in range(num_samples_per_cluster):
            idx_units[i] = idx_temp
            # scaling_factor = np.random.uniform(0.8, 1.5)
            for j in range(channels_pos.shape[0]):
                predicted_ptp = positions_templates[idx_temp, 3] / ((([positions_templates[idx_temp, 0],
                                                                       positions_templates[idx_temp, 1]] -
                                                                      channels_pos[j]) ** 2).sum() +
                                                                    positions_templates[idx_temp, 2] ** 2) ** 0.5
                new_predicted_ptp = alpha[i] / (((x_z[:, i] - channels_pos[j]) ** 2).sum() + y[i] ** 2) ** 0.5
                new_templates[i, :, j] = templates[idx_temp, :, j] * new_predicted_ptp / predicted_ptp
            i += 1
    return new_templates, new_templates.ptp(1), relocated_positions, idx_units


def time_center_templates(templates_chans):
    centered_templates = np.zeros(templates_chans.shape)
    for i in range(templates_chans.shape[0]):
        mc = templates_chans[i].ptp(0).argmax()
        offset = 42 - templates_chans[i, :, mc].argmin()
        centered_templates[i] = np.roll(templates_chans[i], offset, axis=0)
    return centered_templates


def normalize_inputs(templates):
    """
    Normalize across time and all samples. Waveforms assumed to be roughly Gaussian, even though they aren't.
    """
    mean = np.mean(templates, axis=(0, 1))
    var = np.var(templates, axis=(0, 1))
    return (templates - mean) / var
