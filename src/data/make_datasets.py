from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import gamma
from tqdm import tqdm


def time_center_templates(templates):
    centered_templates = np.zeros(templates.shape)
    for i in range(templates.shape[0]):
        mc = templates[i].ptp(0).argmax()
        offset = 42 - templates[i, :, mc].argmin()
        centered_templates[i] = np.roll(templates[i], offset, axis=0)
    return centered_templates


def predict_ptp(template_position, channel_position):
    """
    Predict PTP amplitude on each recording channel using the point cloud model from Boussard et al. 2021.
    """
    x, y, z, alpha = template_position
    x_c, z_c = channel_position
    predicted_ptp = alpha / (((np.array([x - x_c, z - z_c])) ** 2).sum() + z ** 2) ** 0.5
    return predicted_ptp


def resample_template(template):
    """
    Random temporal shift in sampling intervals for a given template.
    """
    num_timesteps, num_channels = template.shape
    time_range = range(num_timesteps)
    template_spline = CubicSpline(time_range, template, axis=0)
    shift = np.random.rand()
    resampled_template = np.array([template_spline(t + shift) for t in time_range])
    return resampled_template


def dataset_saver(func):
    """
    Time-center and save datasets.
    """
    def wrapper_dataset_saver(*args, experiment_data_dir, experiment_name, **kwargs):
        templates, predicted_ptps, positions, idx_units = func(*args, **kwargs)

        templates = time_center_templates(templates)
        templates = templates.swapaxes(1, 2)
        positions = np.transpose(positions)

        experiment_dir = Path(f"{experiment_data_dir}/{experiment_name}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print("Saving templates to folder {}, array of shape: {}".format(experiment_dir, templates.shape))
        np.save("{}/templates.npy".format(experiment_dir), templates)
        print("Saving predicted PTPs to folder {}, array of shape: {}".format(experiment_dir, predicted_ptps.shape))
        np.save("{}/predicted_ptps.npy".format(experiment_dir), predicted_ptps)
        print("Saving positions to folder {}, array of shape: {}".format(experiment_dir, positions.shape))
        np.save("{}/positions.npy".format(experiment_dir), positions)
        print("Saving unit indices to folder {}, array of shape: {}".format(experiment_dir, idx_units.shape))
        np.save("{}/unit_idxs.npy".format(experiment_dir), idx_units)

    return wrapper_dataset_saver


@dataset_saver
def featurization_dataset(templates, template_positions, channel_positions, a, loc, scale,
                          n_samples=10000, noise_path=None):
    """
    Produces a dataset for feature learning. For each sample, randomly sample a template and position parameters
    x, y, z, and alpha, then use the point cloud model from Boussard et al. 2021 to project that template to a new
    location and estimate its new PTP amplitude on each recording channel.
    """

    num_templates, num_timesteps, num_channels = templates.shape

    if noise_path:
        print("Adding noise to dataset")
        noise = np.load(noise_path)
        num_timesteps_noise, num_channels_noise = noise.shape
        if num_timesteps_noise < num_timesteps:
            raise ValueError(f"Recording needs to be at least {num_timesteps} timesteps for these templates.")
        if num_channels_noise < num_channels:
            raise ValueError(f"Recordings needs at least {num_channels} channels for these templates.")
    else:
        print("No noise to add to dataset")

    chan_pos_mean = channel_positions[:, 1].mean()

    x = np.random.uniform(-150, 182, n_samples)
    y = np.random.uniform(0, 150, n_samples)
    z = np.random.normal(chan_pos_mean, 25, n_samples)
    alpha = gamma.rvs(a, loc, scale, size=n_samples)

    outside_idxs = (y ** 2 + (x - 16) ** 2 > 150 ** 2)
    num_outside = outside_idxs.sum()

    while num_outside > 0:
        x[outside_idxs] = np.random.uniform(-150, 182, num_outside)
        y[outside_idxs] = np.random.uniform(0, 150, num_outside)
        outside_idxs = (y ** 2 + (x - 16) ** 2 > 150 ** 2)
        num_outside = outside_idxs.sum()

    relocated_positions = np.vstack((x, y, z, alpha))

    idx_units = np.zeros(n_samples)
    new_templates = np.zeros((n_samples, num_timesteps, num_channels))

    for i in tqdm(range(n_samples)):
        idx_temp = np.random.choice(np.arange(template_positions.shape[0]))
        idx_units[i] = idx_temp

        if noise_path:
            noise_timesteps = np.random.randint(num_timesteps_noise - num_timesteps)
            noise_chans = np.random.randint(num_channels_noise - num_channels)
            sample_noise = noise[noise_timesteps:noise_timesteps + num_timesteps,
                                 noise_chans:noise_chans + num_channels]

        resampled_template = resample_template(templates[idx_temp, :, :])

        for j in range(channel_positions.shape[0]):
            predicted_ptp = predict_ptp(template_positions[idx_temp, :], channel_positions[j])
            new_predicted_ptp = predict_ptp(relocated_positions[:, i], channel_positions[j])
            new_templates[i, :, j] = resampled_template[:, j] * new_predicted_ptp / predicted_ptp

            if noise_path:
                new_templates[i, :, j] += sample_noise[:, j]

    return new_templates, new_templates.ptp(1), relocated_positions, idx_units


@dataset_saver
def clustering_dataset(templates, template_positions, channel_positions, a=3, loc=100, scale=500, n_clusters=20,
                       num_samples_per_cluster=100, x_var=20, y_var=20, z_var=10, noise_path=None):
    """
    Produces a dataset for feature evaluation on evaluation performance. For each cluster, randomly sample a
    template and set of mean position parameters x, y, z, and alpha. Then add some Gaussian noise to each positional
    parameter (variance arbitrarily selected as of now based on evaluation performance) and generate resulting
    waveforms.

    Note: drift sampling means there is the potential for templates to be projected outside of normal ranges.
    """
    num_templates, num_timesteps, num_channels = templates.shape

    if num_templates < n_clusters:
        raise ValueError("Too few templates available to create desired number of clusters.")

    if noise_path:
        print("Adding noise to dataset")
        noise = np.load(noise_path)
        num_timesteps_noise, num_channels_noise = noise.shape
        if num_timesteps_noise < num_timesteps:
            raise ValueError(f"Recording needs to be at least {num_timesteps} timesteps for these templates.")
        if num_channels_noise < num_channels:
            raise ValueError(f"Recordings needs at least {num_channels} channels for these templates.")
    else:
        print("No noise to add to dataset")

    chan_pos_mean = channel_positions[:, 1].mean()

    # For each cluster, randomly select a mean position
    mean_x = np.random.uniform(-150, 182, n_clusters)
    mean_y = np.random.uniform(0, 150, n_clusters)
    mean_z = np.random.normal(chan_pos_mean, 25, n_clusters)
    mean_alpha = gamma.rvs(a, loc, scale, size=n_clusters)

    outside_idxs = (mean_y ** 2 + (mean_x - 16) ** 2 > 150 ** 2)
    num_outside = outside_idxs.sum()
    while num_outside > 0:
        mean_x[outside_idxs] = np.random.uniform(-150, 182, num_outside)
        mean_y[outside_idxs] = np.random.uniform(0, 150, num_outside)
        outside_idxs = (mean_y ** 2 + (mean_x - 16) ** 2 > 150 ** 2)
        num_outside = outside_idxs.sum()

    # Now repeat each mean by the number of samples desired per cluster
    mean_x = np.repeat(mean_x, num_samples_per_cluster)
    mean_y = np.repeat(mean_y, num_samples_per_cluster)
    mean_z = np.repeat(mean_z, num_samples_per_cluster)
    mean_alpha = np.repeat(mean_alpha, num_samples_per_cluster)

    # Apply "drift" for every sample
    alpha = gamma.rvs(a) * mean_alpha
    x = np.random.normal(mean_x, x_var)
    y = np.abs(np.random.normal(mean_y, y_var))
    z = np.random.normal(mean_z, z_var)

    relocated_positions = np.vstack((x, y, z, alpha))

    new_templates = np.zeros((n_clusters * num_samples_per_cluster, num_timesteps, num_channels))
    idx_units = np.zeros(num_samples_per_cluster * n_clusters)

    # Randomly sample templates
    idx_temps = np.random.choice(np.arange(num_templates), size=n_clusters, replace=False)

    i = 0
    for k in tqdm(range(n_clusters)):
        idx_temp = idx_temps[k]
        for _ in range(num_samples_per_cluster):
            idx_units[i] = idx_temp

            resampled_template = resample_template(templates[idx_temp, :, :])

            if noise_path:
                noise_timesteps = np.random.randint(num_timesteps_noise - num_timesteps)
                noise_chans = np.random.randint(num_channels_noise - num_channels)
                sample_noise = noise[noise_timesteps:noise_timesteps + num_timesteps,
                                     noise_chans:noise_chans + num_channels]

            for j in range(num_channels):
                predicted_ptp = predict_ptp(template_positions[idx_temp, :], channel_positions[j])
                new_predicted_ptp = predict_ptp(relocated_positions[:, i], channel_positions[j])
                new_templates[i, :, j] = resampled_template[:, j] * new_predicted_ptp / predicted_ptp

                if noise_path:
                    new_templates[i, :, j] += sample_noise[:, j]

            i += 1

    return new_templates, new_templates.ptp(1), relocated_positions, idx_units


@dataset_saver
def positional_invariance_dataset(templates, template_positions, channel_positions, a, loc, scale, vary_feature="x",
                                  n_samples=100):
    """
    Produces a dataset to find positional invariance among learned features. Repeatedly sample the specified
    position feature that we want to vary (exactly the same as for the featurization dataset), but sample all
    other parameters (4 of 5: x, y, z, alpha, template) once only and hold them constant for all samples.
    """

    num_templates, num_timesteps, num_channels = templates.shape

    if vary_feature != "alpha":
        alpha_const = gamma.rvs(a, loc, scale)
        alpha = np.full(n_samples, alpha_const)
    else:
        alpha = gamma.rvs(a, loc, scale, size=n_samples)

    if vary_feature != "y":
        y_const = np.random.uniform(0, 150)
        y = np.full(n_samples, y_const)
    else:
        y = np.random.uniform(0, 150, n_samples)

    if vary_feature != "x":
        x_const = np.random.uniform(-150, 182)
        x = np.full(n_samples, x_const)
    else:
        x = np.random.uniform(-150, 182, n_samples)

    chan_pos_mean = channel_positions[:, 1].mean()
    if vary_feature != "z":
        z_const = np.random.normal(chan_pos_mean, 25)
        z = np.full(n_samples, z_const)
    else:
        z = np.random.normal(chan_pos_mean, 25, n_samples)

    outside_idxs = (y ** 2 + (x - 16) ** 2 > 150 ** 2)
    num_outside = outside_idxs.sum()
    while num_outside > 0:
        if vary_feature != "y":
            y_const = np.random.uniform(0, 150)
            y = np.full(n_samples, y_const)
        else:
            y[outside_idxs] = np.random.uniform(0, 150, num_outside)
        if vary_feature != "x":
            x_const = np.random.uniform(-150, 182)
            x = np.full(n_samples, x_const)
        else:
            x[outside_idxs] = np.random.uniform(-150, 182, num_outside)
        outside_idxs = (y ** 2 + (x - 16) ** 2 > 150 ** 2)
        num_outside = outside_idxs.sum()

    relocated_positions = np.vstack((x, y, z, alpha))
    idx_units = np.zeros(n_samples)
    new_templates = np.zeros((n_samples, num_timesteps, num_channels))

    idx_temp = np.random.choice(np.arange(num_templates))
    for i in tqdm(range(n_samples)):
        idx_units[i] = idx_temp
        for j in range(num_channels):
            predicted_ptp = predict_ptp(template_positions[idx_temp, :], channel_positions[j])
            new_predicted_ptp = predict_ptp(relocated_positions[:, i], channel_positions[j])
            new_templates[i, :, j] = templates[idx_temp, :, j] * new_predicted_ptp / predicted_ptp
    return new_templates, new_templates.ptp(1), relocated_positions, idx_units
