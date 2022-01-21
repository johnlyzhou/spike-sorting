from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.data.preprocess_templates import get_max_chan_temps


def plot_templates(templates, base_template=None, n_channels=20, n_templates=None):
    """
    Templates should be of shape (num_samples, num_timesteps, num_channels).
    """
    num_samples, num_timesteps, template_channels = templates.shape

    if n_templates is None or n_templates > num_samples:
        n_templates = num_samples

    max_chan_temp = get_max_chan_temps(templates)

    for i in range(n_templates):
        print(f"Template {i}")
        plt.figure(figsize=(n_channels, 2.5))
        if template_channels == n_channels:
            plt.plot(templates[i, :80, :].T.flatten(), color='red')
        else:
            plt.plot(templates[i, :80, max_chan_temp[i] - n_channels // 2:max_chan_temp[i]
                               + n_channels // 2].T.flatten(), color='blue')
        if base_template is not None:
            plt.plot(base_template[:80, :].T.flatten(), color='green')
        for j in range(n_channels - 1):
            plt.axvline(80 + 80 * j, color='black')
        plt.show()


def plot_template_reconstructions(og_templates, recon_templates, n_channels=20, out_fname=None, n_templates=None):
    """
    Plot and save a specified number of sample reconstructed waveforms overlaid against the original waveforms.
    """
    if n_templates is None or n_templates > recon_templates.shape[0]:
        n_templates = recon_templates.shape[0]

    if out_fname:
        pdf = mpl.backends.backend_pdf.PdfPages(out_fname)

    for i in range(n_templates):
        fig = plt.figure(figsize=(n_channels, 2.5))
        plt.plot(recon_templates[i, :80, :].T.flatten(), color='red')
        plt.plot(og_templates[i, :80, :].T.flatten(), color='blue')
        for j in range(n_channels - 1):
            plt.axvline(80 + 80 * j, color='black')
        plt.title("Reconstructed: {}".format(i))
        plt.show()

        if out_fname:
            pdf.savefig(fig)

    if out_fname:
        pdf.close()


def plot_latent_features_vs_position(latent_reps, positions, vary_feature="x", data_dir=None):
    """
    Plot and save a scatter plot of values for each latent space dimension as a function of a specific positional
    feature value (x, y, z, or alpha).
    """
    [xs, ys, zs, alphas] = positions
    if vary_feature == "x":
        x = xs
    elif vary_feature == "y":
        x = ys
    elif vary_feature == "z":
        x = zs
    elif vary_feature == "alpha":
        x = alphas
    else:
        raise ValueError("Invalid feature specified! Choose one of \'x\', \'y\', \'z\', or \'alpha\'.")

    num_dims = latent_reps.shape[1]
    plt.figure()
    for dim in range(num_dims):
        y = latent_reps[:, dim]
        plt.scatter(x, y, s=5, cmap="viridis", label=dim)
    plt.title("{} Position vs. Latent Dimensions".format(vary_feature))
    plt.xlabel(vary_feature)
    plt.ylabel("Latent Dimension Values")
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.show()

    if data_dir:
        Path("{}/position_invariance".format(data_dir)).mkdir(parents=True, exist_ok=True)
        plt.savefig("{}/position_invariance/{}_vs_{}_latents.png".format(data_dir, vary_feature, num_dims))