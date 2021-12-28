from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_template_reconstructions(og_templates, reconstructed_templates, n_channels=20, n_samples=20, out_fname=None):
    """
    Plot and save a specified number of sample reconstructed waveforms overlaid against the original waveforms.
    """
    pdf = mpl.backends.backend_pdf.PdfPages(out_fname)

    for i in range(n_samples):
        fig = plt.figure(figsize=(n_channels, 2.5))
        plt.plot(reconstructed_templates[i, :80, :].T.flatten(), color='red')
        for j in range(n_channels - 1):
            plt.axvline(80 + 80 * j, color='black')
        plt.plot(og_templates[i, :80, :].T.flatten(), color='blue')
        for j in range(n_channels - 1):
            plt.axvline(80 + 80 * j, color='black')
        plt.title("Reconstructed: {}".format(i))
        pdf.savefig(fig)

    pdf.close()


def plot_latent_features_vs_position(latent_reps, positions, data_dir, vary_feature="x"):
    """
    Plot and save a scatter plot of values for each latent space dimension as a function of a specific positional
    feature value (x, y, z, or alpha).
    """
    num_dims = latent_reps.shape[1]
    plt.figure()
    [xs, zs, ys, alphas] = positions
    if vary_feature == "x":
        x = xs
    elif vary_feature == "z":
        x = zs
    elif vary_feature == "y":
        x = ys
    elif vary_feature == "alpha":
        x = alphas
    else:
        raise ValueError("Invalid feature specified! Choose one of \'x\', \'y\', \'z\', or \'alpha\'.")
    for dim in range(num_dims):
        y = latent_reps[:, dim]
        plt.scatter(x, y, s=5, cmap="viridis", label=dim)
    plt.title("{} Position vs. Latent Dimensions".format(vary_feature))
    plt.xlabel(vary_feature)
    plt.ylabel("Latent Dimension Values")
    plt.legend(bbox_to_anchor=(1.1, 1.05))

    Path("{}/position_invariance".format(data_dir)).mkdir(parents=True, exist_ok=True)
    plt.savefig("{}/position_invariance/{}_vs_{}_latents.png".format(data_dir, vary_feature, num_dims))
