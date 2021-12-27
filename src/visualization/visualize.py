import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_og_vs_reconstructed(og_templates, reconstructed_templates, n_channels=20, n_samples=20, out_fname=None):
    n_channels = 20
    pdf = mpl.backends.backend_pdf.PdfPages(out_fname)

    for i in range(n_samples):
        fig = plt.figure(figsize=(n_channels, 2.5))
        plt.plot(reconstructed_templates[i, :80, :].T.flatten(), color='blue')
        for j in range(19):
            plt.axvline(80 + 80 * j, color='black')
        plt.plot(og_templates[i, :80, :].T.flatten(), color='red')
        for j in range(19):
            plt.axvline(80 + 80 * j, color='black')
        plt.title("reconstructed: {}".format(i))
        pdf.savefig(fig)

    pdf.close()
