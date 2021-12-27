import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from src.models.spike_vaes import SpikeSortingPSVAE


def clustering_analysis(system_cls, checkpoint, templates, clusters):
    model = system_cls.load_from_checkpoint(checkpoint)
    outputs = model(torch.tensor(templates).float())

    if system_cls == SpikeSortingPSVAE:
        supervised_latents = outputs[0].detach().numpy()
        unsupervised_latents = outputs[1].detach().numpy()
        reps = np.hstack((supervised_latents, unsupervised_latents))
    else:
        reps = outputs[0].detach().numpy()

    km = KMeans(n_clusters=20, random_state=4995).fit(reps)
    ari = adjusted_rand_score(km.labels_, clusters)

    z = TSNE(n_components=2, verbose=1, random_state=4995).fit_transform(reps)
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], cmap="tab20", c=clusters)
    plt.title(f"k-Means ARI = {ari:.3f}")
    plt.savefig(f"visualizations/{model.config['name']}/clustering.png")

    # Perform clustering with unsupervised latents only as well
    if system_cls == SpikeSortingPSVAE:
        reps = unsupervised_latents
        km = KMeans(n_clusters=20, random_state=4995).fit(reps)
        ari = adjusted_rand_score(km.labels_, clusters)

        z = TSNE(n_components=2, verbose=1, random_state=4995).fit_transform(reps)
        plt.figure()
        plt.scatter(z[:, 0], z[:, 1], cmap="tab20", c=clusters)
        plt.title(f"k-Means ARI = {ari:.3f} (Unsupervised Latents Only)")
        plt.savefig(f"visualizations/{model.config['name']}/unsupervised_latents_clustering.png")
