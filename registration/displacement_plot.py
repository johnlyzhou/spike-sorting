
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
mpl.use('Agg')
import os
import matplotlib.cm as cm
import yass
from yass import set_config


def electrode2space(config, sigma=28):
    x = int(config.geom[:, 1].max())
    M = np.exp(-(config.geom[:, 1:] - np.arange(x))**2/(2*sigma**2))
    return np.divide(M.T, M.sum(0, keepdims=True).T + 1e-6).T


def get_means_windows(config, nwindows):
    variance = config.geom[:, 1].max()/(0.5*nwindows)
    x = int(config.geom[:, 1].max())
    if nwindows == 1:
        return np.ones((1, x)), []
    else:
        space = int(x//(nwindows+1))
        means = np.linspace(space, x-space, nwindows, dtype=np.int16)
        windows = np.zeros((nwindows, x))
        M = electrode2space(config)
        for i in range(nwindows):
            windows[i] = np.exp(-(np.matmul(config.geom[:, 1],
                                            M) - means[i])**2 / (2*variance**2))
        return windows, means


def filename_to_params(fname):
    params = os.path.splitext(fname)[0].split("_")
    values = {}

    for param in params:
        key, val = param.split("-")
        values[key] = val
    return values


def dict_to_string(d):
    formatted = ""
    for key, val in d.items():
        formatted += "{}={}, ".format(key, val)
    return formatted[:-2]


def plot(displacement_estimate, fname, means, col):
    # _, means = get_means_windows(config, nwindow, variance=var)
    x = np.arange(np.shape(displacement_estimate)
                  [1])  # just as an example array
    for i in range(np.shape(displacement_estimate)[0]):
        if i == 0:
            plt.plot(x, displacement_estimate[i] - displacement_estimate[i,
                                                                         0] + means[i]/10, label=dict_to_string(filename_to_params(fname)), color=col)
        else:
            plt.plot(x, displacement_estimate[i] - displacement_estimate[i,
                                                                         0] + means[i]/10, color=col)

    # pp.plot(ar, displacement_estimate)


def main():
    fname_config = '/media/peter/2TB/hyundong/cortexlab/drift.yaml'
    config = set_config(fname_config, 'alltmp')
    filenames = os.listdir("test/")
    fontP = FontProperties()
    fontP.set_size('xx-small')
    vir = cm.get_cmap('viridis', 12)
    i=0
    for filename in filenames:
        displacement_estimate = np.loadtxt("test/" + filename)
        params = filename_to_params(filename)
        _, means = get_means_windows(config, int(params['w']))

        plot(displacement_estimate, filename, means, vir(i/12))
        i+=1
    lgd = plt.legend(bbox_to_anchor=(1.05, 1))
    plt.xlabel('time (s)')
    plt.ylabel('displacement (um)')
    #plt.yticks([])
    title = plt.suptitle('Cortex Lab - 10 Windows')
    plt.savefig("cortexlab_10windows.jpg", dpi=1000, bbox_extra_artists=(lgd, title), bbox_inches='tight')


if __name__ == "__main__":
    main()
