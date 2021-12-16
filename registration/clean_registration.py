import os
import os.path
from os import path
import time
import sys
import numpy as np
import torch
import pickle
from tqdm import tqdm
import scipy
import scipy.interpolate as ip
from scipy.signal import convolve
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gridspec
from fast_histogram import histogram2d

import yass
from yass import set_config, read_config
from yass.reader import READER
from yass import preprocess, detect, cluster, postprocess, deconvolve, residual, noise, soft_assignment
from yass.preprocess.util import filter_standardize_batch, get_std, _butterworth, merge_filtered_files
from yass.template import run_template_computation


# Calculate Histograms
def electrode2space(config, sigma=28):
    x = int(config.geom[:, 1].max())
    M = np.exp(-(config.geom[:, 1:] - np.arange(x))**2/(2*sigma**2))
    return np.divide(M.T, M.sum(0, keepdims=True).T + 1e-6).T


def calc_histogram(ts, M, gauss_window, nbins=10, a=8, b=15,  apply_bg_removal=False, quant=0.9):
    ts = torch.from_numpy(ts).cuda().float()
    mp2d = torch.nn.MaxPool2d(kernel_size=[121, 1], stride=[20, 1]).cuda()
    ptp_sliding = mp2d(ts[None])[0] + mp2d(-ts[None])[0]

    if apply_bg_removal:
        for i in range(3):
            quantile_s = torch.kthvalue(ptp_sliding, int(
                quant * ptp_sliding.shape[1]), dim=1, keepdims=True)[0]  # size num_timepoints
            ptp_sliding = torch.nn.functional.relu(ptp_sliding - quantile_s)
            quantile_t = torch.kthvalue(ptp_sliding, int(
                quant * ptp_sliding.shape[0]), dim=0, keepdims=True)[0]  # size num_channels
            ptp_sliding = torch.nn.functional.relu(ptp_sliding - quantile_t)

    ptp_sliding = np.matmul(ptp_sliding.cpu().numpy(), M)*gauss_window
    bins = np.tile(np.arange(M.shape[1]), ptp_sliding.shape[0])
    hist_arrays = histogram2d(ptp_sliding.ravel(), bins, bins=(
        nbins, M.shape[1]), range=[[a, b], [0, M.shape[1]]])
    hist_arrays /= (hist_arrays.sum(0, keepdims=True)+1e-6)
    del ts
    torch.cuda.empty_cache()
    return ptp_sliding, hist_arrays


def calc_displacement(displacement, n_iter=100):
    n_batch = displacement.shape[0]
    p = np.zeros(displacement.shape[0])
    pprev = p
    for i in range(n_iter):
        mat_norm = displacement + p.repeat(repeats=n_batch).reshape(
            (n_batch, n_batch)) - p.repeat(repeats=n_batch).reshape((n_batch, n_batch)).T
        p += 2*((displacement-np.diag(displacement)).sum(1) -
                (n_batch-1)*p)/np.linalg.norm(mat_norm)
    return p


def calc_displacement_matrix(hist, nbins, disp=100, step_size=1, batchsize=200):
    num_hist = hist.shape[0]
    possible_displacement = np.arange(-disp, disp + step_size, step_size)
    hist = torch.from_numpy(hist).cuda().float()
    c2d = torch.nn.Conv2d(in_channels=1, out_channels=num_hist, kernel_size=[
                          nbins, hist.shape[-1]], stride=1, padding=[0, possible_displacement.size//2], bias=False).cuda()
    c2d.weight[:, 0] = hist
    displacement = np.zeros([hist.shape[0], hist.shape[0]])
    for i in tqdm(range(hist.shape[0]//batchsize)):
        displacement[i*batchsize:(i+1)*batchsize] = possible_displacement[c2d(
            hist[i*batchsize:(i+1)*batchsize, None])[:, :, 0, :].argmax(2).cpu()]
    return calc_displacement(displacement), displacement


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


# Displacement estimate

def smooth_displacement(estimated_displacement, num_hist=1000):
    arr_conv = np.arange(-num_hist, num_hist + 1)
    window = np.exp(-arr_conv**2/(2*64))/(np.sqrt(2*np.pi*64))
    return convolve(estimated_displacement, window, mode='same')


def create_displacement_map(config, displacement_estimate, windows):
    x = int(config.geom[:, 1].max())
    displacement_map = np.zeros((x, displacement_estimate.shape[1]))
    for i in tqdm(range(x)):
        sum_window = 0
        for j in range(windows.shape[0]):
            displacement_map[i] += windows[j, i]*displacement_estimate[j, :]
            sum_window += windows[j, i]
        displacement_map[i] /= sum_window
    return displacement_map


def estimate_displacement(config, reader, n_batch, nbins, nwindows, a_val, b_val, output_dir):

    M = electrode2space(config)

    hist = np.zeros([nwindows, n_batch, nbins, M.shape[1]])
    windows, _ = get_means_windows(config, nwindows)
    for j in range(nwindows):
        W = windows[j]
        for i in tqdm(range(n_batch)):
            ts = reader.read_data_batch(i, add_buffer=True)
            hist[j, i] = calc_histogram(ts, M, W, nbins=nbins, a=a_val, b=b_val)[1]
    np.save("{}/hist_w-{}_a-{}_b-{}_bn-{}.npy".format(output_dir, nwindows,
                                                         a_val, b_val, nbins), hist)
    displacement_estimate = np.zeros((nwindows, n_batch))
    for j in range(nwindows):
        displacement_estimate[j] = smooth_displacement(
            calc_displacement_matrix(hist[j], nbins)[0])

    np.savetxt("{}/w-{}_a-{}_b-{}_bn-{}.txt".format(output_dir, nwindows,
                                                         a_val, b_val, nbins), displacement_estimate)


def main():
    nbins = [10, 20]
#     a_vals = [8, 15]
#     b_vals = [10, 20]
    a_vals = [1, 2]
    b_vals = [50, 50]
    # Number of windows for non-rigid registration -> set to 1 for rigid registration
    nwindows = [1, 5, 10]
    # READ DATA
    fname_configs = ['/media/peter/2TB/hyundong/NP1/drift_np1.yaml',
                     '/media/peter/2TB/hyundong/cortexlab/drift.yaml', '/media/peter/2TB/hyundong/CSHL047/drift.yaml']
    recording_paths = ['/media/peter/2TB/hyundong/NP1/standardized.bin',
                       '/media/peter/2TB/hyundong/cortexlab/standardized.bin', '/media/peter/2TB/hyundong/CSHL047/standardized.bin']


    for fname_config, recording_path in zip(fname_configs, recording_paths):
        expt_dir = fname_config.split('/')[-2]
        #os.mkdir(expt_dir)
        print("working on {}".format(fname_config.split('/')[-2]))

        config = set_config(fname_config, 'alltmp')
        reader = READER(recording_path, 'float16', config, 1)
        n_chan = config.recordings.n_channels
        n_batch = reader.n_batches
        print(reader.n_batches)
        print(n_chan)
        print("config set up")

        for nwindow in nwindows:
            for a_val, b_val in zip(a_vals, b_vals):
                for nbin in nbins:
                    print("working on nwindows: {}, nbins: {}, a: {}, b: {}".format(
                        nwindow, nbin, a_val, b_val))
                    if path.exists("{}/w-{}_a-{}_b-{}_bn-{}.txt".format(expt_dir, nwindow, a_val, b_val, nbin)):
                        print("already done, skipping...")
                        continue
                    else:
                        start = time.time()
                        estimate_displacement(config, reader, n_batch, nbin,
                                              nwindow, a_val, b_val, expt_dir)
                        end = time.time()
                        print("{} minutes for nwindows: {}, nbins: {}, a: {}, b: {}".format(
                            (end-start)/60, nwindow, nbin, a_val, b_val))


if __name__ == "__main__":
    main()
