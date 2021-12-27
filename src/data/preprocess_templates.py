import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim_ls
from tqdm import tqdm

from optimization_metrics import minimize_ls


TOTAL_NUM_CHANNELS = 384  # For Neuropixels 2.0 probe


def get_max_chan_temps(templates):
    return templates.ptp(1).argmax(1)


def get_argmin_ptp(templates, max_chan_temp):
    argmin_ptp = np.zeros(templates.shape[0])
    for i in range(templates.shape[0]):
        argmin_ptp[i] = templates[i, :, max_chan_temp[i]].argmin()


def plot_templates(templates, max_chan_temp, n_channels=20):
    """
    Use to pick out and remove bad templates, i.e. those that don't look like waveforms or have collisions.
    """
    for i in range(templates.shape[0]):
        print(i)
        plt.figure(figsize=(n_channels, 2.5))
        plt.plot(templates[i, :80, max_chan_temp[i] - n_channels // 2:max_chan_temp[i] + n_channels // 2].T.flatten())
        for j in range(19):
            plt.axvline(80 + 80 * j, color='black')
        plt.show()


def take_channel_range(templates, n_channels_loc=20):
    """
    Take specified number of channels around main channel, i.e. channel containing the maximum PTP amplitude.
    """
    templates_chans = np.zeros((templates.shape[0], templates.shape[1], n_channels_loc))
    templates_chans_ptp = np.zeros((templates.shape[0], n_channels_loc))

    for i in range(templates.shape[0]):
        mc = templates[i].ptp(0).argmax()
        if mc <= n_channels_loc // 2:
            channels_wfs = np.arange(0, n_channels_loc)
        elif mc > TOTAL_NUM_CHANNELS - n_channels_loc:
            channels_wfs = np.arange(TOTAL_NUM_CHANNELS - n_channels_loc, TOTAL_NUM_CHANNELS)
        else:
            up_or_down = templates[i].ptp(0)[mc + 2] > templates[i].ptp(0)[mc - 2]
            if up_or_down and mc % 2 == 0:
                channels_wfs = np.arange(mc - n_channels_loc // 2 + 2, mc + n_channels_loc // 2 + 2)
            elif up_or_down:
                channels_wfs = np.arange(mc - n_channels_loc // 2 + 1, mc + n_channels_loc // 2 + 1)
            elif mc % 2 == 1:
                channels_wfs = np.arange(mc - n_channels_loc // 2 - 1, mc + n_channels_loc // 2 - 1)
            else:
                channels_wfs = np.arange(mc - n_channels_loc // 2, mc + n_channels_loc // 2)
        templates_chans[i] = templates[i, :, channels_wfs].T
        templates_chans_ptp[i] = templates[i, :, channels_wfs].T.ptp(0)

    return templates_chans, templates_chans_ptp


def localize_wfs(waveforms_ptp, geom_array):
    """
    Estimate location of neuron producing waveform using optimization framework in Boussard et al. 2021.
    """
    n_temp = waveforms_ptp.shape[0]
    output = np.zeros((n_temp, 4))
    channels_pos = geom_array[:waveforms_ptp.shape[1]]
    for i in tqdm(range(n_temp)):
        y_init = 22
        z_com = (waveforms_ptp[i] * channels_pos[:, 1]).sum() / waveforms_ptp[i].sum()
        x_com = (waveforms_ptp[i] * channels_pos[:, 0]).sum() / waveforms_ptp[i].sum()
        alpha_init = waveforms_ptp[i].max() * (
                (((channels_pos - [x_com, z_com]) ** 2).sum(1).min() + y_init ** 2) ** 0.5)
        output[i] = optim_ls.least_squares(minimize_ls, x0=[x_com, z_com, y_init, alpha_init], bounds=(
            [-150, -200, 0, 0], [182, 200, np.max([y_init + 10, 150]), np.max([alpha_init + 10, 10000])]),
                                           args=(waveforms_ptp[i], channels_pos), tr_solver='exact')['x']
    return output