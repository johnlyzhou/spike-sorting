import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

from src.data.optimization_metrics import minimize_ls


TOTAL_NUM_CHANNELS = 384  # For Neuropixels 2.0 probe


def get_max_chan_temps(templates):
    return templates.ptp(1).argmax(1)


def get_argmin_ptp(templates, max_chan_temp):
    argmin_ptp = np.zeros(templates.shape[0])
    for i in range(templates.shape[0]):
        argmin_ptp[i] = templates[i, :, max_chan_temp[i]].argmin()
    return argmin_ptp


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
    Estimate location of unit using optimization framework in Boussard et al. 2021.
    """
    num_templates, num_channels = waveforms_ptp.shape
    output = np.zeros((num_templates, 4))
    channels_pos = geom_array[:num_channels]
    for i in tqdm(range(num_templates)):
        y_init = 22
        z_com = (waveforms_ptp[i] * channels_pos[:, 1]).sum() / waveforms_ptp[i].sum()
        x_com = (waveforms_ptp[i] * channels_pos[:, 0]).sum() / waveforms_ptp[i].sum()
        alpha_init = waveforms_ptp[i].max() * (
                (((channels_pos - [x_com, z_com]) ** 2).sum(1).min() + y_init ** 2) ** 0.5)
        output[i] = least_squares(minimize_ls, x0=[x_com, z_com, y_init, alpha_init], bounds=(
            [-150, -200, 0, 0], [182, 200, np.max([y_init + 10, 150]), np.max([alpha_init + 10, 10000])]),
                                           args=(waveforms_ptp[i], channels_pos), tr_solver='exact')['x']
    return output

