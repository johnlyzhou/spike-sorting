"""
Optimization metrics to use when localizing waveform templates.
"""


def minimize_ls(vec, ptps, channels_pos):
    return ptps - vec[3] / (((channels_pos - [vec[0], vec[1]]) ** 2).sum(1) + vec[2] ** 2) ** 0.5


def minimize_ls_bis(vec, ptps, channels_pos):
    return ptps * (((channels_pos - [vec[0], vec[1]]) ** 2).sum(1) + vec[2] ** 2) ** 0.5 - vec[3]


def minimize_summed(vec, ptps, channels_pos):
    return ((ptps - vec[3] / (((channels_pos - [vec[0], vec[1]]) ** 2).sum(1) + vec[2] ** 2) ** 0.5) ** 2).sum()


def minimize_de(vec, ptps, channels_pos):
    return (ptps * (((channels_pos - [vec[0], vec[1]]) ** 2).sum(1) + 400) ** 0.5 - vec[3]).mean()
