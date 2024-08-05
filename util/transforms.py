from scipy.signal import tukey
import numpy as np
from tqdm import tqdm
import torch


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def fft_real(x):
    """
    assume fft axis in dim=-1
    """
    ntime = x.shape[-1]
    nfast = nextpow2(2 * ntime - 1)
    return torch.fft.rfft(x, n=nfast, dim=-1)


def fft_real_normalize(x):
    """"""
    x = x.clone()
    x -= torch.mean(x, dim=-1, keepdims=True)
    x /= x.square().sum(dim=-1, keepdims=True).sqrt()
    return fft_real(x)


def xcorr_lag(nlag):
    nxcor = 2 * nlag - 1
    return torch.arange(-(nxcor // 2), -(nxcor // 2) + nxcor)


def xcorr_freq(data1, data2, dt, maxlag=0.5, channel_shift=0):
    """
    cross-correlatin in frequency domain
    """
    # xcorr
    data_freq1 = fft_real_normalize(data1)
    data_freq2 = fft_real_normalize(data2)
    nlag = int(maxlag / dt)
    nfast = (data_freq1.shape[-1] - 1) * 2
    if channel_shift > 0:
        xcor_freq = data_freq1 * torch.roll(torch.conj(data_freq2), channel_shift, dims=-2)
    else:
        xcor_freq = data_freq1 * torch.conj(data_freq2)
    xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
    xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[..., nfast // 2 - nlag + 1 : nfast // 2 + nlag]
    xcor_time_axis = (xcorr_lag(nlag)*dt).numpy()
    xcor_info = {'nx': data1.shape[0], 'nt': len(xcor_time_axis), 'dt': dt, 'time_axis': xcor_time_axis}
    return xcor, xcor_info


def taper_time(data, alpha=0.8):
    taper = tukey(data.shape[-1], alpha)
    return data * torch.tensor(taper, device=data.device, dtype=data.dtype)


def moving_average(data, ma):
    """
    moving average with AvgPool1d along axis=0
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
    m = torch.nn.AvgPool1d(ma, stride=1, padding=ma // 2)
    data_ma = m(data.transpose(1, 0))[:, : data.shape[0]].transpose(1, 0)
    return data_ma


def Pkic_from_Ckij_Skij(Ckij, Skij):
    """
    polarity vector (with -1 ambiguity corrected across channel) from relative polarity matrix within cluster I
    Args:
        Ckij: relative polarity matrix with zero shift, shape=[nchan, nevent, nevent]
        Skij: relative polarity matrix with shift one along channel axis, shape=[nchan, nevent, nevent]
    Returns:
        Pkic: corrected polarity vectors, shape=[nchan, nevent]
    """
    nchan, nevent, _ = Ckij.shape
    Pki = torch.zeros([nchan, nevent])
    Pkic = torch.zeros([nchan, nevent])
    sigma_perc_0 = torch.zeros(nchan)
    sigma_perc_1 = torch.zeros(nchan)
    sigma_ratio_0 = torch.zeros(nchan)
    sigma_ratio_1 = torch.zeros(nchan)
    for k in tqdm(range(nchan)):
        u, s, v = torch.linalg.svd(Ckij[k, :, :])
        Pki[k, :] = u[:, 0]  # @Rijk[k, :, :]
        Pkic[k, :] = u[:, 0]
        sigma_perc_0[k] = s[0] / torch.sum(s)
        sigma_ratio_0[k] = s[0] / s[1]
        if k >= 1:
            u, s, v = torch.linalg.svd(Skij[k, :, :])
            csign = torch.sign(torch.dot(u[:, 0], v[0, :]))
            Pkic[k, :] *= torch.sign(torch.dot(Pkic[k, :], Pkic[k - 1, :]) * csign)
            sigma_perc_1[k] = s[0] / torch.sum(s)
            sigma_ratio_1[k] = s[0] / s[1]
    Pkic_info = {
        "Pki": Pki,
        "sigma_perc_0": sigma_perc_0,
        "sigma_perc_1": sigma_perc_1,
        "sigma_ratio_0": sigma_ratio_0,
        "sigma_ratio_1": sigma_ratio_1,
    }
    return Pkic, Pkic_info