import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable


_fontsize = 15
params = {
    "image.interpolation": "nearest",
    "savefig.dpi": 300,  # to adjust notebook inline plot size
    "axes.labelsize": _fontsize,  # fontsize for x and y labels (was 10)
    "axes.titlesize": _fontsize,
    "font.size": _fontsize,
    "legend.fontsize": _fontsize,
    "xtick.labelsize": _fontsize,
    "ytick.labelsize": _fontsize,
    "text.usetex": False,
}
import matplotlib

matplotlib.rcParams.update(params)


def set_to_numpy(data, xx, tt):
    """"""
    if not isinstance(data, np.ndarray):
        data = data.numpy()
    if xx is not None and not isinstance(xx, np.ndarray):
        xx = xx.numpy()
    if tt is not None and not isinstance(tt, np.ndarray):
        tt = tt.numpy()
    return data, xx, tt


def set_skip_n(n, n_max=5000):
    """
    set skip_n based on maximum n to show
    Args:
        n: number of data points along some-axis
        n_max: maximum number of time data points to show along some-axis
    """
    if n > n_max:
        skip_n = np.ceil(n / n_max).astype(int)
        return skip_n
    else:
        return 1


def set_skip_n_window(nx, xx, xlimit, n_max=5000):
    """
    set skip_n based on windowed x range
    Args:
        xx: x-axis
        xlimit: xlimit of the window
        n_max:  maximum number of time data points to show along some-axis
    """
    if xx is None:
        xx = np.arange(nx)
    if xlimit is not None:
        n = np.count_nonzero(np.logical_and(xx >= xlimit[0], xx <= xlimit[1]))
    else:
        n = len(xx)
    return set_skip_n(n, n_max=n_max)


def invert_axis(ax, axis="y"):
    """"""
    if axis == "y":
        limit = ax.get_ylim()
        if limit[-1] > limit[0]:
            ax.invert_yaxis()
    if axis == "x":
        limit = ax.get_xlim()
        if limit[-1] > limit[0]:
            ax.invert_xaxis()


def ix_axis_(xx, ix=None, xlimit=None, skip_x=None):
    """
    select subset of axis
    Args:
        xx: x-axis
        ix: selected indices
        xlimit: xlimit for x-axis
        skip_x: skip every skip_x element along x-axis
    """
    if ix is None:
        if xlimit is not None:
            ix = np.logical_and(xx >= xlimit[0], xx <= xlimit[1])
            xx = xx[ix]
        else:
            ix = np.logical_and(xx >= np.min(xx), xx <= np.max(xx))
        if skip_x is None:
            skip_x = 1
        xx = xx[::skip_x]
        ix = np.where(ix)[0][::skip_x]
    else:
        xx = xx[ix]
    return xx, ix


def select_data_window(
    data,  xx=None, tt=None, ix=None, it=None, xlimit=None, tlimit=None, skip_x=None, skip_t=None
):
    """
    select data window
    Args:
        data: input 2d data [nx, nt]
        xx: x-axis
        tt: t-axis
        ix: selected indices for x-axis
        it: selected indices for t-axis
        xlimit: if not None, select data x-axis within xlimit
        tlimit: if not None, select data t-axis within tlimit
        skip_x: skip every skip_x element along x-axis
        skip_t: skip every skip_t element along t-axis
    Returns:
        data_window: selected window
        xx: x-axis for selected window
        tt: t-axis for selected window
    """
    nchan, ntime = data.shape
    if xx is None:
        xx = np.arange(nchan)
    if tt is None:
        tt = np.arange(ntime)
    xx, ix = ix_axis_(xx, ix=ix, xlimit=xlimit, skip_x=skip_x)
    tt, it = ix_axis_(tt, ix=it, xlimit=tlimit, skip_x=skip_t)
    data_select = data[ix, :][:, it]
    return data_select, xx, tt


def show_data(
    data,
    xx=None,
    tt=None,
    skip_x=None,
    skip_t=None,
    qdp=True,
    nmax_x=1e5,
    nmax_t=5e3,
    xlimit=None,
    tlimit=None,
    perc=99,
    vmin=None,
    vmax=None,
    normalize=False,
    style="seismic",
    cmap=plt.get_cmap("seismic"),
    figsize=(15, 7),
    ax=None,
    rasterized=True,
    cbar=True,
    cbar_size="3%",
    cbar_pad=0.05,
    cbar_label='',
    tref_datetime=None,
    locator=None,
    interval=None,
    grid=False,
):
    """
    show das data: data shape in [nchan, ntime]
    Args:
        data; shape=[nchan, ntime]
        xx: axis along channel
        tt: axis along time (can be datetime)
        skip_x: skip every skip_x points along channel axis
        skip_t: skip every skip_t points along time axis
        qdp: quick dirty plot, if False, skip_x and skip_t are turned off
        xlimit: channel-axis limit
        tlimit: time-axis limit
        perc: clipping
        vmin:
        vmax:
        style: 'seismic': time axis in vertical direction
                'normal':  time axis in horizontal direction
        rasterized: if rasterize the pcolormesh when saved to pdf
    """
    nchan, ntime = data.shape
    data, xx, tt = set_to_numpy(data, xx, tt)
    if tref_datetime is not None and tlimit is not None and isinstance(tlimit[0], datetime):
        tlimit = [(t-tref_datetime).total_seconds() for t in tlimit]
    if not qdp:
        skip_x = 1
        skip_t = 1
    if skip_x is None:
        skip_x = set_skip_n_window(nchan, xx, xlimit, n_max=nmax_x)
        if skip_x > 1:
            print(f"Downsample in channel axis during plot: {skip_x}")
    if skip_t is None:
        skip_t = set_skip_n_window(ntime, tt, tlimit, n_max=nmax_t)
        if skip_t > 1:
            print(f"Downsample in time axis during plot: {skip_t}")
    data_show, xx_show, tt_show = select_data_window(
        data, xx=xx, tt=tt, xlimit=xlimit, tlimit=tlimit, skip_x=skip_x, skip_t=skip_t
    )
    if normalize:
        data_show = data_show / np.std(data, axis=-1, keepdims=True)
    if tref_datetime is not None:
        # tt_show = [tref_datetime + timedelta(seconds=t) for t in tt_show]
        # print(tt_show[0])
        tt_show = tref_datetime + timedelta(seconds=1) * tt_show
        #tt_show = tref_datetime + timedelta(seconds=1) * (tt_show-tt_show[0])
    clipVal = np.percentile(np.absolute(data_show), perc)
    if vmin is None:
        vmin = -clipVal
    if vmax is None:
        vmax = clipVal
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if style == "seismic":
        pcm = ax.pcolormesh(
            xx_show,
            tt_show,
            data_show.T,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            shading="nearest",
            rasterized=rasterized,
        )
        ax.set_xlabel("Channel")
        ax.set_ylabel("Time (sec)")
        invert_axis(ax, axis="y")
    elif style == "normal":
        pcm = ax.pcolormesh(
            tt_show,
            xx_show,
            data_show,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            shading="nearest",
            rasterized=rasterized,
        )
        ax.set_ylabel("Channel")
        ax.set_xlabel("Time (sec)")
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
        hcbar = fig.colorbar(pcm, cax=cax, orientation="vertical")
        hcbar.set_label(cbar_label, rotation=270)
    if tref_datetime is not None:
        if style == "seismic":
            format_axis = "y"
            ax.set_ylabel("Time")
        elif style == "normal":
            format_axis = "x"
            ax.set_xlabel("Time")
        fig, ax = _axis_datetime_formatter(
            fig, ax, tt_show, interval, locator, axis=format_axis
        )
    if grid:
        ax.grid(visible=True)
    return fig, ax


def show_das_data(
    data,
    info,
    xlimit=None,
    tlimit=None,
    skip_x=None,
    skip_t=None,
    usedatetime=False,
    locator=None,
    interval=None,
    ax=None,
    cbar=True,
    **kwargs,
):
    """
    show_data wrapper for DAS record:
    Inputs:
        data: das data: shape in [nchan, ntime]
        info: das info dict: keys=['begTime', 'endTime', 'nx', 'nt', 'dx', 'dt']
        xlimit: channel range
        tlimit: time range
        usedatetime: use datetime for time axis or not
            interval: interval for datetime time-axis format
            locator: locator for datetime time-axis format: ['seconds', 'minutes', 'hours', 'days', 'months']
        **kwargs: to be passed to show_data
    """
    if "channel_axis" in info.keys():
        xx = info["channel_axis"]
    else:
        xx = np.arange(info["nx"])
    if "time_axis" in info.keys():
        tt = info["time_axis"]
    else:
        tt = np.arange(info["nt"]) * info["dt"]
    # # use datetime for time axis
    if usedatetime:
        tref_datetime = info["begTime"]
    else:
        tref_datetime = None
    fig, ax = show_data(
        data,
        xx=xx,
        tt=tt,
        xlimit=xlimit,
        tlimit=tlimit,
        skip_x=skip_x,
        skip_t=skip_t,
        ax=ax,
        cbar=cbar,
        tref_datetime=tref_datetime,
        locator=locator,
        interval=interval,
        **kwargs,
    )
    return fig, ax


def show_cc2d_data(
    data,
    info,
    xlimit=None,
    tlimit=None,
    usedatetime=False,
    locator=None,
    interval=None,
    ax=None,
    **kwargs,
):
    """
    show_data wrapper for TemplateMatch cc2d record:
    Inputs:
        data: das data: shape in [nchan, ntime]
        info: das info dict: keys=['begTime', 'endTime', 'nx', 'nt', 'dx', 'dt', 'tref]
        xlimit: channel range
        tlimit: time range
        usedatetime: use datetime for time axis or not
            interval: interval for datetime time-axis format
            locator: locator for datetime time-axis format: ['seconds', 'minutes', 'hours', 'days', 'months']
        **kwargs: to be passed to show_data
    """
    xx = np.arange(info["nx"])
    if "time_axis" in info.keys():
        tt = info["time_axis"]
    else:
        tt = np.arange(info["nt"]) * info["dt"]
    if xlimit is not None:
        ix = np.logical_and(xx >= xlimit[0], xx <= xlimit[1])
        xx = xx[ix]
    else:
        ix = np.logical_and(xx >= xx[0], xx <= xx[-1])
    if tlimit is not None:
        it = np.logical_and(tt >= tlimit[0], tt <= tlimit[1])
        tt = tt[it]
    else:
        it = np.logical_and(tt >= tt[0], tt <= tt[-1])
    # use datetime for time axis
    if usedatetime:
        tt = [info["begTime"] + timedelta(seconds=t) for t in tt]
    if xlimit is None and tlimit is None:
        fig, ax = show_data(data, xx=xx, tt=tt, ax=ax, **kwargs)
    else:
        fig, ax = show_data(data[np.ix_(ix, it)], xx=xx, tt=tt, ax=ax, **kwargs)
    # format time axis
    if usedatetime:
        style = kwargs.get("style", "seismic")
        if style == "seismic":
            format_axis = "y"
        elif style == "normal":
            format_axis = "x"
        fig, ax = _axis_datetime_formatter(fig, ax, tt, interval, locator, format_axis)
    return fig, ax


def show_polarity_matrix(Pik, figsize=(5, 5), perc=None, vmin=None, vmax=None, rasterized=True, ax=None):
    """
    Show polarity matrix
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if perc is not None:
        clipVal = np.percentile(np.absolute(Pik), perc)
        vmin = -clipVal
        vmax = clipVal
    else:
        if vmin is None:
            vmin = -1
        if vmax is None:
            vmax =  1 
    ax.pcolormesh(Pik, cmap=plt.get_cmap('RdBu_r'), vmin=vmin, vmax=vmax, rasterized=rasterized)
    return fig, ax


# Helper functions
def _axis_datetime_formatter(
    fig, ax, time_axis, interval, locator, format=None, axis="x"
):
    """
    time_axis: time dimension array
    locator: locator for datetime time-axis format: ['seconds', 'minutes', 'hours', 'days', 'months']
    interval: set up tick every N sec/min/hour/day...
    format:  datetime format mannually given to overwrite dformatter
    """
    # format datetime for time axis
    tspan = (time_axis[-1] - time_axis[0]).total_seconds()
    dlocator, dformatter = _check_time_span(tspan, interval=interval, locator=locator)
    if format is not None:
        dformatter = format
    if axis == "x":
        ax.xaxis.set_major_formatter(mdates.DateFormatter(dformatter))
        ax.xaxis.set_major_locator(dlocator)
    elif axis == "y":
        ax.yaxis.set_major_formatter(mdates.DateFormatter(dformatter))
        ax.yaxis.set_major_locator(dlocator)
    fig.autofmt_xdate()
    return fig, ax


def _check_time_span(tspan, interval=None, locator=None):
    # return the corresponding mdates locator given tspan in seconds
    nminute = tspan / 60
    if nminute < 1 or locator == "seconds":
        if interval is None:
            interval = max(np.round(tspan / 10).astype(int), 1)
        return mdates.SecondLocator(interval=interval), "%H:%M:%s"
    nhour = nminute / 60
    if nhour < 1 or locator == "minutes":
        if interval is None:
            interval = max(np.round(nminute / 10).astype(int), 1)
        return mdates.MinuteLocator(interval=interval), "%H:%M"
    nday = nhour / 24
    if nday < 1 or locator == "hours":
        if interval is None:
            interval = max(np.round(nhour / 10).astype(int), 1)
        return mdates.HourLocator(interval=interval), "%H:%M"
    nmonth = nday / 31
    if nmonth < 1 or locator == "days":
        if interval is None:
            interval = max(np.round(nday / 10).astype(int), 1)
        return mdates.DayLocator(interval=interval), "%m-%d"
    nyear = nday / 365
    if nyear < 1 or locator == "months":
        if interval is None:
            interval = max(np.round(nmonth / 10).astype(int), 1)
        return mdates.MonthLocator(interval=interval), "%Y-%m-%d"
