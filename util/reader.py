import h5py


def map_name_event(method="s2l"):
    """
    Mapping name between short and long convention for event dict
    """
    map_name = {
        "ID": "event_id",
        "time": "event_time",
        "lon": "longitude",
        "lat": "latitude",
        "dep": "depth_km",
        "mag": "magnitude",
        "magtype": "magnitude_type",
        "tbeg": "begin_time",
        "tend": "end_time",
        "tref": "time_reference",
        "tbef": "time_before",
        "taft": "time_after",
        "source": "source",
    }
    if method == "s2l":
        pass
    elif method == "l2s":
        map_name = dict((v, k) for k, v in map_name.items())
    return map_name


def change_keyname(dict_in, map_name):
    """
    change keyname given map dict
    """
    dict_out = dict_in.copy()
    map_keys = map_name.keys()
    evt_keys = list(dict_out.keys())
    for k in evt_keys:
        if k in map_keys:
            dict_out[map_name[k]] = dict_out.pop(k)
    return dict_out


def change_event_keyname(event, method="s2l"):
    """
    s2l: short to long
    l2s: long to short
    """
    map_name = map_name_event(method=method)
    event_dict = change_keyname(event, map_name)
    return event_dict


def read_das_eventphase_data_h5(
    fn, phase=None, event=False, dataset_keys=None, attrs_only=False
):
    """
    read event phase data from hdf5 file
    Args:
        fn:  hdf5 filename
        phase: phase name list, e.g. ['P']
        dataset_keys: event phase data attributes, e.g. ['snr', 'traveltime', 'shift_index']
        event: if True, return event dict in info_list[0]
    Returns:
        data_list: list of event phase data
        info_list: list of event phase info
    """
    if isinstance(phase, str):
        phase = [phase]
    data_list = []
    info_list = []
    with h5py.File(fn, "r") as fid:
        g_phases = fid["data"]
        phase_avail = g_phases.keys()
        if phase is None:
            phase = list(phase_avail)
        for phase_name in phase:
            if not phase_name in g_phases.keys():
                raise (f"{fn} does not have phase: {phase_name}")
            g_phase = g_phases[phase_name]
            if attrs_only:
                data = []
            else:
                data = g_phase["data"][:]
            info = {}
            for key in g_phase["data"].attrs.keys():
                info[key] = g_phases[phase_name]["data"].attrs[key]
            if dataset_keys is not None:
                for key in dataset_keys:
                    if key in g_phase.keys():
                        info[key] = g_phase[key][:]
                        for kk in g_phase[key].attrs.keys():
                            info[kk] = g_phase[key].attrs[kk]
            data_list.append(data)
            info_list.append(info)
        if event:
            event_dict = dict(
                (key, fid["data"].attrs[key]) for key in fid["data"].attrs.keys()
            )
            event_dict = change_event_keyname(event_dict, method="l2s")
            info_list[0]["event"] = event_dict
    return data_list, info_list