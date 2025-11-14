import numpy as np


def window_spikes(spike_times, end_time, window, slide, absoulte=False, binary=True):   
    window_list = []
    binarized_list = []
    
    binarized = binarize(spike_times, end_time)
    
    start_idx = 0
    while(start_idx + window <= len(binarized)):
        windowed = binarized[start_idx : start_idx + window]
        binarized_list.append(windowed)
        
        t = np.where(windowed == 1)[0]
        if absoulte:      
            window_list.append(t+start_idx)
        else:
            window_list.append(t)
        start_idx += slide
    
    if binary:
        return binarized_list
    else:
        return window_list


def binarize(spike_times, end_time):
    ret = np.zeros(end_time)
    
    for t in spike_times:
        if t >= end_time:
            continue
        ret[int(t)] = 1
        
    return ret


def binarize_all(spike_arr, end_time):
    channels = len(spike_arr)
    ret = []
    for c in range(channels):
        ret.append(binarize(spike_arr[c], end_time))
    
    return np.asarray(ret)


def back_to_time_list_all(binarized_all):
    ret = []
    time = binarized_all.shape[1]
    channel = binarized_all.shape[0]
    
    for c in range(channel):
    
        spike_train = []
        for t in range(time):
            if binarized_all[c, t]:
                spike_train.append(t)
        ret.append(spike_train)
        
    return ret