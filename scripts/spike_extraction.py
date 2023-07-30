import numpy as np

from classes.sorter import SpikeSorter
from classes.grid import Grid
from helpers.constants import SAMPLE_RATE

def get_waveform_data(signals, multiplier, waveform_duration):

    # intialise the spike sorter
    sorter = SpikeSorter(threshold_factor=multiplier, sample_rate=SAMPLE_RATE, waveform_duration=waveform_duration)

    # get the spikes for each recording
    spikes = sorter.get_spikes(signals)

    # convert the spikes into a numpy array
    spike_indices = np.array(spikes)

    # neuron spikes is used to determine the true labels of the spike data, so we'll just give an empty array
    neuron_spikes = []

    # get the locations of all the spikes detected by all channels, and true spike posititons
    merged_spike_indices, true_labels = sorter.merge_spike_indices(spike_indices, neuron_spikes, tolerance=30)

    # extract the waveform data from each of the identified spikes across all electrodes
    waveforms, waveform_info = sorter.get_all_waveforms(
        signals, 
        merged_spike_indices, 
        recenter=True,
        labels=true_labels 
    )

    return waveforms, waveform_info


def get_threshold_value(signal, multiplier):

    # intialise the spike sorter (waveform duration doesn't mattter)
    sorter = SpikeSorter(threshold_factor=multiplier, sample_rate=SAMPLE_RATE, waveform_duration=0.3)

    # calculate the threshold value
    threshold = sorter.get_threshold_value(signal)

    return threshold
