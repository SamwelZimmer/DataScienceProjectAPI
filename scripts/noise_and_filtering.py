import numpy as np
from typing import Dict, List

from classes.grid import Grid
from classes.sorter import SpikeSorter
from helpers.constants import SAMPLE_RATE, SIGNAL_LENGTH


def add_decay_and_noise(signal: Dict[str, List[float]], decay_type: str, decay_rate: float, noise_type: str, noise_std: float, distance_from_neuron: float) -> np.ndarray:
    """
    This function simulates signal attenuation and noise in a neuron recording.
    
    Parameters:
    -----------
    signal: dict
        The input neuron signal as a dictionary with "x" and "y" keys.
    decay_type: str
        The type of decay to be applied.
    decay_rate: float
        The rate of decay.
    noise_type: str
        The type of noise to be added.
    noise_std: float
        The standard deviation of the noise.
    distance_from_neuron: float
        The distance from the neuron.

    Returns:
    --------
    np.ndarray
        The neuron signal with decay and noise applied.
    """

    # convert the voltage signal data to a numpy array
    neuron_signal = np.array(signal["y"])

    # initialise a Grid object to simulate an electrode recording
    grid = Grid(decay_rate=decay_rate, signal_length=SIGNAL_LENGTH, sample_rate=SAMPLE_RATE, decay_type=decay_type)

    # simulate signal attenuation and noise
    decay = grid.decay_mulitplier(distance=distance_from_neuron)
    electrode_recording = (neuron_signal * decay) + grid.noise(neuron_signal, type=noise_type, noise_stddev=noise_std)

    return electrode_recording


def filter_electrode_signal(signal: np.ndarray, filter_type: str, low: float, high: float, order: int) -> np.ndarray:
    """
    This function applies a filter to the input signal.
    
    Parameters:
    -----------
    signal: np.ndarray
        The input signal as a numpy array.
    filter_type: str
        The type of filter to be applied.
    low: float
        The lower limit of the filter.
    high: float
        The upper limit of the filter.
    order: int
        The order of the filter.

    Returns:
    --------
    np.ndarray
        The filtered signal.
    """

    # initialise a SpikeSorter object (threshold_factor and waveform_duration don't matter for this script)
    sorter = SpikeSorter(threshold_factor=5, sample_rate=SAMPLE_RATE, waveform_duration=0.3)

    # filter the signal
    filtered_signal = sorter.filter_signal(signal=signal, type=filter_type, low=low, high=high, order=order)

    return filtered_signal


def generate_electrode_signal(signal: Dict[str, List[float]], decay_type: str, decay_rate: float, noise_type: str, noise_std: float, filter_type: str, low: float, high: float, distance_from_neuron: float=3, order: int=4) -> Dict[str, List[float]]:
    """
    This function generates an electrode signal by first adding decay and noise to the input signal, 
    then applying a filter.

    Parameters:
    -----------
    signal: dict
        The input signal as a dictionary with "x" and "y" keys.
    decay_type: str
        The type of decay to be applied.
    decay_rate: float
        The rate of decay.
    noise_type: str
        The type of noise to be added.
    noise_std: float
        The standard deviation of the noise.
    filter_type: str
        The type of filter to be applied.
    low: float
        The lower limit of the filter.
    high: float
        The upper limit of the filter.
    distance_from_neuron: float, optional
        The distance from the neuron. Default is 3.
    order: int, optional
        The order of the filter. Default is 4.

    Returns:
    --------
    dict
        The generated electrode signal as a dictionary with "x" and "y" keys.
    """

    # get the time component of the signal
    time = signal["x"]

    # simulate attenuation and noise
    electrode_recording = add_decay_and_noise(signal, decay_type, decay_rate, noise_type, noise_std, distance_from_neuron)

    # filter the signal
    filtered_recording = filter_electrode_signal(electrode_recording, filter_type, low, high, order)

    # return the signal with voltage and time components
    return { "x": time, "y": filtered_recording.tolist() }