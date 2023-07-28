import numpy as np

from classes.grid import Grid
from helpers.constants import SAMPLE_RATE, SIGNAL_LENGTH


def add_decay_and_noise(signal, decay_type, decay_rate, noise_type, noise_std, distance_from_neuron):
    neuron_signal = np.array(signal["y"])

    grid = Grid(decay_rate=decay_rate, signal_length=SIGNAL_LENGTH, sample_rate=SAMPLE_RATE, decay_type=decay_type)

    decay = grid.decay_mulitplier(distance=distance_from_neuron)
    electrode_recording = (neuron_signal * decay) + grid.noise(neuron_signal, type=noise_type, noise_stddev=noise_std)

    return electrode_recording


def filter_electrode_signal(signal, filter_type, low, high):
    pass


def generate_electrode_signal(signal, decay_type, decay_rate, noise_type, noise_std, filter_type, low, high, distance_from_neuron=3):

    electrode_recording = add_decay_and_noise(signal, decay_type, decay_rate, noise_type, noise_std, distance_from_neuron)

    filtered_recording = filter_electrode_signal(electrode_recording, filter_type, low, high)

    print(filtered_recording)