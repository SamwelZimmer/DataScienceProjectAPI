from typing import List, Tuple
import numpy as np

from helpers.constants import SAMPLE_RATE, SIGNAL_LENGTH, RANDOM_STATE
from classes.grid import Grid
from classes.sorter import SpikeSorter

def simulate_recording(placements, neuron_params, processing_params):

    neuron_type, lmbda, v_rest, v_thres, t_ref, fix_random_seed = neuron_params["neuron_type"], neuron_params["lambda"], neuron_params["v_rest"], neuron_params["v_thres"], neuron_params["t_ref"], neuron_params["fix_random_seed"]
    decay_type, decay_rate, noise_type, noise_std, filter_type, low, high = processing_params["decay_type"], processing_params["decay_rate"], processing_params["noise_type"], processing_params["noise_std"], processing_params["filter_type"], processing_params["low"], processing_params["high"]

    # fix random seed if requested
    if fix_random_seed:
        seed = RANDOM_STATE
    else:
        seed = None
    
    # convert placements into row and column coordinates
    placements = convert_placements(placements)

    # get the positions of the neurons and electrodes
    neuron_coords, electrode_coords = get_positions(placements)

    # initialise the grid object
    grid = Grid(
        signal_length=SIGNAL_LENGTH,
        sample_rate=SAMPLE_RATE,
        random_state=seed,
        size=len(placements),
        decay_rate=decay_rate,
        decay_type=decay_type,
        neuron_type=neuron_type,
        lmbda=lmbda,
        v_rest=v_rest,
        v_thres=v_thres,
        t_ref=t_ref,
        noise_type=noise_type,
        noise_std=noise_std
    )

    # add neurons and electrodes to the grid
    grid.add_neurons(neuron_coords)
    grid.add_electrodes(electrode_coords)

    # simulate neuron stimulation and electrode recording
    grid.generate_signals()

    # get the electrode recordings as a list
    signals = [grid.electrode_dict[i]["signal"] for i in grid.electrode_dict.keys()]

    # initialise the spike sorter object (threshold factor and waveform_duration not important in the case)
    sorter = SpikeSorter(threshold_factor=5, sample_rate=SAMPLE_RATE, waveform_duration=0.3)

    # filter the signals
    filtered_signals = sorter.filter_signals(signals=signals, type=filter_type, low=low, high=high, order=4)

    # convert this list of numpy arrays to regular arrays
    filtered_signals = [np_signal.tolist() for np_signal in filtered_signals]

    # form a time array
    time = np.linspace(0, SIGNAL_LENGTH, SAMPLE_RATE * SIGNAL_LENGTH)

    # return signals
    return time.tolist(), filtered_signals


def get_positions(grid: List[List[int]]):
    ones = []
    twos = []

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                ones.append((i, j))
            elif grid[i][j] == 2:
                twos.append((i, j))

    return ones, twos


def convert_placements(placements: List[int]) -> List[List[int]]:
    size = int(np.sqrt(len(placements)))
    return [placements[i*size:(i+1)*size] for i in range(size)]
