import numpy as np

from classes.neuron import EulerLIF, RKLIF
from helpers.constants import SAMPLE_RATE, SIGNAL_LENGTH

def generate_signal(neuron_type="standard", lmbda=14, v_rest=-70, v_thres=-10, t_ref=0.02, fix_random_seed=False):

    if fix_random_seed:
        seed = 1729
    else:
        seed = None

    # initialise relevant neuron type (the frontend should make it impossible to choose an unknown neuron type)
    if neuron_type == "euler":
        neuron = EulerLIF(v_rest=v_rest, sample_rate=SAMPLE_RATE, thr=v_thres, tf=SIGNAL_LENGTH, t_ref=t_ref, lmbda=lmbda, seed=seed)
    elif neuron_type == "standard":
        neuron = RKLIF(v_rest=v_rest, sample_rate=SAMPLE_RATE, thr=v_thres, tf=SIGNAL_LENGTH, t_ref=t_ref, lmbda=lmbda, seed=seed)
    else:
        raise ValueError("Unknown Neuron Type")
    
    # generate the signal
    values = neuron.simulate()

    # generate a time array for the signal
    time = np.linspace(0, SIGNAL_LENGTH, SAMPLE_RATE * SIGNAL_LENGTH)

    return { "x": time.tolist(), "y": values.tolist() }