from scripts.neuron_signal_generator import generate_signal
from scripts.noise_and_filtering import generate_electrode_signal

# generate signal
neuron_signal = generate_signal()

generate_electrode_signal(
    signal=neuron_signal,
    decay_type='square',
    decay_rate=2,
    noise_type='gaussian',
    noise_std=0.5,
    filter_type='bandpass',
    low=500,
    high=3000
)