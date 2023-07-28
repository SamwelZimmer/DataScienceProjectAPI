# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Tuple
# import pandas as pd
# import scipy

# class SpikeSorter():
#     def __init__(self, threshold_factor, waveform_duration, sample_rate=25000):
#         self.threshold_factor = threshold_factor
#         self.sample_rate = sample_rate
#         self.dt = 1 / self.sample_rate
#         self.waveform_duration = waveform_duration

#     ## needs modification
#     def filter_signal(self, signal, type, low: int, high: int, order: int):
#         # the nyquist frequency
#         nyquist = 0.5 * self.sample_rate

#         # form upper/lower frequency bounds
#         lower = low / nyquist
#         upper = high / nyquist

#         if type == "bandpass":
#             # filter the data using the butterworth filter
#             b, a = scipy.signal.butter(order, [lower, upper], btype='band')

#             # this compensates for phase shifting as it does forward and backwards pass
#             filtered_signal = scipy.signal.filtfilt(b, a, signal, padlen=len(signal)-1)

#             return filtered_signal
#         else:
#             print("Unknown Filter Type: ", type)

#     def filter_signals(self, signals, type: str="bandpass", low: int=500, high: int=3000, order: int=4, visualise: bool=False, figsize: Tuple[int]=(8, 8)):

#         filtered_signals = [self.filter_signal(signal=signal, type=type, low=low, high=high, order=order) for signal in signals]

#         if visualise:
#             fig, axs = plt.subplots(len(filtered_signals), 1, figsize=figsize)

#             if len(filtered_signals) == 1:
#                 axs = [axs]

#             for electrode_id, ax in enumerate(axs):
#                 filtered_signal = filtered_signals[electrode_id]
#                 time = np.arange(len(filtered_signal))
#                 ax.plot(time, filtered_signal, label=f'Electrode {electrode_id} Combined Signal', c="k", lw=0.5)
#                 ax.set_title(f'Electrode {electrode_id} Combined Signal')
#                 ax.set_xlabel('Time (s)')
#                 ax.set_ylabel('Filtered Signal (mV)')
#                 ax.legend()

#             plt.tight_layout()
#             plt.show()
            
#         return filtered_signals
    
#     def get_threshold_value(self, signal: np.ndarray, is_negative: bool=False) -> float:
#         # calculate robust s.d. using the mean absolute deviation (MAD)
#         sigma = np.median(np.abs(signal - np.median(signal)) / 0.6745)

#         df = pd.DataFrame(signal)
#         df.to_csv("../data/example_signal.csv")

#         # set the threshold for this channel
#         if is_negative:
#             return -1 * self.threshold_factor * sigma # not sure if its always negative??
#         else:
#             return self.threshold_factor * sigma
    
#     def detect_spikes(self, y: np.ndarray, threshold: float, minimum_gap: int=1, use_absolute_threshold: bool=False, flipped: bool=False) -> np.ndarray:
#         """
#         Detects spikes (or troughs) in a given signal.

#         Parameters
#         ----------
#         y: np.ndarray
#             The input signal.
#         threshold: float
#             The threshold value for spike detection. If `use_absolute_threshold` is False, this is a relative value.
#         minimum_gap: int, optional
#             The minimum number of samples between spikes. Default is 1.
#         use_absolute_threshold : bool, optional
#             If True, `threshold` is an absolute value. If False, `threshold` is a relative value. Default is False.
#         flipped: bool, optional
#             If True, the function will detect troughs (downward spikes) instead of peaks (upward spikes). Default is False.

#         Returns
#         -------
#         np.ndarray
#             An array of indices in `y` where spikes were detected.

#         Raises
#         ------
#         ValueError
#             If `y` is an unsigned array.

#         Notes
#         -----
#         This function uses a first order difference method to detect spikes. It first computes the first differential of `y`, then finds the indices where the differential changes sign (indicating a peak or trough). It then filters these indices based on the `threshold` value and the `minimum_gap` between spikes.

#         If `flipped` is True, the function detects troughs instead of peaks. This is done by reversing the sign of the differential and the `threshold` value.

#         The function returns an array of indices in `y` where spikes (or troughs) were detected.
#         """

#         # Check if y is unsigned array
#         if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
#             raise ValueError("y must be signed")
        
#         # Convert relative threshold to absolute if necessary
#         if not use_absolute_threshold:
#             threshold = threshold * (np.max(y) - np.min(y)) + np.min(y)

#         # Compute the first differential
#         dy = np.diff(y)

#         # Propagate left and right values successively to fill all plateau pixels (no gradient)
#         zeros, = np.where(dy == 0)

#         # Check if the signal is totally flat
#         if len(zeros) == len(y) - 1:
#             return np.array([], dtype=np.int64)
        
#         # Find the peaks or troughs
#         if flipped:
#             extrema = np.where((np.hstack([dy, 0.0]) > 0.0) & (np.hstack([0.0, dy]) < 0.0) & (np.less(y, threshold)))[0]
#         else:
#             extrema = np.where((np.hstack([dy, 0.0]) < 0.0) & (np.hstack([0.0, dy]) > 0.0) & (np.greater(y, threshold)))[0]

#         # Handle multiple peaks or troughs, respecting the minimum distance
#         if extrema.size > 1 and minimum_gap > 1:
#             sorted_extrema = extrema[np.argsort(y[extrema])][::-1]
#             rem = np.ones(y.size, dtype=bool)
#             rem[extrema] = False

#             for extremum in sorted_extrema:
#                 if not rem[extremum]:
#                     sl = slice(max(0, extremum - minimum_gap), extremum + minimum_gap + 1)
#                     rem[sl] = True
#                     rem[extremum] = False

#             extrema = np.arange(y.size)[~rem]

#         return extrema

#     def __get_true_labels(self, merged_spikes: np.ndarray, neuron_spikes: List[List]) -> List[Tuple]:
#         true_labels = []
#         for i, neuron in enumerate(neuron_spikes):
#             for spike in neuron:
#                 if spike in merged_spikes:
#                     true_labels.append((spike, i))
        

#         # ensure the list is returned in order of index
#         true_labels = sorted(true_labels, key=lambda x: x[0])
#         return [i[1] for i in true_labels]

#     def merge_spike_indices(self, spike_indices: np.ndarray[np.ndarray], neuron_spikes: List[List], tolerance: int=1) -> np.ndarray:
#         """
#         Merge spike indices from multiple channels into a single array. 
#         If the points are close together only the midpoint is added.

#         Parameters:
#         -----------
#         spike_indices: list of numpy arrays
#             Each numpy array contains spike indices for a single channel.
#         tolerance: int, optional
#             The maximum distance between spike indices that will be considered as the same spike.
#             Indices within this distance will be replaced by their midpoint.

#         Returns:
#         --------
#         numpy array
#             A single array of merged spike indices.
#         """

#         # flatten all indices into a single list
#         all_indices = np.concatenate(spike_indices)
        
#         # sort the indices
#         all_indices.sort()
        
#         # initialise the output list with the first index
#         merged_indices = [all_indices[0]]
        
#         # go through the sorted list and merge indices that are close together
#         for index in all_indices[1:]:
#             if index - merged_indices[-1] <= tolerance:
#                 # if the current index is close to the last one in the output list, replace the last one with their average (rounded to nearest integer)
#                 merged_indices[-1] = round((merged_indices[-1] + index) / 2)
#             else:
#                 # if the current index is not close to the last one, add it to the output list
#                 merged_indices.append(index)
        
#         true_labels = self.__get_true_labels(merged_indices, neuron_spikes)
        
#         return np.array(merged_indices, dtype=int), true_labels

#     def get_waveforms(self, y: np.ndarray, spike_indices: np.ndarray, recenter: bool, window_shift_ratio: float=0.5) -> Tuple[np.ndarray[np.ndarray], List[dict]]:
#         """
#         Extracts waveforms from a signal at given indices.

#         Parameters
#         ----------
#         y : np.ndarray
#             The input signal.
#         spike_indices : np.ndarray
#             The indices in `y` where spikes were detected.
#         duration : int
#             The duration of the waveform in milliseconds.
#         sample_rate : int
#             The sample rate of the signal in Hz.
#         window_shift_ratio : float, optional
#             The ratio of the window size to shift the window to the left of the spike. Default is 0.5.

#         Returns
#         -------
#         waveforms : np.ndarray
#             A nested numpy array of extracted waveforms.
#         waveform_info : list of dict
#             A list of dictionaries containing information about each extracted waveform.

#         Notes
#         -----
#         The dictionaries of waveform_info contain the starting and finishing index of the waveform, its greatest positive 
#         and negative amplitudes and the values of the waveform (corresponding to the data in `waveforms`).
        
#         """

#         # calculate the number of samples to extract around each spike
#         window_size = int(self.sample_rate * self.waveform_duration / 1000)

#         # calculate the number of samples to shift the window
#         shift = int(window_size * window_shift_ratio)

#         waveforms = []
#         waveform_info = []

#         # iterate over the spike indices
#         for i in spike_indices:

#             # calculate the start and end of the window
#             start = int(i - shift)
#             end = int(start + window_size)

#             # extract the waveform
#             waveform = y[start:end]

#             # shift the waveform so that the peak is in the center of the window
#             if recenter:
#                 peak_position = np.where(waveform == np.max(waveform))[0][0]
#                 center_position = window_size // 2
#                 difference = peak_position - center_position

#                 start = int(i - shift)
#                 end = int(start + window_size)
#                 waveform = y[start + difference:end + difference]

#             # append the waveform to the list
#             waveforms.append(waveform)

#             # store information about the waveform
#             spike_info = {
#                 'spike_start': start,
#                 'spike_end': end,
#                 'lowest_value': np.min(waveform),
#                 'highest_value': np.max(waveform),
#                 'values': waveform
#             }

#             waveform_info.append(spike_info)

#         # Convert the lists to numpy arrays
#         waveforms = np.array(waveforms)

#         return waveforms, waveform_info

#     def get_all_waveforms(self, signals: np.ndarray, spike_indices: np.ndarray, recenter=False, visualise: bool=False, labels: List=[]):
#         waveforms = []
#         waveform_info = []

#         # loop through each of the electrode signals
#         for i, signal in enumerate(signals):

#             # get waveforms for this channel
#             waveforms_channel, waveform_info_channel = self.get_waveforms(y=signal, spike_indices=spike_indices, recenter=recenter)
            
#             # append the waveforms and waveform_info to the lists
#             waveforms.append(waveforms_channel)
#             waveform_info.append(waveform_info_channel)

#         # convert waveforms and waveform_info to numpy arrays
#         waveforms = np.array(waveforms)
#         waveform_info = np.array(waveform_info)

#         if visualise:
#             # determine the number of channels
#             n_channels = waveforms.shape[0]

#             # calculate the size of the grid
#             grid_size = int(np.ceil(np.sqrt(n_channels)))

#             # create a figure and subplots
#             fig, axs = plt.subplots(grid_size, grid_size, figsize=(5, 5))

#             labels_provided = len(labels) > 0

#             if labels_provided and len(labels) == len(waveforms[0]):
#                 # get unique labels
#                 unique_labels = np.unique(labels)

#                 # create a colormap
#                 colormap = plt.cm.get_cmap('viridis', len(unique_labels))

#                 # create a dictionary mapping each label to a color
#                 label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
#                 print(label_to_color)

#             # make sure axs is always an array
#             if n_channels == 1:
#                 axs = np.array([[axs]])

#             # flatten the axes array
#             axs = axs.flatten()

#             # iterate over the channels
#             for channel_idx in range(n_channels):
#                 ax = axs[channel_idx]
#                 # iterate over the waveforms in this channel and plot each on top of each other
#                 for idx, waveform in enumerate(waveforms[channel_idx]):
#                     t = np.arange((- len(waveform) / 2), (len(waveform) / 2) ) * self.dt * 1000 
#                     if labels_provided and len(labels) == len(waveforms[0]):
#                         # get the color for this waveform based on its label
#                         color = label_to_color[labels[idx]]
#                     else:
#                         color = "black"

#                     # plot the waveform
#                     ax.plot(t, waveform, lw=.5, c=color)

#                 ax.set_xlabel('Time (ms)')
#                 ax.set_ylabel('Voltage (mV)')
#                 ax.set_title(f'Waveforms (Electrode {channel_idx})')

#             plt.tight_layout()
#             plt.show()


#         return waveforms, waveform_info
        
#     def get_spikes(self, signals, visualise=False, figsize: Tuple[int]=(8, 12)):

#         # initialise spikes as an empty list
#         spikes = []
#         # loop through each of the channels (electrodes)
#         for i, signal in enumerate(signals):
#             threshold = self.get_threshold_value(signal=signal, is_negative=False)

#             # detect spikes in this channel
#             spikes_channel = self.detect_spikes(signal, threshold, use_absolute_threshold=True)

#             # append the detected spikes to the list
#             spikes.append(spikes_channel)
        
#         if visualise:
#             self.show_spikes(signals, spikes, figsize)

#         return spikes
    
#     def show_spikes(self, signals, spikes, figsize):
#         # create a figure with one subplot for each electrode
#         fig, axs = plt.subplots(len(spikes), 1, figsize=figsize)

#         # handle the case where there's only one electrode
#         if len(spikes) == 1:
#             axs = [axs]

#         # plot the combined signals from the electrodes
#         for electrode_id, ax in enumerate(axs):

#             # get the combined signal
#             combined_signal = signals[electrode_id]

#             # plot the combined signal
#             time = np.arange(len(combined_signal)) / self.sample_rate
#             ax.plot(time, combined_signal, label=f'Electrode {electrode_id} Combined Signal', c="k", lw=0.5)

#             threshold = self.get_threshold_value(signal=combined_signal, is_negative=False)
#             ax.axhline(y=threshold, c="blue", linewidth=0.5, zorder=0, label=f"Threshold = {round(threshold, 3)} mV")

#             ax.scatter(time[spikes[electrode_id]], combined_signal[spikes[electrode_id]], c="red")

#             ax.set_title(f'Electrode {electrode_id} Recorded Signal')
#             ax.set_xlabel('Time')
#             ax.set_ylabel('Signal')
#             ax.legend()

#         # Adjust the layout and show the plot
#         plt.tight_layout()
#         plt.show()

#     def show_spike_train(self, waveforms, figsize: Tuple[int]=(8, 8)):
#         # determine the number of channels
#         n_channels =  waveforms.shape[0]

#         # create a figure and subplots
#         fig, axs = plt.subplots(n_channels, 2, figsize=figsize, sharex=True)

#         # make sure axs is always an array
#         if n_channels == 1:
#             axs = np.array([[axs]])

#         # iterate over the channels
#         for electrode_id in range(n_channels):
#             signal = grid.electrode_dict[electrode_id]["signal"]

#             time = np.arange(len(signal)) / grid.sample_rate

#             # plot the raw data for this channel
#             axs[electrode_id, 0].plot(time, np.array(signal), label='Raw Data', c='k', lw=0.5)

#             # iterate through each spike and plot its waveform for this channel
#             for spike in waveform_info[electrode_id]:
#                 spike_window = slice(spike["spike_start"], spike["spike_end"])
#                 axs[electrode_id, 1].plot(time[spike_window], signal[spike_window], c='k', lw=0.5)

#             # add x labels to bottom plots
#             if electrode_id == n_channels - 1:
#                 axs[electrode_id, 0].set_xlabel('Time (s)')
#                 axs[electrode_id, 1].set_xlabel('Time (s)')

#             axs[electrode_id, 0].set_ylim((min(signal) + (min(signal) * 0.1) , max(signal) + (max(signal) * 0.1)))
#             axs[electrode_id, 1].set_ylim((min(signal) + (min(signal) * 0.1) , max(signal) + (max(signal) * 0.1)))
#             axs[electrode_id, 0].set_xlim((0, max(time)))
#             axs[electrode_id, 1].set_xlim((0, max(time)))

#         fig.supylabel("Voltage")
#         fig.suptitle("Electrode Recording vs Spike Train")
#         plt.tight_layout()
#         plt.show()