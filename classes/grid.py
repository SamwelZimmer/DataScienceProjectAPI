import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
from scipy.stats import mode

from classes.neuron import EulerLIF, RKLIF

class Grid:
    def __init__(self, size:int=9, decay_rate:float=1.0, signal_length:int=1, sample_rate:int=25000, neuron_type:str="standard", decay_type:str="square", lmbda: int=14, v_rest: int=-70, v_thres: int=-10, t_ref: float=0.02, noise_type: str="gaussian", noise_std: float=0.5, random_state:bool=None) -> None:
        self.signal_length = signal_length
        self.sample_rate = sample_rate
        self.step_size = 1 / sample_rate
        self.neuron_type = neuron_type
        self.random_state = random_state
        self.decay_type = decay_type
        self.width = size
        self.height = size
        self.decay_rate = decay_rate
        self.neurons_dict = {}
        self.electrode_dict = {}
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.lmbda = lmbda
        self.v_rest = v_rest
        self.v_thres = v_thres
        self.t_ref = t_ref
        self.noise_type = noise_type
        self.noise_std = noise_std

        # random state doesn't yet do anything
        if self.random_state is not None:
            random.seed(self.random_state)

    def __select_neuron_type(self): 
        neuron_types_dict = {
            "standard": RKLIF(v_rest=self.v_rest, sample_rate=self.sample_rate, thr=self.v_thres, tf=self.signal_length, t_ref=self.t_ref, lmbda=self.lmbda, seed=self.random_state),
            "euler": EulerLIF(v_rest=self.v_rest, sample_rate=self.sample_rate, thr=self.v_thres, tf=self.signal_length, t_ref=self.t_ref, lmbda=self.lmbda, seed=self.random_state)
        }
        
        if self.neuron_type not in neuron_types_dict.keys():
            raise ValueError("Neuron Type Not Recognised")
        
        return neuron_types_dict[self.neuron_type]

    def __place_neuron(self, row, col) -> None:
        self.grid[row][col] = "Neuron"

    def __place_electrode(self, empty_grid: List[List], row, col) -> Tuple[int]:
        if empty_grid[row][col] == "Neuron" or empty_grid[row][col] == "Electrode":
            self.__place_electrode(empty_grid)
        else:
            empty_grid[row][col] = "Electrode"
            return row, col
        
    def __calculate_distances(self, x, y):
        """Distance to Neuron"""
        distances = []
        for neuron in self.neurons_dict.keys():
            neuron_x, neuron_y = self.neurons_dict[neuron]["row"], self.neurons_dict[neuron]["col"]
            distances.append(np.sqrt((x - neuron_x)**2 + (y - neuron_y)**2))
        return distances
        
    def decay_mulitplier(self, distance: float) -> int:
        k = self.decay_rate
        if self.decay_type == "linear":
            return max(0, k - distance)
        elif self.decay_type == "exponential":
            return np.exp(-k * distance)
        elif self.decay_type == "square":
            return k /( distance ** 2)
        elif self.decay_type == "inverse":
            if distance != 0:
                return k / distance
            else:
                return np.inf
        else:
            raise ValueError(f"Unknown decay type: {type}")
        
    def noise(self, signal, noise_stddev: int=0.5, type: str="gaussian"):
        if type == "gaussian":
            return np.random.normal(0, noise_stddev, signal.shape)
        elif type == "none":
            return np.random.normal(0, 0, signal.shape)
        else:
            print(f"Unknown Noise Type: {type}")

    def add_neurons(self, positions: List[Tuple]=False):
        if positions:
            for i, position in enumerate(positions):
                try:
                    row, col = position[0], position[1]
                    self.__place_neuron(row, col)
                    self.neurons_dict[i] = {"row": row, "col": col, "object": self.__select_neuron_type()}
                except:
                    print("Something went wrong placing your neurons")
        else:
            row, col = self.width // 2, self.height // 2
            self.__place_neuron(row, col)
            self.neurons_dict[0] = {"row": row, "col": col, "object": self.__select_neuron_type()}

    def add_electrodes(self, positions: List[Tuple]=False):
        if positions:
            for i, position in enumerate(positions):
                try:
                    row, col = position[0], position[1]

                    # if something already exist at this grid position, give a warning
                    if self.grid[row][col]:
                        print(f"WARNING... There is already a {self.grid[row][col]} at ({row},{col})" )

                    self.__place_electrode(self.grid, row, col)
                    self.electrode_dict[i] = { "row": row, "col": col, "distances": self.__calculate_distances(row, col), "signals": [] }
                except:
                    print("Something went wrong placing your neurons")
        else:
            for i in range(3):
                row, col = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
                self.electrode_dict[i] = { "row": row, "col": col, "distances": self.__calculate_distances(row, col), "signals": [] }
                self.__place_electrode(self.grid, row, col)

    def generate_signals(self):
        if len(self.neurons_dict.keys()) == 0 or len(self.electrode_dict.keys()) == 0:
            raise ValueError("You need both Neurons and Electrodes to record signals")
        
        # stimulate each of the neurons to produce a signal
        for neuron in self.neurons_dict.keys():
            neuron_signal = self.neurons_dict[neuron]["object"].simulate()
            self.neurons_dict[neuron]["signal"] = neuron_signal

            self.neurons_dict[neuron]["spikes"] = self.neurons_dict[neuron]["object"].identify_neuron_spikes(neuron_signal)

            # each electrode records this signal with its intensity depending on the distance from the neuron
            for electrode in self.electrode_dict.keys():
                distance = self.electrode_dict[electrode]["distances"][neuron]
                decay = self.decay_mulitplier(distance=distance)
                # print("Neuron:", neuron , "Electrode:", electrode, "Distance:", distance)
                self.electrode_dict[electrode]["signals"].append((neuron_signal * decay) + self.noise(neuron_signal, noise_stddev=self.noise_std, type=self.noise_type))
        
        # combine the signals of all of the neurons and combine them (only neccesary if more than one neuron exists)
        if len(list(self.neurons_dict.keys())) > 0:
            for electrode in self.electrode_dict.keys():
                combined_signal = np.zeros_like(self.electrode_dict[electrode]["signals"][0])
                for neuron in self.neurons_dict.keys():
                    combined_signal += self.electrode_dict[electrode]["signals"][neuron]

                self.electrode_dict[electrode]["signal"] = combined_signal

    def plot_grid(self, figsize: Tuple[int]=(6,6)) -> None:
        plt.figure(figsize=figsize)
        
        # extract neuron and electrode positions
        neuron_positions = [(v["row"], v["col"]) for v in self.neurons_dict.values()]
        electrode_positions = [(v["row"], v["col"]) for v in self.electrode_dict.values()]

        # if there are neurons, plot them
        if neuron_positions:
            neuron_rows, neuron_cols = zip(*neuron_positions) # unzip into x and y coordinates
            plt.scatter(neuron_rows, neuron_cols, color='r', label='Neurons', s=100)

            # add text labels for neurons
            for i, (x, y) in enumerate(neuron_positions):
                plt.text(x, y, f'{i}', fontsize=20)

        # if there are electrodes, plot them
        if electrode_positions:
            electrode_rows, electrode_cols = zip(*electrode_positions) # unzip into x and y coordinates
            plt.scatter(electrode_rows, electrode_cols, color='b', label='Electrodes', s=100)
            # add text labels for electrodes
            for i, (x, y) in enumerate(electrode_positions):
                plt.text(x, y, f'{i}', fontsize=20)

        plt.xlim(0, self.width - 1)
        plt.ylim(0, self.height - 1)
        plt.grid(True)
        plt.legend()
        plt.title("Grid with Neurons and Electrodes")
        plt.show()

    def determine_signal_strengths(self, labels, waveforms, visualise=False):

        # get unique labels
        unique_labels = np.unique(labels)

        # loop through each electrode
        for i in range(waveforms.shape[0]):

            # initialise 'avg_waveform' and 'avg_waveform_peak' keys for this electrode
            self.electrode_dict[i]["avg_waveform"] = {}
            self.electrode_dict[i]["avg_waveform_peak"] = {}
            self.electrode_dict[i]["baseline_to_peak"] = {}

            # get the waveforms for this electrode
            electrode_waveforms = waveforms[i]

            
            # estimate the signal baseline using the mode
            signal_baseline = mode(self.electrode_dict[i]["signal"])[0][0]
            
            # loop through each unique label
            for label in unique_labels:
                
                # get the waveforms for this label at this electrode
                label_waveforms = electrode_waveforms[labels == label]
                
                # compute the average waveform for this label at this electrode
                average_waveform = np.mean(label_waveforms, axis=0)
                
                # store the average waveform info in the electrode_dict
                self.electrode_dict[i]["avg_waveform"][label] = average_waveform

                # store the difference between the average vaeform peak and the signals baseline
                self.electrode_dict[i]["avg_waveform_peak"][label] = np.max(average_waveform) 
                self.electrode_dict[i]["baseline_to_peak"][label] = abs(np.max(average_waveform) - signal_baseline) 

        # show a plot if requested
        if visualise:
            # get the number of unique labels and electrodes
            n_labels = len(unique_labels)
            n_electrodes = len(self.electrode_dict)

            # create a figure with a subplot for each electrode
            fig, axs = plt.subplots(n_electrodes, n_labels, figsize=(n_labels*2, n_electrodes*2))

            # make sure axs is always a 2D array, even when n_electrodes or n_labels is 1
            if n_electrodes == 1:
                axs = axs[np.newaxis, :]
            if n_labels == 1:
                axs = axs[:, np.newaxis]

            # loop through each electrode and each label
            for i in range(n_electrodes):
                for j in range(n_labels):
                    # get the label and average waveform for this subplot
                    label = unique_labels[j]
                    average_waveform = self.electrode_dict[i]["avg_waveform"][label]
                    
                    # plot the average waveform
                    axs[i, j].plot(average_waveform, c="k")
                    
                    # set the title of the subplot to the label
                    axs[0, j].set_title(f'Label {label}')
                    axs[i, 0].set_ylabel(f'Electrode {i}')


            # adjust the layout so that subplots do not overlap
            plt.suptitle("Average Waveform for each Neuron \nIdentified by Each Electrode")
            plt.tight_layout()
            plt.show()

    def show_raw_signals(self, combined: bool = True, figsize: Tuple[int] = (12, 12)):
        neuron_ids, electrode_ids = list(self.neurons_dict.keys()), list(self.electrode_dict.keys())

        if combined:
            fig, axs = plt.subplots(len(electrode_ids), 1, figsize=figsize)

            # handle the case where there's only one electrode
            if len(electrode_ids) == 1:
                axs = [axs]

            # plot the combined signals from the electrodes
            for electrode_id, ax in enumerate(axs):
                
                # get the combined signal
                combined_signal = self.electrode_dict[electrode_id]["signal"]

                # plot the combined signal
                time = np.arange(len(combined_signal)) / grid.sample_rate
                ax.plot(time, combined_signal, label=f'Electrode {electrode_id} Combined Signal', c="k", lw=0.5)
                ax.set_title(f'Electrode {electrode_id} Combined Signal')
                ax.set_xlabel('Time')
                ax.set_ylabel('Signal')
                ax.legend()

        else:
            # if neuron_ids is a single value, convert it to a list
            if not isinstance(neuron_ids, list):
                neuron_ids = [neuron_ids]

            fig, axs = plt.subplots(len(electrode_ids) + 1, len(neuron_ids), figsize=figsize)

            # handle the case where there's only one neuron and multiple electrodes
            if len(neuron_ids) == 1 and len(electrode_ids) > 1:
                axs = np.expand_dims(axs, axis=1)

            # plot the original signals from the neurons
            for ax, neuron_id in zip(axs[0], neuron_ids):
                neuron_signal = self.neurons_dict[neuron_id]["signal"]
                spike_locations = self.neurons_dict[neuron_id]["spikes"]
                time = np.arange(len(neuron_signal)) / self.sample_rate
                ax.plot(time, neuron_signal, c="k", lw=0.5)
                ax.set_title(f'Signal Emitted by Neuron {neuron_id}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Signal')

            # plot the recorded signals from the electrodes
            for i, electrode_id in enumerate(electrode_ids, start=1):
                for j, ax in enumerate(axs[i]):
                    electrode_signal = self.electrode_dict[electrode_id]["signals"][neuron_ids[j]]
                    time = np.arange(len(neuron_signal)) / self.sample_rate
                    ax.plot(time, electrode_signal, c="k", lw=0.5)
                    spike_locations = self.neurons_dict[j]["spikes"]
                    ax.set_title(f"Electrode {electrode_id} Recording Neuron {j}'s Signal")
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Voltage (mV)')

        # adjust the layout and show the plot
        plt.tight_layout()
        plt.show()
