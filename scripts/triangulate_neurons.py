import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

from classes.triangulator import Triangulator


def triangulate_neurons(signals, placements, labels, waveforms, decay_type):

    print("step 1")

    # create a dictionary for the electrodes and the true neurons
    electrode_dict, neuron_dict = create_placement_dicts(electrode_signals=signals, placements=placements)

    print("step 2")

    # add informtion about each electrode to the dictionary (e.g. )
    electrode_dict = determine_signal_strengths(labels=labels, waveforms=np.array(waveforms), electrode_dict=electrode_dict, signals=signals, visualise=False)

    print("step 3")

    triangulator = Triangulator(labels=labels, decay_type=decay_type, electrode_dict=electrode_dict, neuron_dict=neuron_dict, grid_size=int(np.sqrt(len(placements))))

    print("step 4")

    triangulator.triangulate()

    print("step 5")

    construction_dict = get_construction_data(electrode_dict, neuron_dict, triangulator)

    print("step 6")

    return construction_dict


def calculate_distance(neuron_dict, x, y):
    distances = []
    for neuron in neuron_dict.keys():
        neuron_x, neuron_y = neuron_dict[neuron]["row"], neuron_dict[neuron]["col"]
        distances.append(np.sqrt((x - neuron_x)**2 + (y - neuron_y)**2))
    return distances


def create_placement_dicts(electrode_signals, placements):
    # convert flat grid placements into a nested structure
    grid_size = int(np.sqrt(len(placements)))
    grid = np.reshape(placements, (grid_size, grid_size))

    neuron_dict = {}
    electrode_dict = {}

    # find row and column of each neuron
    neuron_num = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 1:
                neuron_dict[neuron_num] = {"row": i, "col": j}
                neuron_num += 1
    
    # find row and column of each electrode
    electrode_num = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 2:
                electrode_dict[electrode_num] = {"row": i, "col": j}
                electrode_dict[electrode_num]["signal"] = electrode_signals[electrode_num]
                electrode_dict[electrode_num]["distances"] = calculate_distance(neuron_dict=neuron_dict, x=i, y=j)
                electrode_num += 1

    return electrode_dict, neuron_dict


def determine_signal_strengths(labels, waveforms, electrode_dict, signals, visualise=False):    
    # get unique labels
    unique_labels = np.unique(labels)

    # loop through each electrode
    for i in range(waveforms.shape[0]):

        # initialise 'avg_waveform' and 'avg_waveform_peak' keys for this electrode
        electrode_dict[i]["avg_waveform"] = {}
        electrode_dict[i]["avg_waveform_peak"] = {}
        electrode_dict[i]["baseline_to_peak"] = {}

        # get the waveforms for this electrode
        electrode_waveforms = waveforms[i]


        # estimate the signal baseline using the mode
        mode_result = mode(electrode_dict[i]["signal"])

        if np.isscalar(mode_result.mode):
            signal_baseline = mode_result.mode
        else:
            signal_baseline = mode_result.mode[0]
        
        # loop through each unique label
        for label in unique_labels:
            
            # get the waveforms for this label at this electrode
            label_waveforms = electrode_waveforms[labels == label]
            
            # compute the average waveform for this label at this electrode
            average_waveform = np.mean(label_waveforms, axis=0)
            
            # store the average waveform info in the electrode_dict
            electrode_dict[i]["avg_waveform"][label] = average_waveform

            # store the difference between the average vaeform peak and the signals baseline
            electrode_dict[i]["avg_waveform_peak"][label] = np.max(average_waveform) 
            electrode_dict[i]["baseline_to_peak"][label] = abs(np.max(average_waveform) - signal_baseline) 

    # show a plot if requested
    if visualise:
        # get the number of unique labels and electrodes
        n_labels = len(unique_labels)
        n_electrodes = len(electrode_dict)

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
                average_waveform = electrode_dict[i]["avg_waveform"][label]
                
                # plot the average waveform
                axs[i, j].plot(average_waveform, c="k")
                
                # set the title of the subplot to the label
                axs[0, j].set_title(f'Label {label}')
                axs[i, 0].set_ylabel(f'Electrode {i}')
                
    return electrode_dict


def get_construction_data(electrode_dict, neuron_dict, triangulator):

    # initialise empty dictionary
    construction_dict = {}

    # get some of the information from the Triangulator object
    triangulated_info = triangulator.triangulate_info

    # get true neuron positions and add to dictionary
    construction_dict["true_neuron_positions"] = [(v["row"], v["col"]) for v in neuron_dict.values()]

    # get all electrode positions and add to dictionary
    construction_dict["all_electrode_positions"] = [(v["row"], v["col"]) for v in electrode_dict.values()]

    # get all predicted neuron positions and add to dictionary
    construction_dict["predicted_neuron_positions"] = [triangulated_info[i]["intersection_point"] for i in triangulated_info.keys()]

    print(triangulated_info.keys())

    # add info for each identified neuron to the dictionary
    for i in triangulated_info.keys():
        construction_dict[i] = {}

        try:
            construction_dict[i]["true_neuron_position"] = construction_dict["true_neuron_positions"][i]
        except:
            construction_dict[i]["true_neuron_position"] = (1, 1)
            
        construction_dict[i]["predicted_neuron_position"] = triangulated_info[i]["intersection_point"]
        construction_dict[i]["circles"] = triangulated_info[i]["circle_info"]
        construction_dict[i]["used_electrodes"] = triangulated_info[i]["electrodes_by_strength"][:3]
        construction_dict[i]["intersecting_lines"] = triangulated_info[i]["perpendicular_lines"]
    
    return construction_dict
