import numpy as np
from classes.cluster import MOG, KNN
from helpers.constants import RANDOM_STATE

def get_clusters(cluster_type, k_type, k, reduced_data):

    # conver the data from the frontend format to a format suitable for the backend
    b = revert_to_original_format(reduced_data)
    x = np.array(b)
    x = np.transpose(x, (1, 0, 2)).reshape((-1, x.shape[0] * x.shape[2]))
    
    # initialise clustering model
    if cluster_type == "gmm":
        cluster = MOG(x, random_state=RANDOM_STATE)
    elif cluster_type == "knn":
        cluster = KNN(x, random_state=RANDOM_STATE)
    # room to use more clustering algorithms
    else:
        cluster = None

    # find 'k' depending on the user's choice
    if k_type == "manual":
        chosen_k = k
    elif k_type == "bic":
        k_optimal, BIC, LL = cluster.find_optimal_k(K=[1, 2, 3, 4, 5, 6, 7, 8, 9], visualise=False)
        chosen_k = k_optimal
    else:
        chosen_k = 1
 
    ind, m, S, p = cluster.cluster(k=chosen_k)

    return ind.tolist()


def revert_to_original_format(reduced_data):
    n_components = reduced_data["quantities"]["n_components"]
    n_channels = reduced_data["quantities"]["n_channels"]
    n_spikes = reduced_data["quantities"]["n_spikes"]

    b = []

    for channel in range(1, n_channels+1):
        channel_data = reduced_data[f"channel_{channel}"]["points"]
        channel_array = []

        for component in range(1, n_components+1):
            component_data = channel_data[f"component_{component}"]
            channel_array.append(component_data)

        # convert list of component data to a numpy array and transpose it
        channel_array = np.array(channel_array).T

        b.append(channel_array)

    return b

