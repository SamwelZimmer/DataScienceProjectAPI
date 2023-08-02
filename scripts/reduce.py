from classes.reducer import PCA_Reducer, UMAP_Reducer, TSNE_Reducer

def dimensional_reduction(model, n_components, waveforms):

    # initialise the relevant dimensionality reduction model
    reducer = intialise_model(model, n_components)

    # standardise the waveforms 
    waveforms_scaled = reducer.scaler(waveforms)

    # crush the waveforms in n components (only pca have variance ratios)
    if model == "pca":
        b, explained_variance_ratios = reducer.reduce(waveforms=waveforms_scaled)
    else:
        b = reducer.reduce(waveforms=waveforms_scaled)

    # convert data to a usable format for the frontend
    reduced_data = {"quantities": { "n_components": n_components, "n_channels": len(b), "n_spikes": len(b[0]) }}
    
    for electrode in range(len(b)):
        reduced_data[f"channel_{electrode + 1}"] = {"points": {}}

        points = b[electrode]

        for i in range(n_components):
            reduced_data[f"channel_{electrode + 1}"]["points"][f"component_{i + 1}"] = points[: , i].tolist()

    
    return reduced_data


def intialise_model(model: str, n_components: int):
    if model == "pca":
        reducer = PCA_Reducer(n_components=n_components)
    elif model == "umap":
        reducer = UMAP_Reducer(n_components=n_components)
    elif model == "tsne":
        reducer = TSNE_Reducer(n_components=n_components)

    # add other models if neccessary
    return reducer

