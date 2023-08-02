from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap.umap_ import UMAP


class Reducer:
    def __init__(self) -> None:
        self.values = []

    def scaler(self, waveforms):
        scaler = StandardScaler()
        return np.array([scaler.fit_transform(waveform) for waveform in waveforms])

    def reduce():
        pass

    def show_dimensional_reduction():
        pass


class PCA_Reducer(Reducer):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    def reduce(self, waveforms: np.ndarray) -> Tuple:
        self.explained_variance_ratios = []
        for i in range(waveforms.shape[0]):
            waveform = waveforms[i]

            # initialise PCA
            pca = PCA(n_components=self.n_components)

            self.values.append(pca.fit_transform(waveform))

            self.explained_variance_ratios.append(pca.explained_variance_ratio_)
        
        return self.values, self.explained_variance_ratios
    
    def show_dimensional_reduction(self, labels=[], figsize=(5, 5)):
        nrows = 2
        ncols = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        is_labelled = len(labels) > 0

        for i, (data, evr) in enumerate(zip(self.values, self.explained_variance_ratios)):
            row = i // ncols
            col = i % ncols
            x = [point[0] for point in data]
            y = [point[1] for point in data]

            if is_labelled:
                unique_labels = np.unique(labels)
                colormap = plt.cm.get_cmap('viridis', len(unique_labels))
                label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
                colors = [label_to_color[label] for label in labels]

            axs[row, col].set_title(f"Channel {i + 1} PCA")
            axs[row, col].set_xlabel(f"P1 ({round(evr[0] * 100, 2)}%)")
            axs[row, col].set_ylabel(f"P2 ({round(evr[1] * 100, 2)}%)")

            if is_labelled:
                axs[row, col].scatter(x, y, s=10, c=colors)
            else:
                axs[row, col].scatter(x, y, s=10, c="k")

        plt.suptitle("PCA Dimensionality Reduction")
        plt.tight_layout()
        plt.show()

    def show_first_principle_component(self, labels=[], figsize=(8, 8)):
        # number of channels
        num_channels = len(self.values)

        is_labelled = len(labels) > 0

        # create subplots
        fig, axs = plt.subplots(num_channels, num_channels, figsize=figsize)

        # iterate through each pair of channels
        for i in range(num_channels):
            for j in range(num_channels):
                
                if is_labelled:
                    unique_labels = np.unique(labels)
                    colormap = plt.cm.get_cmap('viridis', len(unique_labels))
                    label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
                    colors = [label_to_color[label] for label in labels]

                # for each subplot, plot the first principal component of channel i against channel j
                if is_labelled:
                    axs[i, j].scatter(b[i][:, 0], self.values[j][:, 0], c=colors, s=10)
                else:
                    axs[i, j].scatter(b[i][:, 0], self.values[j][:, 0], c="k", s=10)
                axs[i, j].set_xlabel(f"Electrode {i}")
                axs[i, j].set_ylabel(f"Electrode {j}")

        plt.suptitle("First Principal Component")
        plt.tight_layout()
        plt.show()


class UMAP_Reducer(Reducer):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    def reduce(self, waveforms: np.ndarray):
        for i in range(waveforms.shape[0]):
            waveform = waveforms[i]

            # initialise UMAP
            umap = UMAP(n_components=self.n_components)
            self.values.append(umap.fit_transform(waveform))

        return self.values
    
    def show_dimensional_reduction(self, labels=[], figsize=(5, 5)):
        nrows = 2
        ncols = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        is_labelled = len(labels) > 0

        for i, data in enumerate(self.values):
            row = i // ncols
            col = i % ncols
            x = [point[0] for point in data]
            y = [point[1] for point in data]

            if is_labelled:
                unique_labels = np.unique(labels)
                colormap = plt.cm.get_cmap('viridis', len(unique_labels))
                label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
                colors = [label_to_color[label] for label in labels]

            axs[row, col].set_title(f"Channel {i + 1} UMAP")
            axs[row, col].set_xlabel(f"P1")
            axs[row, col].set_ylabel(f"P2")

            if is_labelled:
                axs[row, col].scatter(x, y, s=10, c=colors)
            else:
                axs[row, col].scatter(x, y, s=10, c="k")

        plt.suptitle("UMAP Dimensionality Reduction")
        plt.tight_layout()
        plt.show()


class TSNE_Reducer(Reducer):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    def reduce(self, waveforms: np.ndarray):
        for i in range(waveforms.shape[0]):
            waveform = waveforms[i]

            # initialise t-SNE
            tsne = TSNE(n_components=self.n_components)
            self.values.append(tsne.fit_transform(waveform))

        return self.values
    
    def show_dimensional_reduction(self, labels=[], figsize=(5, 5)):
        nrows = 2
        ncols = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        is_labelled = len(labels) > 0

        for i, data in enumerate(self.values):
            row = i // ncols
            col = i % ncols
            x = [point[0] for point in data]
            y = [point[1] for point in data]

            if is_labelled:
                unique_labels = np.unique(labels)
                colormap = plt.cm.get_cmap('viridis', len(unique_labels))
                label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}
                colors = [label_to_color[label] for label in labels]

            axs[row, col].set_title(f"Channel {i + 1} t-SNE")
            axs[row, col].set_xlabel(f"P1")
            axs[row, col].set_ylabel(f"P2")

            if is_labelled:
                axs[row, col].scatter(x, y, s=10, c=colors)
            else:
                axs[row, col].scatter(x, y, s=10, c="k")

        plt.suptitle("t-SNE Dimensionality Reduction")
        plt.tight_layout()
        plt.show()