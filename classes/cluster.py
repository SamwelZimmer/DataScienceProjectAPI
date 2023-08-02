import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture 


class Clustering:
    def __init__(self, x: np.ndarray, random_state: bool=None):
        self.x = x
        self.random_state = random_state

    def compute_log_likelihood(self, x, m, S, p):
        """
        Compute the log-likelihood of the data under the model.

        Parameters:
            x (np.array): N by D
            m (np.array): k by D
            S (np.array): D by D by k
            p (np.array): k by 1

        Returns:
            LL (float): Log-likelihood
        """
        k = m.shape[0]
        n = x.shape[0]
        LL = 0
        for i in range(n):
            temp = 0
            for j in range(k):
                try:
                    temp += p[j] * multivariate_normal(m[j], S[j]).pdf(x[i])
                except:
                    print(f"Covariance matrix for component {j} is not positive definite:")
                    print(S[j])
                    return
            LL += np.log(temp)
        return LL

    def bic(self, m, S, p):
        """
        Compute the BIC for a fitted model bic, LL = mog_bic(x,k) computes the the Bayesian Information 
        Criterion value and the log-likelihood of the fitted model.

        Parameters:
            x (np.array): N by D
            m (np.array): k by D
            S (np.array): D by D by k
            p (np.array): k

        Returns:
            bic (float): BIC
            LL (float): Log-likelihood
        """
        n, d = self.x.shape
        k = m.shape[0]

        # Compute the log-likelihood
        LL = self.compute_log_likelihood(self.x, m, S, p)

        # Compute the number of parameters
        P = k * (d + (d * (d + 1) / 2) + 1)

        # Compute the BIC
        bic = -2 * LL + P * np.log(n)
        
        return bic, LL

    def find_optimal_k(self, K, visualise=False, figsize=(10, 6)):
        BIC = np.zeros((len(K)))
        LL = np.zeros((len(K)))

        # loop over the number of clusters
        for i, k in enumerate(K):
            # fit the GMM to the data
            ind, m, S, p = self.cluster(k)

            # compute the BIC and log-likelihood
            bic, ll = self.bic(m, S, p)

            # store the results
            BIC[i] = bic
            LL[i] = ll

        optimal_k = K[np.argmin(BIC)]
        
        # plot the BIC as a function of the number of clusters
        if visualise:
            plt.figure(figsize=figsize)
            plt.plot(K, BIC, marker='o', c="k")
            plt.title('BIC as a function of the number of clusters')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('BIC')
            plt.grid(True)
            plt.show()

        return optimal_k, BIC, LL

    def assess_performance(self, predicted_labels, true_labels):
        if len(predicted_labels) == len(true_labels):
            accuracy = accuracy_score(true_labels, true_labels)
            precision = precision_score(true_labels, true_labels, average="weighted")
            recall = recall_score(true_labels, true_labels, average="weighted")
            f1 = f1_score(true_labels, true_labels, average="weighted")

            return accuracy, precision, recall, f1
        else:
            print("The number of predicted labels and true labels are not the same")
            return None, None, None, None
        
    def cluster():
        pass

class MOG(Clustering):
    def __init__(self, x: np.ndarray, random_state: bool=None):
        super().__init__(x, random_state)

    def cluster(self, k: int) -> Tuple[np.ndarray]:
        """
        Fit Mixture of Gaussian model
        ind, m, S, p = mog(x, k) fits a Mixture of Gaussian model to the data
        in x using k components. The output ind contains the MAP assignments of the
        datapoints in x to the found clusters. The outputs m, S, p contain
        the model parameters.

        Parameters
        ----------
        x: np.array 
            The datapoints -> N (Number of datapoints) x D (dimensionality of the data)
        k: int
            Number of clusters

        Returns
        ----------
        ind: np.array 
            Cluster indicators
        m: np.array
            Cluster means    (k x D)
        S: np.array 
            Cluster covarience matricies    (D x D x k)
        p: np.array 
            Cluster prior probabilities    (k x 1)
        """
        
        # fit the model
        gmm = GaussianMixture(n_components=k, random_state=self.random_state)
        gmm.fit(self.x)

        # get the MAP assignments of the data points
        ind = gmm.predict(self.x)

        # get the model parameters
        m = gmm.means_
        S = gmm.covariances_
        p = gmm.weights_

        return (ind, m, S, p)


class KNN(Clustering):
    def __init__(self, x):
        super().__init__(x)
    
    def cluster():
        pass