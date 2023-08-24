import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

class Transform():
    def __init__(self, num_latent):
        self.num_latent = num_latent

    def fit(self, X, method):
        dims = X.shape
        if len(dims)>2:
            print('Found tensor- Reshaping data to -1, num_unit')
            X = X.reshape(-1, dims[-1])

        if method == 'PCA':
            self.method = 'PCA'
            self.mean_ = np.mean(X, axis=0)

            _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
            self.components_ = Vt.T[:,:self.num_latent]
            self.variance_explained = ((S**2))/np.sum((S**2))
            self.varianve = (S**2)/(dims[-1]-1)

        elif method == 'FA':
            self.method = 'FA'
            FA = FactorAnalysis(n_components=self.num_latent)
            FA.fit(X)
            self.components_ = FA.components_
            self.mean_ = FA.mean_
            self.FA = FA


    def transform(self, X, ensure_orthogonality):
        dims = X.shape
        if len(dims)>2:
            print('Found tensor- Reshaping data to -1, num_unit')
            X = X.reshape(-1, dims[-1])
        # Perform Transform for each method
        if self.method == 'PCA':
            Latent = (X - self.mean_) @ self.components_
            Latent = np.reshape(Latent, newshape=dims[:-1] + (self.num_latent,))
        
        elif self.method == 'FA':
            if ensure_orthogonality: 
                U, _, _ = np.linalg.svd(self.components_.T)
                np.shape((X - self.mean_))
                Latent = (X - self.mean_) @ U[:self.num_latent,: ].T
                Latent = np.reshape(Latent, newshape=dims[:-1] + (self.num_latent,))
            else:
                Latent = self.FA.transform(X)
                Latent = np.reshape(Latent, newshape=dims[:-1] + (self.num_latent,))



        return Latent
        

    def plot_traj(self, X_latent, which_trials, which_times, dim = 2, color_map = None, hue = None):

        if color_map is None:
            cm = plt.get_cmap('inferno')       
        else:
            cm = color_map

        if hue is None:
            color_list = ['k' for _ in range(X_latent.shape[0])]
        else:
            num_group  = len(np.unique(hue))   
            color = cm(np.linspace(0,0.85,num_group))
            color_list = [color[hue[tr]] for tr in range(X_latent.shape[0])]

        if dim == 2:
            for tr in which_trials:
                plt.plot(X_latent[tr,:,0].T,X_latent[tr,:,1].T, color=color_list[tr])
                plt.scatter(X_latent[tr,which_times,0],X_latent[tr,which_times,1], color=color_list[tr])
        elif dim == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            for tr in which_trials:
                ax.plot3D(X_latent[tr,:,0],X_latent[tr,:,1],X_latent[tr,:,2], color=color_list[tr])
                ax.scatter3D(X_latent[tr,which_times,0],X_latent[tr,which_times,1],X_latent[tr,which_times,2], color=color_list[tr])
        else:
            print("Dimension must be 2 or 3!")
            