import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd
from dPCA import dPCA as dpca

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
            


class jPCA():
    def __init__(self, **kwargs):
        self.num_comp_pc = kwargs.get('num_latent', 6)
        self.force_skewness = kwargs.get('force_skewness', True)
        self.aling_x_axis = kwargs.get('align_x_axis', True)
        

    def fit(self, X):
        # Normalize data, very important!
        n_mean = np.mean(X, axis=0, keepdims=True)
        n_range = np.max(np.vstack(X), axis=0, keepdims=True) - np.min(np.vstack(X), axis=0, keepdims=True)
        rate_scaled = (X - n_mean)/(n_range+5)

        transform = Transform(num_latent=self.num_comp_pc)
        transform.fit(rate_scaled, method='PCA')
        rate_red = transform.transform(rate_scaled, ensure_orthogonality=True)

        # Fit the dynamicals system to data
        A_red_n = rate_red[:, 1:, :]
        A_red_n_1 = np.diff(rate_red, axis=1)
        # Reshape to fit least squares
        X = np.reshape(A_red_n, (-1, A_red_n.shape[2]))
        X_dot = np.reshape(A_red_n_1, (-1, A_red_n_1.shape[2]))

        # Fit M with no constraints
        if self.force_skewness:
            M =  self.skew_sym_regress(X, X_dot)
        else:
            M = np.linalg.lstsq(X, X_dot, rcond=None)[0]

        print('R2 for linear fit: {}'.format(np.round(self.R2(X_dot, X@M), 3)))
        # Get eigenvalues and eigenvectors of the dynamical system
        L, V = np.linalg.eig(M)
        # Remove any small imaginary components
        L = np.imag(L)
        if L[0]<0:
            v1 = V[:, 0:1] + V[:, 1:2]
            v2 = 1j*(V[:, 0:1] - V[:, 1:2])
        else:
            v1 = V[:, 1:2] + V[:, 0:1]
            v2 = 1j*(V[:, 1:2] - V[:, 0:1])
        # Get the jPCA projections
        # v1 and v2 are real but the complex component is still here, hence np.real
        self.jpca_w = np.real(np.concatenate((v1, v2), axis=1))/np.sqrt(2)

        
    def transform(self, X):
        # Normalize data, very important!
        n_mean = np.mean(X, axis=0, keepdims=True)
        n_range = np.max(np.vstack(X), axis=0, keepdims=True) - np.min(np.vstack(X), axis=0, keepdims=True)
        rate_scaled = (X - n_mean)/(n_range+5)

        # Initial PCA
        transform = Transform(num_latent=self.num_comp_pc)
        transform.fit(rate_scaled, method='PCA')
        rate_red = transform.transform(rate_scaled, ensure_orthogonality=True)
        print('Var explained by initial PCA {}'.format(np.round(sum(transform.variance_explained[0:self.num_comp_pc]), 3)))
        rate_jpca = np.matmul(rate_red, self.jpca_w)
        rate_jpca = np.matmul(rate_red, self.jpca_w/np.linalg.norm(self.jpca_w))
        print('Var explained by 2 jPCs {}'.format(np.round(np.sum(np.var(rate_jpca, axis=0))/np.sum(np.var(rate_red, axis=0)), 3)))
        

        ## Rotate axis so that planning is aligned with X axis
        if self.aling_x_axis:
            transform = Transform(num_latent=rate_jpca.shape[-1])
            transform.fit(rate_jpca[:, 0, :], method='PCA')
            rate_jpca = transform.transform(rate_jpca, ensure_orthogonality=True)
        return rate_jpca

    def R2(self, y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)    
    # Helper functions for getting the skew-symmetric matrix    
    def skew_sym_regress(self, X, X_dot, tol=1e-4):
        """
        Fits a skew-symmetric matrix M to the data X_dot = X @ M.T
        """
        def _objective(h, X, X_dot):
            _, N = X.shape
            M = _reshape_vec2mat(h, N)
            return 0.5 * np.linalg.norm(X @ M.T - X_dot, ord='fro')**2


        def _reshape_mat2vec(M, N):
            upper_tri_indices = np.triu_indices(N, k=1)
            return M[upper_tri_indices]
        
        def _reshape_vec2mat(h, N):
            M = np.zeros((N, N))
            upper_tri_indices = np.triu_indices(N, k=1)
            M[upper_tri_indices] = h
            return M - M.T

        def _grad_f(h, X, X_dot):
            _, N = X.shape
            M = _reshape_vec2mat(h, N)
            dM = (X.T @ X @ M.T) - X.T @ X_dot
            return _reshape_mat2vec(dM.T - dM, N)

        # Initialize with least squares
        T, N = X.shape
        M_lstq, _, _, _ = np.linalg.lstsq(X, X_dot, rcond=None)
        M_lstq = M_lstq.T
        M_init = 0.5 * (M_lstq - M_lstq.T)
        h_init = _reshape_mat2vec(M_init, N)

        options=dict(maxiter=10000, gtol=tol)
        result = minimize(lambda h: _objective(h, X, X_dot),
                            h_init,
                            jac=lambda h: _grad_f(h, X, X_dot),
                            method='CG',
                            options=options)
        if not result.success:
            print("Optimization failed.")
            print(result.message)
        M = _reshape_vec2mat(result.x, N)
        assert(np.allclose(M, -M.T))
        return M.T
    
class dPCA():
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self.soft_norm_value = kwargs.get('soft_norm_value', 5)
        self.pie_plot = kwargs.get('pie_plot', True)
    
    def __pre_process(self, X):
        units_mean = np.mean(X, axis=(0,1), keepdims=True)
        n_range = np.max(np.vstack(X), axis=0, keepdims=True) - np.min(np.vstack(X), axis=0, keepdims=True)
        self.X_norm = (X - units_mean)/(n_range + self.soft_norm_value)

    def fit(self, X):
        # Apply preprocessing
        self.__pre_process(X)
        # define the dPCA object
        self.dpca = dpca.dPCA(labels='ct', join={'c':['c'], 't':['t'], 'ct':['ct']}, n_components=self.n_components)
        self.dpca.protect = ['t']
        self.dpca.fit(self.X_norm.transpose(2,0,1))

    def transform(self, X):
        # Apply preprocessing
        self.__pre_process(X)  
        Z = self.dpca.transform(self.X_norm.transpose(2,0,1))

        if self.pie_plot:
            df = pd.DataFrame(columns=['factor', 'pc', 'variance_explained'])
            counter = 0
            for name in self.dpca.explained_variance_ratio_.keys():
                for c in range(self.dpca.n_components):
                    df.loc[counter] = {'factor': name, 'pc': c, 'variance_explained': self.dpca.explained_variance_ratio_[name][c]}
                    counter += 1

            sns.barplot(df, x='pc', y='variance_explained', hue='factor', palette=sns.color_palette('muted', 3))
            plt.ylabel('Ratio of explained variance')

        return Z

