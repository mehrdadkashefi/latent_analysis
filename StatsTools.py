import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

class VarDecompose():
    """ 
    An anlysis tool for decomposing the variance of a set of hidden variables into different components.
    The components are defined by a set of indicator variables.
    The hidden data (Y) should be in the shape of (num_conditions, num_timepoints, num_hidden_variables)
    The indicator variables should be in the shape of (num_conditions, 1), similar conditions will have the same values
    Example:
        Indicators = {'H': home_idx, 'T': target_idx, 'H:T': np.hstack((home_idx, target_idx))}
        VarDec = ST.VarDecompose(Indicators, ortho_ineraction=True, verbose=1)
        tss, fss =  VarDec.fit(Y) 
        VarDec.plot(width=4, height=3, save_dir=current_script_path)
    """
    def __init__(self, Indicators, ortho_ineraction = 1,  verbose=1):
        # Define model covariance G_m
        self.G_m = self.G_maker()
        for key in Indicators.keys():
            D = self.IndicatorMatrix('identity', Indicators[key])
            if ortho_ineraction == 1:
                if ':' in key:
                    print('Performing orthogonalization on interaction term')
                    X = np.c_[self.IndicatorMatrix('identity', Indicators[key.split(':')[0]]),self.IndicatorMatrix('identity', Indicators[key.split(':')[1]])]
                    D = D - X @ np.linalg.pinv(X) @ D
                
            K = Indicators[key].shape[0]
            H = np.eye(K) - np.ones((K, K)) / K
            g = H@D@D.T@H.T
            self.G_m.add(g/np.trace(g), key)

        if verbose:
            self.G_m.plot()

    def fit(self, Y):
        self.num_cond = Y.shape[0]
        X = np.zeros((self.num_cond**2, self.G_m.num_models))
        for i, g in enumerate(self.G_m.G):
            X[:, i] = g.flatten()

        self.beta = np.zeros((self.G_m.num_models, Y.shape[1])) 
        self.tss = np.zeros((Y.shape[1], 1))
        self.fss = np.zeros((self.G_m.num_models, Y.shape[1]))

        for t in range(Y.shape[1]):
            y = Y[:, t, :]
            # Get empirical G
            K = y.shape[0]
            H = np.eye(K) - np.ones((K, K)) / K
            G_emp = H@y@y.T@H.T
            # Do regression
            self.beta[:, t] = nnls(X, G_emp.reshape(-1,))[0]
            self.tss[t] = np.trace(G_emp)  # Total variance of Hiddens
            for f, gm in enumerate(self.G_m.G):
                self.fss[f, t] = np.trace(gm)*self.beta[f, t]
        return self.tss, self.fss
    
    def plot(self, **kwargs):
        width = kwargs.get('width', 15)
        height = kwargs.get('height', 5)
        save_dir = kwargs.get('save_dir', None)
        name = kwargs.get('name', '')
        # Plot the results
        plt.figure(figsize=(width, height))
        plt.plot(self.tss, color='r')
        plt.plot(np.sum(self.fss, axis=0), color='k', linestyle='--')
        plt.legend(['TSS','FSS'])
        plt.ylabel('Var (a.u.)')
        plt.xticks([])
        if save_dir is not None:
            plt.savefig(save_dir + name + 'VarTot.pdf')
        plt.figure(figsize=(width, height))
        plt.plot(self.fss.T)
        plt.ylabel('Var exp (a.u.)')
        plt.xticks([])
        plt.legend(self.G_m.name)
        if save_dir is not None:
            plt.savefig(save_dir + name + 'VarExp.pdf')
        plt.figure(figsize=(width, height))
        plt.plot(self.fss.T/(np.sum(self.fss, axis=0, keepdims=True).T + np.finfo(float).eps))
        plt.ylabel('Var exp Norm')
        if save_dir is not None:
            plt.savefig(save_dir + name + 'VarExpNorm.pdf')
    # A class for organizing Gs    
    class G_maker():
        def __init__(self):
            self.G = []
            self.name = []
            self.num_models = 0
        def add(self, G, name):
            self.G.append(G)
            self.name.append(name)
            self.num_models += 1

        def get_names(self):
            print('#Gs: ',len(self.name))
            for n in self.name:
                print(n)
        def plot(self, **kwargs):
            save_dir = kwargs.get('save_dir', None)
            name = kwargs.get('name', '')
            fig, ax = plt.subplots(1, len(self.G))
            for i, g in enumerate(self.G):
                ax[i].imshow(g)
                ax[i].set_title(self.name[i])
                ax[i].set_xticks([])
                ax[i].set_yticks([])
            if save_dir is not None:
                plt.savefig(save_dir + name +  'Gs.pdf')
    # helper function for making one-hot vectors from categorical data
    def IndicatorMatrix(self, what, vec):
        transp=0
        if vec.shape[0]== 1:
            vec = np.transpose(vec)
            transp =  1
        (row,col) = vec.shape
        unique_array, unique_idx = np.unique(vec, return_inverse=True, axis=0)
        K = len(unique_array) # Number of classes
        # What to do:
        if what == 'identity':  # Dummy coding
            X = np.zeros((row,K))
            for i in range(K):
                X[unique_idx==i,i] = 1
        return X