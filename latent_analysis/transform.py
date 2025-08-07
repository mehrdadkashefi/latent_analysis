"""
Collection of tools for linear dimensionality reduction methods of neural data
@Author: Mehrdad Kashefi
"""
import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd
from dPCA import dPCA as dpca
from scipy.optimize import minimize, NonlinearConstraint

class PCA():
    """ Simple Principal Component Analysis (PCA)

    Args:
        num_latent (int)
            number of latent dimensions
    """

    def __init__(self, num_latent):
        self.num_latent = num_latent

    def fit(self, X):
        """ Fit the model to the data

        Args: 
            X (np.array)
                data to fit the model (Samples x Units)
        """
        dims = X.shape
        if len(dims)>2:
            #print('Found tensor- Reshaping data to -1, num_unit')
            X = X.reshape(-1, dims[-1])

        self.method = 'PCA'
        self.mean_ = np.mean(X, axis=0)

        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt.T[:,:self.num_latent]
        self.variance_explained = ((S**2))/np.sum((S**2))
        self.varianve = (S**2)/(dims[-1]-1)

    def transform(self, X):
        """ Transform the data to the latent space

        Args:
            X (np.array)
                data to transform (Samples x Units)
        Returns:
            Latent (np.array)
                transformed data in the latent space (Samples x num_latent)
        """
        dims = X.shape
        if len(dims)>2:
            #print('Found tensor- Reshaping data to -1, num_unit')
            X = X.reshape(-1, dims[-1])
        # Perform Transform for each method
        if self.method == 'PCA':
            Latent = (X - self.mean_) @ self.components_
            Latent = np.reshape(Latent, newshape=dims[:-1] + (self.num_latent,))
        
        return Latent
    
class FA():
    """ Simple Factor Analysis (FA)

    Args:
        num_latent (int)
            number of latent dimensions
    """

    def __init__(self, num_latent):
        self.num_latent = num_latent

    def fit(self, X):
        """ Fit the model to the data

        Args: 
            X (np.array)
                data to fit the model (Samples x Units)
        """
        dims = X.shape
        if len(dims)>2:
            #print('Found tensor- Reshaping data to -1, num_unit')
            X = X.reshape(-1, dims[-1])

        self.method = 'FA'
        FA = FactorAnalysis(n_components=self.num_latent)
        FA.fit(X)
        self.components_ = FA.components_
        self.mean_ = FA.mean_
        self.FA = FA

    def transform(self, X, ensure_orthogonality):
        """ Transform the data to the latent space

        Args:
            X (np.array)
                data to transform (Samples x Units)
            ensure_orthogonality (bool)
                if True, the components are orthogonalized
        Returns:
            Latent (np.array)
                transformed data in the latent space (Samples x num_latent)
        """
        dims = X.shape
        if len(dims)>2:
            #print('Found tensor- Reshaping data to -1, num_unit')
            X = X.reshape(-1, dims[-1])
        # Perform Transform for each method
        if ensure_orthogonality: 
            U, _, _ = np.linalg.svd(self.components_.T)
            np.shape((X - self.mean_))
            Latent = (X - self.mean_) @ U[:self.num_latent,: ].T
            Latent = np.reshape(Latent, newshape=dims[:-1] + (self.num_latent,))
        else:
            Latent = self.FA.transform(X)
            Latent = np.reshape(Latent, newshape=dims[:-1] + (self.num_latent,))
        
        return Latent

class jPCA():
    """ Class for fitting and transforming data with jPCA

    Args:
        num_latent (int)
            number of latent dimensions, default 6
        force_skewness (bool)
            if True, the dynamical system is forced to be skew-symmetric, default True
        align_x_axis (bool)
            if True, the jPCA axes are aligned with the x-axis, default True
    """
    def __init__(self, **kwargs):
        self.num_comp_pc = kwargs.get('num_latent', 6)
        self.force_skewness = kwargs.get('force_skewness', True)
        self.aling_x_axis = kwargs.get('align_x_axis', True)
        

    def fit(self, X):
        """ Fit the jPCA model to the data

        Args:
            X (np.array)
                data to fit the model (Samples x Units)
        """
        # Normalize data, very important!
        n_mean = np.mean(X, axis=0, keepdims=True)
        n_range = np.max(np.vstack(X), axis=0, keepdims=True) - np.min(np.vstack(X), axis=0, keepdims=True)
        rate_scaled = (X - n_mean)/(n_range+5)

        transform = Transform(num_latent=self.num_comp_pc)
        transform.fit(rate_scaled, method='PCA')
        rate_red = transform.transform(rate_scaled, ensure_orthogonality=True)
        self.variance_explained_pca = np.round(sum(transform.variance_explained[0:self.num_comp_pc]), 3)
        print('Var explained by initial PCA {}'.format(np.round(sum(transform.variance_explained[0:self.num_comp_pc]), 3)))

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
        self.variance_explained_rot_fit = np.round(self.R2(X_dot, X@M), 3)
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
        """ Transform the data to the jPCA space

        Args:
            X (np.array)
                data to transform (Samples x Units)
        Returns:
            rate_jpca (np.array)
                transformed data in the jPCA space (Samples x num_latent)
        """
        # Normalize data, very important!
        n_mean = np.mean(X, axis=0, keepdims=True)
        n_range = np.max(np.vstack(X), axis=0, keepdims=True) - np.min(np.vstack(X), axis=0, keepdims=True)
        rate_scaled = (X - n_mean)/(n_range+5)

        # Initial PCA
        transform = Transform(num_latent=self.num_comp_pc)
        transform.fit(rate_scaled, method='PCA')
        rate_red = transform.transform(rate_scaled, ensure_orthogonality=True)
        rate_jpca = np.matmul(rate_red, self.jpca_w)
        self.variance_explained_jpcs = np.round(np.sum(np.var(rate_jpca, axis=0))/np.sum(np.var(rate_red, axis=0)), 3)
        print('Var explained by 2 jPCs {}'.format(np.round(np.sum(np.var(rate_jpca, axis=0))/np.sum(np.var(rate_red, axis=0)), 3)))

        ## Rotate axis so that planning is aligned with X axis
        if self.aling_x_axis:
            transform = Transform(num_latent=rate_jpca.shape[-1])
            transform.fit(rate_jpca[:, 0, :], method='PCA')
            rate_jpca = transform.transform(rate_jpca, ensure_orthogonality=True)
            # Make sure rotations are always CCW
            if np.cross(np.append(transform.components_[:,0], 0), np.append(transform.components_[:,1], 0))[-1]:
                rate_jpca = np.matmul(rate_jpca, np.array([[1,  0],[0 ,-1]]))
        return rate_jpca

    def R2(self, y_true, y_pred):
        return 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)    
    # Helper functions for getting the skew-symmetric matrix    
    def skew_sym_regress(self, X, X_dot, tol=1e-4):
        """ Fits a skew-symmetric matrix M to the data X_dot = X @ M.T

        Args:
            X (np.array)
                data (Samples x Units)
            X_dot (np.array)
                derivative of the data (Samples x Units)
            tol (float)
                tolerance for the optimization, default 1e-4
        Returns:
            M (np.array)
                skew-symmetric matrix
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
    """ Class for fitting and transforming data with dPCA, using dPCA toolbox

    Args:
        n_components (int)
            number of latent dimensions, default 5
        soft_norm_value (float)
            soft normalization value, default 5
        plot (bool)
            if True, plot the explained variance, default True
    """
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self.soft_norm_value = kwargs.get('soft_norm_value', 5)
        self.plot = kwargs.get('plot', True)
    
    def __pre_process(self, X):
        units_mean = np.mean(X, axis=(0,1), keepdims=True)
        n_range = np.max(np.vstack(X), axis=0, keepdims=True) - np.min(np.vstack(X), axis=0, keepdims=True)
        self.X_norm = (X - units_mean)/(n_range + self.soft_norm_value)

    def fit(self, X):
        """ Fit the dPCA model to the data

        Args:
            X (np.array)
                data to fit the model (Condition x Time x Units)
        """
        # Apply preprocessing
        self.__pre_process(X)
        # define the dPCA object
        self.dpca = dpca.dPCA(labels='ct', join={'c':['c'], 't':['t'], 'ct':['ct']}, n_components=self.n_components)
        self.dpca.protect = ['t']
        self.dpca.fit(self.X_norm.transpose(2,0,1))

    def plot_explained_variance(self):
        plt.figure()
        sns.barplot(self.var_explained_df, x='pc', y='variance_explained', hue='factor', palette=sns.color_palette('muted', 3))
        plt.ylabel('Ratio of explained variance')
        plt.show()

    def transform(self, X):
        """ Transform the data to the dPCA space

        Args:
            X (np.array)
                data to transform (Condition x Time x Units)
        Returns:
            Z (np.array)
                transformed data in the dPCA space (Condition x Time x Units
        """
        # Apply preprocessing
        self.__pre_process(X)  
        Z = self.dpca.transform(self.X_norm.transpose(2,0,1))

        self.var_explained_df = pd.DataFrame(columns=['factor', 'pc', 'variance_explained'])
        counter = 0
        for name in self.dpca.explained_variance_ratio_.keys():
            for c in range(self.dpca.n_components):
                self.var_explained_df.loc[counter] = {'factor': name, 'pc': c, 'variance_explained': self.dpca.explained_variance_ratio_[name][c]}
                counter += 1
                
        if self.plot:
            self.plot_explained_variance()
            

        return Z

class OrthogonalPCA():
    """ Class for fitting and transforming data with Orthogonal PCs for planning and execution

    Args:
        method (str) 
            method to use for optimization (maniopt or scipy_minimize), default maniopt
        n_components (int)
            number of latent dimensions, default 5
        soft_norm_value (float)
            soft normalization value, default 5
        verbose (int)
            verbosity level, default 1
    """
    def __init__(self, **kwargs):
        self.method = kwargs.get('method', 'maniopt')
        self.n_components = kwargs.get('n_components', 20)
        self.soft_norm_value = kwargs.get('soft_norm_value', 5)
        self.verbose = kwargs.get('verbose', 1)

    def pre_process(self, X):
        units_mean = np.mean(X, axis=0, keepdims=True)
        n_range = np.max(np.vstack(X), axis=0, keepdims=True) - np.min(np.vstack(X), axis=0, keepdims=True)
        return (X - units_mean)/(n_range + self.soft_norm_value)
    
    def fit(self, data_prep, data_exe):
        """ Fit the Orthogonal PCA model to the data
        
        Args:
            data_prep (np.array)
                data for planning (Samples x Units)
            data_exe (np.array)
                data for execution (Samples x Units)
        Returns:
            W_p (np.array)
                weights for planning
            W_e (np.array)
                weights for execution
        """
        # Preprocess (remove condition mean and soft scaling)
        data_prep = self.pre_process(data_prep)
        data_exe = self.pre_process(data_exe)

        # Estimate the covariance matrices
        N_p = np.vstack(data_prep).T
        N_e = np.vstack(data_exe).T
        num_units = N_p.shape[0]

        # Estimate the Covariance
        C_p = (1/(N_p.shape[1]-1))*((N_p - np.mean(N_p,axis=1, keepdims=True))@(N_p - np.mean(N_p,axis=1, keepdims=True)).T)
        C_e = (1/(N_e.shape[1]-1))*((N_e - np.mean(N_e,axis=1, keepdims=True))@(N_e - np.mean(N_e,axis=1, keepdims=True)).T)

        _, S_e, _ = np.linalg.svd(C_e)
        _, S_p, _ = np.linalg.svd(C_p)
        S_e = np.sum(S_e)
        S_p = np.sum(S_p)
        # Optimization
        if self.method == 'scipy_minimize':
            # Run the optimization for double PCAs
            num_dim = self.n_components
            # Initialize the weights with PCA on segments of data
            transform = Transform(num_latent=data_prep.shape[-1])
            transform.fit(data_exe, method='PCA')
            W_e = transform.components_[:,:num_dim]
            #W_e = np.random.randn(num_units, num_dim)


            transform = Transform(num_latent=data_prep.shape[-1])
            transform.fit(data_prep, method='PCA')
            W_p = transform.components_[:,:num_dim]
            #W_p = np.random.randn(num_units, num_dim)

            def  mycost(x):
                W_p = x[:int(len(x)/2)].reshape(num_units, num_dim) 
                W_e = x[int(len(x)/2):].reshape(num_units, num_dim)
                f = 0.5*((np.trace(W_p.T@C_p@W_p)/S_p ) + (np.trace(W_e.T@C_e@W_e)/S_e) )
                return -f

            x0 = np.concatenate((W_p.reshape(-1,), W_e.reshape(-1,)))


            con1 = lambda x: ((x[:int(len(x)/2)].reshape(num_units, num_dim)).T@ x[int(len(x)/2):].reshape(num_units, num_dim)).reshape(-1,)
            nlc1 = NonlinearConstraint(con1, np.zeros((num_dim,num_dim)).reshape(-1,) ,np.zeros((num_dim,num_dim)).reshape(-1,))

            con2 = lambda x: ((x[:int(len(x)/2)].reshape(num_units, num_dim)).T@ (x[:int(len(x)/2)].reshape(num_units, num_dim))).reshape(-1,)
            nlc2 = NonlinearConstraint(con2, np.eye((num_dim)).reshape(-1,) ,np.eye((num_dim)).reshape(-1,))

            con3 = lambda x: ((x[int(len(x)/2):].reshape(num_units, num_dim)).T@ (x[int(len(x)/2):].reshape(num_units, num_dim))).reshape(-1,)
            nlc3 = NonlinearConstraint(con3, np.eye((num_dim)).reshape(-1,) ,np.eye((num_dim)).reshape(-1,))


            res = minimize(mycost, x0,constraints=(nlc1, nlc2, nlc3), method='trust-constr',
                        options={'disp': True, 'maxiter':500, 'verbose':self.verbose})
            
            # Get the final weights
            x_opt = res.x
            W_p = x_opt[:int(len(x_opt)/2)].reshape(num_units, num_dim) 
            W_e = x_opt[int(len(x_opt)/2):].reshape(num_units, num_dim)
        elif self.method == 'maniopt':
            import autograd.numpy as np_ag
            import pymanopt
            from pymanopt.manifolds import Stiefel
            from pymanopt.optimizers import TrustRegions

            def create_cost_and_derivates(manifold, C_p, C_e, S_p, S_e):
                euclidean_gradient = euclidean_hessian = None
                @pymanopt.function.autograd(manifold)   # possible backends("autograd", "jax", "numpy", "pytorch", "tensorflow")
                def cost(w):
                    W_p = w[:, :self.n_components]
                    W_e = w[:, self.n_components:]
                    return -0.5*((np_ag.trace(W_p.T@C_p@W_p)/S_p ) + (np_ag.trace(W_e.T@C_e@W_e)/S_e))
                
                return cost, euclidean_gradient, euclidean_hessian
            
            manifold = Stiefel(num_units, self.n_components * 2)
            cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(manifold, C_p, C_e, S_p, S_e)

            problem = pymanopt.Problem(
                manifold,
                cost,
                euclidean_gradient=euclidean_gradient,
                euclidean_hessian=euclidean_hessian,
            )
            optimizer = TrustRegions(verbosity=self.verbose)
            w_opt = optimizer.run(problem).point
            W_p = w_opt[:, :self.n_components]
            W_e = w_opt[:, self.n_components:]

        # Plot variance explained by each dimension
        print('Planning: ', np.trace(W_p.T@C_p@W_p)/S_p)
        print('Execution: ', np.trace(W_e.T@C_e@W_e)/S_e) 
        ## Check condtions
        print('Normality of W_p:', np.isclose(W_p.T@W_p, np.eye(W_p.shape[1])).all())
        print('Normality of W_e:', np.isclose(W_e.T@W_e, np.eye(W_e.shape[1])).all())
        print('Orthogonality of W_e and W_p:', np.isclose(W_e.T@W_p, np.zeros(W_e.shape[1])).all())

        # Perform a final pca to sort the dimensions (This step does not change the amound of variance explained)
        # Rotate Ws to maximize variance explained by first dimension
        transform = Transform(num_latent=W_p.shape[-1])
        transform.fit(data_prep @ W_p, method='PCA')
        W_p = W_p @ transform.components_
        # Execution
        transform = Transform(num_latent=W_e.shape[-1])
        transform.fit(data_exe @ W_e, method='PCA')
        W_e = W_e @ transform.components_

        return W_p, W_e

class OrthogonalPCA3(OrthogonalPCA):
    """ Class for fitting and transforming data with Orthogonal PCs for planning and execution and a plan-execution subspace

    Args:
        n_components (int)
            number of latent dimensions, default 5
        soft_norm_value (float)
            soft normalization value, default 5
        verbose (int)
            verbosity level, default 1
    """

    def fit(self, data_prep, data_exe):
        """ Fit the Orthogonal PCA model to the data
        
        Args:
            data_prep (np.array)
                data for planning (Samples x Units)
            data_exe (np.array)
                data for execution (Samples x Units)
        Returns:
            W_p (np.array)
                weights for planning
            W_e (np.array)
                weights for execution
        """
        # Preprocess (remove condition mean and soft scaling)
        data_prep = self.pre_process(data_prep)
        data_exe = self.pre_process(data_exe)
        data_prepexe = np.concatenate((data_prep, data_exe), axis=1)
        data_prepexe = self.pre_process(data_prepexe)

        # Estimate the covariance matrices
        N_p = np.vstack(data_prep).T
        N_e = np.vstack(data_exe).T
        N_pe = np.vstack(data_prepexe).T
        num_units = N_p.shape[0]

        # Estimate the Covariance
        C_p = (1/(N_p.shape[1]-1))*((N_p - np.mean(N_p,axis=1, keepdims=True))@(N_p - np.mean(N_p,axis=1, keepdims=True)).T)
        C_e = (1/(N_e.shape[1]-1))*((N_e - np.mean(N_e,axis=1, keepdims=True))@(N_e - np.mean(N_e,axis=1, keepdims=True)).T)
        C_pe = (1/(N_pe.shape[1]-1))*((N_pe - np.mean(N_pe,axis=1, keepdims=True))@(N_pe - np.mean(N_pe,axis=1, keepdims=True)).T)

        _, S_e, _ = np.linalg.svd(C_e)
        _, S_p, _ = np.linalg.svd(C_p)
        _, S_pe, _ = np.linalg.svd(C_pe)
        S_e = np.sum(S_e)
        S_p = np.sum(S_p)
        S_pe = np.sum(S_pe)
   
        import autograd.numpy as np_ag
        import pymanopt
        from pymanopt.manifolds import Stiefel
        from pymanopt.optimizers import TrustRegions

        def create_cost_and_derivates(manifold, C_p, C_e,C_pe, S_p, S_e, S_pe):
            euclidean_gradient = euclidean_hessian = None
            @pymanopt.function.autograd(manifold)   # possible backends("autograd", "jax", "numpy", "pytorch", "tensorflow")
            def cost(w):
                W_p = w[:, :self.n_components]
                W_e = w[:, self.n_components:2*self.n_components]
                W_pe = w[:, 2*self.n_components:3*self.n_components]
                return -(1/3)*((np_ag.trace(W_p.T@C_p@W_p)/S_p ) + (np_ag.trace(W_e.T@C_e@W_e)/S_e) + (np_ag.trace(W_pe.T@C_pe@W_pe)/S_pe))
            
            return cost, euclidean_gradient, euclidean_hessian
        
        manifold = Stiefel(num_units, self.n_components * 3)
        cost, euclidean_gradient, euclidean_hessian = create_cost_and_derivates(manifold, C_p, C_e,C_pe, S_p, S_e, S_pe)

        problem = pymanopt.Problem(
            manifold,
            cost,
            euclidean_gradient=euclidean_gradient,
            euclidean_hessian=euclidean_hessian,
        )
        optimizer = TrustRegions(verbosity=self.verbose)
        w_opt = optimizer.run(problem).point
        W_p = w_opt[:, :self.n_components]
        W_e = w_opt[:, self.n_components:2*self.n_components]
        W_pe = w_opt[:, 2*self.n_components:3*self.n_components]

        # Plot variance explained by each dimension
        print('Planning: ', np.trace(W_p.T@C_p@W_p)/S_p)
        print('Execution: ', np.trace(W_e.T@C_e@W_e)/S_e) 
        print('Plan-Execution: ', np.trace(W_pe.T@C_e@W_pe)/S_e) 
        ## Check condtions
        print('Normality of W_p:', np.isclose(W_p.T@W_p, np.eye(W_p.shape[1])).all())
        print('Normality of W_e:', np.isclose(W_e.T@W_e, np.eye(W_e.shape[1])).all())
        print('Normality of W_pe:', np.isclose(W_pe.T@W_pe, np.eye(W_pe.shape[1])).all())

        print('Orthogonality of W_e and W_p:', np.isclose(W_e.T@W_p, np.zeros(W_e.shape[1])).all())
        print('Orthogonality of W_e and W_pe:', np.isclose(W_e.T@W_pe, np.zeros(W_e.shape[1])).all())
        print('Orthogonality of W_p and W_pe:', np.isclose(W_p.T@W_pe, np.zeros(W_p.shape[1])).all())

        # Perform a final pca to sort the dimensions (This step does not change the amound of variance explained)
        # Rotate Ws to maximize variance explained by first dimension
        transform = Transform(num_latent=W_p.shape[-1])
        transform.fit(data_prep @ W_p, method='PCA')
        W_p = W_p @ transform.components_
        # Execution
        transform = Transform(num_latent=W_e.shape[-1])
        transform.fit(data_exe @ W_e, method='PCA')
        W_e = W_e @ transform.components_
        # Plan-Execution
        transform = Transform(num_latent=W_pe.shape[-1])
        transform.fit(data_exe @ W_pe, method='PCA')
        W_pe = W_pe @ transform.components_

        return W_p, W_e, W_pe