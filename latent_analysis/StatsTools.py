"""
Collection of tools for statistical analysis of neural data
@Author: Mehrdad Kashefi
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocess
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.signal import savgol_filter
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from tqdm import tqdm
from PcmPy.regression import RidgeDiag
import mat73 as mat73

class VarDecompose():
    """ 
    An anlysis tool for decomposing the variance of a set of hidden variables into different components.
    The components are defined by a set of indicator variables.
    The hidden data (Y) should be in the shape of (num_conditions, num_timepoints, num_hidden_variables)
    The indicator variables should be in the shape of (num_conditions, 1), similar conditions will have the same values
    
    Args:
        Indicators (dict)
            A dictionary of indicator variables 
        ortho_ineraction (bool)
            If True, the interaction terms will be orthogonalized
        verbose (bool)
            If True, the covariance matrices will be plotted
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
        """ Fit the model to the data

        Args:
            Y (np.array)
                The hidden data (num_conditions, num_timepoints, num_hidden_variables)
                
        Returns:
            tss (np.array)
                Total variance of hidden variables
            fss (np.array)
                Explained variance by each model
        """
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
        """ Plot the results

        Args:
            width (int)
                Width of the figure, default is 15
            height (int)
                Height of the figure, default is 5
            save_dir (str)
                Directory to save the figures
            name (str)
                Name of the figures
        """
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
    
class TimePointClassifier():
    """ 
    An anlysis tool for classification of experimental conditions from continuous variables like position, velocity, average FR, etc.
    The continuous data (X) should be in the shape of (num_conditions, num_timepoints, num_variables)
    The associated class value (y) (num_conditions, )

    Example:
        TClassifier = ST.TimepointClassifier()
        acc, acc_chance =  TClassifier.fit(X, y) 

    Args:
        num_fold (int)
            Number of folds for cross-validation, default is 5
        num_core (int)
            Number of cores for parallel processing, default is 10
        num_sampling_rep (int)
            Number of sampling repetitions, default is 30
    """
    def __init__(self, num_fold = 5, num_core = 10, num_sampling_rep = 30):
        self.num_fold = num_fold
        self.num_sampling_rep = num_sampling_rep
        self.num_core = num_core
  
    def fit(self, X, y):
        """ Run the classification models on every timepoint

        Args:
            X (np.array)
                The continuous data (num_conditions, num_timepoints, num_variables)
            y (np.array)
                The associated class value (num_conditions, )
        Returns:
            acc (np.array)
                Accuracy of the model
            acc_chance (np.array)
                Chance level accuracy
        """
        self.X = X
        self.y = y
        
        if self.num_core ==1:
            acc = np.zeros((X.shape[1],2))
            for t in tqdm(range(X.shape[1])):
                acc[t, :] = self.par_function_classification(t)
        else:
            acc = np.zeros((X.shape[1],))
            with multiprocess.Pool(processes=self.num_core) as pool:
                acc  = pool.map(self.par_function_classification, range(X.shape[1]))
            acc = np.array(acc) 
            
        self.acc_chance = acc[:, 1]
        self.acc = acc[:, 0]
        return self.acc, self.acc_chance
    
    def plot(self, winseize = 40):
        plt.plot(savgol_filter(self.acc, window_length=winseize, polyorder=3), color='k')
        plt.plot(self.acc, alpha=0.4 , color='k')
        plt.plot(savgol_filter(self.acc_chance, window_length=winseize, polyorder=3), color='r', linewidth=.5)
        plt.plot(self.acc_chance, alpha=0.4, color='r', linewidth=.5)
        plt.xlabel('Time (ms)')
        plt.ylabel('Accuracy')
        plt.legend(['Accuracy', 'Chance'])

    def par_function_classification(self, t):
        X = self.X[:, t, :]
        y = self.y
        acc_temp = []
        acc_rand_temp = []
        for _ in range(self.num_sampling_rep):
            acc_temp.append(self.classifier(X,y, randomize_y = False))
            acc_rand_temp.append(self.classifier(X,y, randomize_y = True))
        return np.mean(acc_temp), np.mean(acc_rand_temp)
    
    def classifier(self, X,y, randomize_y = False):
        if randomize_y:
            rand_idx = np.random.permutation(len(y))
            y = y[rand_idx]
        kf = KFold(n_splits= self.num_fold, shuffle=True)
        score = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = RidgeClassifierCV().fit(X_train, y_train)
            score.append(clf.score(X_test, y_test))
        score = np.mean(score)
        return score
    
# Runs a family of models on every timepoint
class Model():
    """ RUN RMD-like models on each time point

    Args:
        name (str)
            Name of the model
        M (np.array)
            The model matrix (num_samples, num_features)
        fit_intercept (bool)
            If True, the model will fit an intercept
    Kwargs:
        feature_indicator (np.array)
            A binary array indicating the features that should be included in the model
    """
    def __init__(self,name, M, fit_intercept, **kwargs):
        self.name = name
        self.M = M
        self.num_sample = M.shape[0]
        self.num_feature = M.shape[1]
        self.fit_intercept = fit_intercept
        self.feature_indicator = kwargs.get('feature_indicator', np.zeros((M.shape[1],),dtype = np.int8))
        print("Model ", self.name, "Shape of M: ",M.shape)


    def fit(self, Y, method, **kwargs):
        """ Fit the model to the data on each timepoint

        Args:
            Y (np.array)
                The hidden data (num_conditions, num_timepoints, num_hidden_variables)
            method (str)
                The method of fitting the model
        Kwargs:
            n_kfold (int)
                Number of folds for cross-validation, default is 4
            unit_eval (bool)
                If True, the evaluation will be done on each unit, default is False
            n_kfold_in (int)
                Number of folds for inner cross-validation, default is 2
            lambda_list (list)
                List of regularization parameters, default is [1e-2,1e-1,1,1e1,1e2, 1e3]
            fit_score (str)
                The score for fitting the model, default is 'r'
        """
        self.method = method
        self.n_kfold = kwargs.get('n_kfold',4)
        self.unit_eval = kwargs.get('unit_eval', False)
        if self.method == 'RidgeCV':
            self.n_kfold_in = kwargs.get('n_kfold_in',2)
            self.lambda_list = kwargs.get('lambda_list', [1e-2,1e-1,1,1e1,1e2, 1e3])
            self.fit_score = kwargs.get('fit_score', 'r')
            # Select the scorer for inter kfold
            if self.fit_score == 'r':
                scorer = make_scorer(r)
            elif self.fit_score == 'r2':
                scorer = make_scorer(R2)
            else:
                print("Enter a valid score name (r or r2)")

        num_time_sample = Y.shape[1]
        self.num_units = Y.shape[-1]
        # Zeros for validation results
        if self.unit_eval:
            self.r = np.zeros((self.n_kfold, num_time_sample, self.num_units))
            self.R2 = np.zeros((self.n_kfold, num_time_sample, self.num_units))
        else:
            self.r = np.zeros((self.n_kfold, num_time_sample, 1))
            self.R2 = np.zeros((self.n_kfold, num_time_sample, 1))

        # Fit the model
        # Fit with Cross-validated ridge
        if self.method == 'RidgeCV':
            for T in tqdm(range(num_time_sample)):
                YY = np.squeeze(Y[:, T, :])
                # k_fold cv
                kf = KFold(n_splits=self.n_kfold, shuffle=True)
                fold_count = 0
                for train_index, test_index in kf.split(self.M):
                    X_train, X_test = self.M[train_index], self.M[test_index]
                    y_train, y_test = YY[train_index], YY[test_index]

                    ridge = RidgeCV(alphas=self.lambda_list, fit_intercept= self.fit_intercept, cv=self.n_kfold_in, scoring = scorer)
                    ridge.fit(X_train, y_train)
                    y_pred = ridge.predict(X_test)
                    self.r[fold_count, T, :] =  r(y_test, y_pred, unit_eval = self.unit_eval)
                    self.R2[fold_count, T, :] =  R2(y_test, y_pred, unit_eval = self.unit_eval)
                    fold_count += 1

        # Fit with PCM
        if self.method == 'PCM':
            for T in tqdm(range(num_time_sample)):
                YY = np.squeeze(Y[:, T, :])
                # k_fold cv
                kf = KFold(n_splits=self.n_kfold, shuffle=True)
                fold_count = 0
                for train_index, test_index in kf.split(self.M):
                    X_train, X_test = self.M[train_index], self.M[test_index]
                    y_train, y_test = YY[train_index], YY[test_index]
                    PCMReg = RidgeDiag(self.feature_indicator, fit_intercept = self.fit_intercept)
                    # Find optimal regularizatoni parameter
                    PCMReg.optimize_regularization(X_train, y_train)
                    PCMReg.theta = PCMReg.theta_ # This will be deprecated after fixing the bug in PCM toolbox
                    PCMReg.fit(X_train, y_train, X=None)
                    if self.fit_intercept:
                        y_pred = PCMReg.predict(X_test) + PCMReg.beta_
                    else:
                        y_pred = PCMReg.predict(X_test)
                    self.r[fold_count, T, :] =  r(y_test, y_pred, unit_eval = self.unit_eval)
                    self.R2[fold_count, T, :] =  R2(y_test, y_pred, unit_eval = self.unit_eval)
                    fold_count += 1
        self.r = np.mean(self.r, axis=0)
        self.R2 = np.mean(self.R2, axis=0)

# Check this R2 value (subtract the mean?)
def R2(Y_true,  Y_pred, **kwargs):
    unit_eval = kwargs.get('unit_eval', False)
    if unit_eval:
        SSR = np.sum((Y_true - Y_pred)**2, axis=0)
        SST = np.sum(Y_true**2, axis=0)
    else:
        SSR = np.sum((Y_true - Y_pred)**2)
        SST = np.sum(Y_true**2)
    return 1 - (SSR/SST)



def r(Y_true,  Y_pred, **kwargs):
    unit_eval = kwargs.get('unit_eval', False)
    if unit_eval:
        r_val = np.sum(Y_true * Y_pred, axis=0)/np.sqrt( np.sum( Y_true**2, axis=0) * np.sum( Y_pred**2, axis=0))
    else:
        r_val = np.sum(Y_true * Y_pred)/np.sqrt( np.sum( Y_true**2) * np.sum( Y_pred**2) )
    return r_val