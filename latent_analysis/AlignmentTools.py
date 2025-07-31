"""
Collection of tools for alignment of neural data
@Author: Mehrdad Kashefi
"""
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import KFold
from latent_analysis.latent_analysis.utils import get_condition_mean

class CCA_svd():
    """Canonical Correlation Analysis using Singular Value Decomposition (SVD)"""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        Q_x, R_x = np.linalg.qr(X)
        Q_y, R_y = np.linalg.qr(Y)

        U, S, Vt = np.linalg.svd(Q_x.T @ Q_y)

        assert len(S) >= self.n_components, "n_components must be less than or equal to the min of number of features in X and Y"
        self._cc = S
        self.cc = S[:self.n_components]

        self.Wx = np.linalg.pinv(R_x) @ U
        self.Wy = np.linalg.pinv(R_y) @ Vt.T

    def transform(self, X, Y):
        X_c = X @ self.Wx[:, :self.n_components]
        Y_c = Y @ self.Wy[:, :self.n_components]
        return X_c, Y_c
        
    def score(self, X, Y):
        # project X into Y and get R2
        X_c = X @ self.Wx[:, :self.n_components]
        Y_c = Y @ self.Wy[:, :self.n_components]

        sse = np.sum((X_c - Y_c) ** 2)
        sst = np.sum((X_c - np.mean(X_c, axis=0)) ** 2)
        return 1 - sse / sst

class Procrustes():
    """
    Procrustes analysis (Based on Matlab implementation)
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """
    def __init__(self, scaling=True, reflection='best'):
        self.scaling = scaling
        self.reflection = reflection
    
    def fit(self, X, Y):
        n_sample, n_chan_X = X.shape
        _ , n_chan_Y = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0**2.).sum()
        ssY = (Y0**2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if n_chan_Y < n_chan_X:
            Y0 = np.concatenate((Y0, np.zeros((n_sample, n_chan_X-n_chan_Y))),axis=1)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U,s,Vt = np.linalg.svd(A,full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if self.reflection != 'best':

            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

            # if that's not what was specified, force another reflection
            if self.reflection != have_reflection:
                V[:,-1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if self.scaling:

            # optimum scaling of Y
            b = traceTA * normX / normY

            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA**2

            # transformed coords
            Z = normX*traceTA*np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY/ssX - 2 * traceTA * normY / normX
            Z = normY*np.dot(Y0, T) + muX

        # translation matrix
        if n_chan_Y < n_chan_X:
            T = T[:n_chan_Y,:]
        c = muX - b*np.dot(muY, T)
        
        #transformation values 
        self.tform = {'rotation':T, 'scale':b, 'translation':c}
    
        return d, Z, self.tform
    
    def transform(self, Y):
        """
        Transform the Y matrix using the transformation matrix.
        """
        Z = np.dot(Y, self.tform['rotation']) * self.tform['scale'] + self.tform['translation']
        return Z
    
    def score(self, X, Y):
        """
        Compute the score of the transformation.
        """
        Z = self.transform(Y) 
        return 1 - np.sum((X - Z ) ** 2) / ((X - X.mean(0))**2).sum()



def collapse_cond_time(data):
    """
    Collapse the condition and time dimensions of the data.
    Args:
        data: numpy array of shape (n_cond, n_time, n_chan)
    Returns:
        numpy array of shape (n_time * n_cond, n_chan)
    """
    n_cond, n_time, n_chan = data.shape
    return data.transpose(2, 0, 1).reshape(n_chan, n_time * n_cond).T


def get_dissimilarity_cond_avr(X, X_conds, Y, Y_conds, n_folds = 2, n_times = 1, method = 'CCA_svd'):
    """
    Compute the dissimilarity between two datasets by first averaging across trial, and using cross-validation.
    Args:
        X: numpy array of shape (n_trials, n_time, n_chan)
        X_conds: An array of length n_trials containing the condition id for each trial in X
        Y: numpy array of shape (n_trials, n_time, n_chan)
        Y_conds: An array of length n_trials containing the condition id for each trial in Y
        n_folds: number of folds for cross-validation
        n_times: number of times to repeat the analysis
        method: method to use for dissimilarity analysis ('CCA_svd', 'CCA_sklearn', 'Procrustes')
    Returns:
        df_score: pandas DataFrame containing the scores for each fold and time
        df: pandas DataFrame containing extra information for CCA methods
    """

    rows = []   
    rows_score = []

    data = {
        'X': X,
        'Y': Y,
    }
    conds = {
        'X': X_conds,
        'Y': Y_conds,
    }
    for t in tqdm(range(n_times)):
        fold_train = {}
        fold_test = {}
        # Get the cross-validation indices first.
        for name in ['X', 'Y']:
            train_idx = []
            test_idx = []

            kf = KFold(n_splits=n_folds, shuffle=True)
            for fold_i, (train_index, test_index) in enumerate(kf.split(data[name])):
                train_idx.append(train_index)
                test_idx.append(test_index)
            
            fold_train[name] = train_idx
            fold_test[name] = test_idx  

        # Run the dissimilarity analysis with the cross-validation indices    
        for fold in range(n_folds):
            #print('Time %d' % (t + 1), 'Fold %d' % (fold + 1),)
            # Fit CCA on the training data
            d_X_train = collapse_cond_time(
            get_condition_mean(data['X'][fold_train['X'][fold], : , :], 
                            conds['X'][fold_train['X'][fold]])
            )
            
            d_Y_train = collapse_cond_time(
                get_condition_mean(data['Y'][fold_train['Y'][fold], : , :], 
                                conds['Y'][fold_train['Y'][fold]])
            )

            # Test data
            d_X_test = collapse_cond_time(
            get_condition_mean(data['X'][fold_test['X'][fold], : , :], 
                            conds['X'][fold_test['X'][fold]])
            )
            
            d_Y_test = collapse_cond_time(
                get_condition_mean(data['Y'][fold_test['Y'][fold], : , :], 
                                conds['Y'][fold_test['Y'][fold]])
            )

            if 'CCA' in method:
                CCA_n_components = 10
                # Fit CCA
                #print('Fitting CCA')
                t_start = time.time()
                if method == 'CCA_sklearn':
                    from sklearn.cross_decomposition import CCA
                    CCA_max_iter = 1000
                    cca = CCA(n_components= CCA_n_components, max_iter= CCA_max_iter)
                elif method == 'CCA_svd':
                    cca = CCA_svd(n_components= CCA_n_components)
                
                cca.fit(d_X_train, d_Y_train)
                t_end = time.time()
                #print('Fitting CCA done!', 'Time taken: %.2f' % (t_end - t_start), 's')

                # Transform the test data
                X_c, Y_c = cca.transform(d_X_test, d_Y_test)
                # Save scores
                rows_score.append({
                    'Fold': fold,
                    'Time': t,
                    'Score': cca.score(d_X_test, d_Y_test)
                })
                # Compute the dissimilarity
                for cc in range(X_c.shape[1]):
                    rows.append({
                        'Fold': fold,
                        'Time': t,
                        'CC': cc,
                        'Corr': np.corrcoef(X_c[:, cc], Y_c[:, cc])[0, 1]
                    })

            elif 'Procrustes' == method:
                
                procrustes = Procrustes(scaling=True, reflection='best')
                d, Z, tform = procrustes.fit(d_X_train, d_Y_train)
                #Z_test = procrustes.transform(d_Y_test)
                score = procrustes.score(d_X_test, d_Y_test)
                rows_score.append({
                    'Fold': fold,
                    'Time': t,
                    'Score': score
                })

            df = pd.DataFrame(rows)
            df_score = pd.DataFrame(rows_score)
    return df_score, df 