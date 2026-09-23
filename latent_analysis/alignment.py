"""
Collection of tools for alignment of neural data
@Author: Mehrdad Kashefi
"""
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.model_selection import KFold
from latent_analysis.utils import get_condition_mean, collapse_cond_time
from scipy.linalg import eigh

class CCA_svd():
    """ Canonical Correlation Analysis using Singular Value Decomposition (SVD)

    Args:
        n_components (int)
            number of CC components, default 2
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, Y):
        """ Fit the CCA model to the data

        Args:
            X (np.array)
                data to fit the model (Samples x Units)
            Y (np.array)
                data to fit the model (Samples x Units)
        """
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
        """ Transforms new data in fitted CCA space

        Args:
            X (np.array)
                data to fit the model (Samples x Units)
            Y (np.array)
                data to fit the model (Samples x Units)
        """
        X_c = X @ self.Wx[:, :self.n_components]
        Y_c = Y @ self.Wy[:, :self.n_components]
        return X_c, Y_c
        
    def score(self, X, Y):
        """ Transforms new data in fitted CCA space and computes the R2 score

        Args:
            X (np.array)
                data to fit the model (Samples x Units)
            Y (np.array)
                data to fit the model (Samples x Units)
        """
        X_c, Y_c = self.transform(X, Y)
        # Compute the R2 score
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

    Args:
        scaling (bool, default True)
            Controls whether the solution includes a scaling component.
        reflection (str, default 'best')
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.
    """
    def __init__(self, scaling=True, reflection='best'):
        self.scaling = scaling
        self.reflection = reflection
    
    def fit(self, X, Y):
        """ Fit the Procrustes model to the data

        Args:
            X (np.array)
                data to fit the model (Samples x Units)
            Y (np.array)
                data to fit the model (Samples x Units)
        Returns:
            d (float)
                the residual sum of squared errors, normalized according to a
                measure of the scale of X, ((X - X.mean(0))**2).sum()
            Z (np.array)
                the matrix of transformed Y-values
            tform (dict)
                a dict specifying the rotation, translation and scaling that
                maps X --> Y
        """
        self.X = X
        self.Y = Y

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

        Args:
            Y (np.array)
                data to transform (Samples x Units)
        Returns:
            Z (np.array)
                the matrix of transformed Y-values
        """
        Z = np.dot(Y, self.tform['rotation']) * self.tform['scale'] + self.tform['translation']
        return Z
    
    def score(self, X, Y):
        """
        Compute the score of the transformation.
        
        Args:
            X (np.array)
                data to transform (Samples x Units)
            Y (np.array)
                data to transform (Samples x Units)
        Returns:
            score (float)
                the R2 score of aligned data.
        """
        Z = self.transform(Y) 
        return 1 - np.sum((X - Z ) ** 2) / ((X - X.mean(0))**2).sum()


def get_dissimilarity_cond_avr(X, X_conds, Y, Y_conds, n_folds = 2, n_times = 1, method = 'CCA_svd'):
    """
    Compute the dissimilarity between two datasets by first averaging across trial, and using cross-validation.
    
    Args:
        X (np.array)
            First dataset (n_trials, n_time, n_chan)
        X_conds (np.array)
            An array of length n_trials containing the condition id for each trial in X
        Y (np.array)
            Second dataset (n_trials, n_time, n_chan)
        Y_conds (np.array)
            An array of length n_trials containing the condition id for each trial in Y
        n_folds (int)
            Number of folds for cross-validation
        n_times (int)
            Number of times to repeat the cross-validation
        method (str)
            Method to use for dissimilarity analysis ('CCA_svd', 'CCA_sklearn', 'Procrustes'). Default is 'CCA_svd'.
    Returns:
        df_score (pd.DataFrame)
            DataFrame containing the scores for each fold and time
        df (pd.DataFrame)
            DataFrame containing extra information for CCA methods
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


def _flat(X):
    """(trials, time, units) -> (trials*time, units).

    Identical to X.transpose(2, 0, 1).reshape(X.shape[-1], -1).T but a
    zero-copy view when X is C-contiguous instead of two full copies.
    """
    return X.reshape(-1, X.shape[-1])


def top_pcs(X, k):
    """Orthonormal basis (n_units, k) for the top-k PC subspace of X.

    X is (n_samples, n_units) and is mean-centred internally, exactly like
    sklearn's PCA. Only the *subspace* matters downstream (trace(U' C U) is
    invariant to rotations/sign flips within it), so we can use whichever
    decomposition is cheapest.
    """
    X0 = X - X.mean(axis=0)
    n, u = X0.shape
    if u <= n:                                   # usual case: units << samples
        S = X0.T @ X0                            # (u, u)
        _, V = eigh(S, subset_by_index=(u - k, u - 1))
    else:                                        # wide case: use the Gram trick
        G = X0 @ X0.T                            # (n, n)
        _, W = eigh(G, subset_by_index=(n - k, n - 1))
        V = X0.T @ W
        V /= np.linalg.norm(V, axis=0, keepdims=True)
    return np.ascontiguousarray(V)


# ----------------------------------------------------------------------
# single alignment index
# ----------------------------------------------------------------------
def alignment_index(A, B, n_dim=10, pca=top_pcs):
    """Variance of A captured by B's top-n_dim PCs, relative to A's own."""
    Amat, Bmat = _flat(A), _flat(B)
    nA, nB = Amat.shape[0], Bmat.shape[0]

    # mean over the concatenation of A and B, without concatenating
    mu = (Amat.sum(axis=0) + Bmat.sum(axis=0)) / (nA + nB)

    UA = pca(Amat, n_dim)
    UB = pca(Bmat, n_dim)

    Ac = Amat - mu                       # one centred copy, not two
    PA = Ac @ UA                         # (n, k) - never form the (u, u) cov
    PB = Ac @ UB
    # trace(U' C_A U) == ||Ac @ U||_F^2 / (nA - 1); the 1/(nA-1) cancels
    return float((PB.ravel() @ PB.ravel()) / (PA.ravel() @ PA.ravel()))


# ----------------------------------------------------------------------
# cross-validated alignment index
# ----------------------------------------------------------------------
class _Half:
    """Everything about one trial-half that does not depend on its partner."""

    __slots__ = ("n", "mean", "S", "U")

    def __init__(self, X, n_dim, pca):
        M = _flat(X)
        self.n = M.shape[0]
        self.mean = M.mean(axis=0)
        X0 = M - self.mean
        self.S = X0.T @ X0               # scatter about its *own* mean, (u, u)
        self.U = pca(M, n_dim)           # (u, k)


def _quad(h, d, U):
    """trace(U' Xc' Xc U) where Xc = X - (joint mean) and d = joint mean - mean_X.

    Xc'Xc = S_X + n_X d d'  (the cross terms vanish because X0 has zero
    column means), so this stays exact while only touching (u, u) and (u, k).
    """
    v = d @ U
    return np.sum(U * (h.S @ U)) + h.n * (v @ v)


def _ai_pair(X, Y):
    d = (Y.n / (X.n + Y.n)) * (Y.mean - X.mean)
    return _quad(X, d, Y.U) / _quad(X, d, X.U)


def _split(X, parts):
    idx = np.random.permutation(X.shape[0])
    k = X.shape[0] // parts
    return [X[idx[i * k:(i + 1) * k]] for i in range(parts)]


def alignment_index_crossval(A, B, n_dim=10, pca=top_pcs, same=None,
                             match_diag=False):
    """
    same : bool or None
        True if A and B are the same condition (diagonal of the matrix).
        None -> inferred as `A is B`.
    match_diag : bool
        If True, off-diagonal halves are also subsampled to quarter size,
        so diagonal and off-diagonal entries use the same number of trials.
    """
    if same:
        # four disjoint quarters: A-halves and B-halves never share trials
        a1, a2, b1, b2 = _split(A, 4)
    elif match_diag:
        a1, a2 = _split(A, 4)[:2]
        b1, b2 = _split(B, 4)[:2]
    else:
        a1, a2 = _split(A, 2)
        b1, b2 = _split(B, 2)

    A1, A2 = _Half(a1, n_dim, pca), _Half(a2, n_dim, pca)
    B1, B2 = _Half(b1, n_dim, pca), _Half(b2, n_dim, pca)

    a = (_ai_pair(A1, B1) + _ai_pair(A2, B2)
         + _ai_pair(A1, B2) + _ai_pair(A2, B1)) \
        / (4 * (_ai_pair(A1, A2) + _ai_pair(A2, A1)))

    b = (_ai_pair(B1, A1) + _ai_pair(B2, A2)
         + _ai_pair(B1, A2) + _ai_pair(B2, A1)) \
        / (4 * (_ai_pair(B1, B2) + _ai_pair(B2, B1)))

    return a + b