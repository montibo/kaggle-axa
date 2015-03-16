import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from random import sample, seed
from sklearn.decomposition import TruncatedSVD
from math import floor
from sklearn import cross_validation

import numpy as np
from numpy.linalg import norm, svd

def inexact_augmented_lagrange_multiplier(X, lmbda=.01, tol=1e-3,
                                          maxiter=100, verbose=True):
    """
    Inexact Augmented Lagrange Multiplier
    """
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1/mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            break
    if verbose:
        print "Finished at iteration %d" % (itr)    
    return A, E

class RegressionDriver(object):
    """Class for Regression-based analysis of Driver traces"""

    def __init__(self, driver, datadict, numberofrows=40): #, numfeatures = 200):
        """Initialize by providing a (positive) driver example and a dictionary of (negative) driver references."""
        seed(42)
        self.driver = driver
        self.numfeatures = self.driver.num_features
        featurelist = []
        self.__clf = GradientBoostingRegressor(n_estimators=300, max_depth=4, min_samples_leaf=2)
        # gbr = GradientBoostingRegressor(n_estimators=500, max_depth=10, max_features=numfeatures, random_state=42)
        # pca = PCA(whiten=True, n_components=numfeatures)
        # estimators = [('polyf', PolynomialFeatures()), ('scale', MinMaxScaler()), ('pca', PCA()), ('gbr', gbr)]
        # self.__clf = Pipeline(estimators)
        self.__indexlist = []
        for trace in self.driver.traces:
            self.__indexlist.append(trace.identifier)
            featurelist.append(trace.features)
        # Initialize train and test np arrays
        self.__traindata = np.asarray(featurelist)
        self.__testdata = np.asarray(featurelist)
        self.__trainlabels = np.ones((self.__traindata.shape[0],))
        data = np.empty((0, self.numfeatures), float)
        setkeys = datadict.keys()
        if driver.identifier in setkeys:
            setkeys.remove(driver.identifier)
        else:
            setkeys = sample(setkeys, len(setkeys) - 1)
        for key in setkeys:
            if key != driver.identifier:
                rand_smpl = [datadict[key][i] for i in sorted(sample(xrange(len(datadict[key])), numberofrows)) ]
                data = np.append(data, np.asarray(rand_smpl), axis=0)
        self.__traindata = np.append(self.__traindata, data, axis=0)
        self.__trainlabels = np.append(self.__trainlabels, np.zeros((data.shape[0],)), axis=0)
        self.__y = np.zeros((self.__testdata.shape[0],))

    def classify(self, nfolds=4):
        """Perform classification"""
        components = self.__traindata.shape[1]

        _, train_rpca_X_np = inexact_augmented_lagrange_multiplier(np.nan_to_num(self.__traindata))
        _, test_rpca_X_np = inexact_augmented_lagrange_multiplier(np.nan_to_num(self.__testdata))
        skf = cross_validation.StratifiedKFold(self.__trainlabels, n_folds=nfolds)

        for train_index, _ in skf:
            X_train = train_rpca_X_np[train_index]
            y_train = self.__trainlabels[train_index]
            self.__clf.fit(X_train, y_train)
            self.__y += self.__clf.predict(test_rpca_X_np)
        self.__y /= float(nfolds)
        # feature_importance = self.__clf.feature_importances_
        # feature_importance = 100.0 * (feature_importance / feature_importance.max())
        # print feature_importance

    def toKaggle(self):
        """Return string in Kaggle submission format"""
        returnstring = ""
        for i in xrange(len(self.__indexlist) - 1):
            returnstring += "%d_%d,%.6f\n" % (self.driver.identifier, self.__indexlist[i], self.__y[i])
        returnstring += "%d_%d,%.6f" % (self.driver.identifier, self.__indexlist[len(self.__indexlist)-1], self.__y[len(self.__indexlist)-1])
        return returnstring

    def validate(self, datadict):
        from sklearn.metrics import roc_auc_score
        testdata = np.empty((0, self.numfeatures), float)
        y_true = np.empty((0,), float)
        for key in datadict.keys():
            currenttestdata = np.asarray(datadict[key])
            testdata = np.append(testdata, currenttestdata, axis=0)
            if key != self.driver.identifier:
                y_true = np.append(y_true, np.zeros((currenttestdata.shape[0],)), axis=0)
            else:
                y_true = np.append(y_true, np.ones((currenttestdata.shape[0],)), axis=0)
        y_score = self.__clf.predict(testdata)
        result = roc_auc_score(y_true, y_score)
        return result