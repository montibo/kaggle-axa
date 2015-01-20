import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from random import sample, seed


class RegressionDriver(object):
    """Class for Regression-based analysis of Driver traces"""

    def __init__(self, driver, datadict, numberofrows=100, numfeatures = 200):
        """Initialize by providing a (positive) driver example and a dictionary of (negative) driver references."""
        seed(42)
        self.driver = driver
        self.numfeatures = self.driver.num_features
        featurelist = []
        # self.__clf = GradientBoostingRegressor(n_estimators=500, max_depth=4)
        gbr = GradientBoostingRegressor(n_estimators=500, max_depth=10, max_features=numfeatures, random_state=42)
        pca = PCA(whiten=True, n_components=numfeatures)
        estimators = [('polyf', PolynomialFeatures()), ('scale', MinMaxScaler()), ('pca', PCA()), ('gbr', gbr)]
        self.__clf = Pipeline(estimators)
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
        self.__y = np.ones((self.__testdata.shape[0],))

    def classify(self):
        """Perform classification"""
        self.__clf.fit(self.__traindata, self.__trainlabels)
        self.__y = self.__clf.predict(self.__testdata)
        feature_importance = self.__clf.steps[-1][1].feature_importances_
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