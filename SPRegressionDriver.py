import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
from random import sample, seed
from sklearn.preprocessing import StandardScaler
from SparseFilter import SparseFilter

class RegressionDriver(object):
    """Class for Regression-based analysis of Driver traces"""

    def __init__(self, driver, datadict, numberofrows=200):
        """Initialize by providing a (positive) driver example and a dictionary of (negative) driver references."""
        seed(42)
        self.driver = driver
        self.numfeatures = self.driver.num_features
        self.numrawfeatures = self.driver.num_rawfeatures
        featurelist = []
        rawfeaturelist = []
        self.__clf = GradientBoostingRegressor(n_estimators=200)
        self.__indexlist = []
        for trace in self.driver.traces:
            self.__indexlist.append(trace.identifier)
            featurelist.append(trace.features)
            temp = trace.rawfeatures
            for i in range(len(trace.rawfeatures), self.numrawfeatures):
                temp.append(-1e7)
            rawfeaturelist.append(temp)
        # Initialize train and test np arrays
        self.__traindata = np.asarray(featurelist)
        self.__testdata = np.asarray(featurelist)
        self.__rawtraindata = np.asarray(rawfeaturelist)
        self.__rawtestdata = np.asarray(rawfeaturelist)

        self.__trainlabels = np.ones((self.__traindata.shape[0],))
        data = np.empty((0, self.numfeatures), float)
        rawdata = np.empty((0, self.numrawfeatures), float)
        # print rawdata.shape
        # print data.shape
        setkeys = datadict.keys()
        if driver.identifier in setkeys:
            setkeys.remove(driver.identifier)
        else:
            setkeys = sample(setkeys, len(setkeys) - 1)
        for key in setkeys:
            if key != driver.identifier:
                rand_smpl = [datadict[key]['feat'][i] for i in sorted(sample(xrange(len(datadict[key]['feat'])), numberofrows))]
                data = np.append(data, np.asarray(rand_smpl), axis=0)
                raw_rand_smpl = [datadict[key]['raw'][i] for i in sorted(sample(xrange(len(datadict[key]['raw'])), numberofrows))]
                temp = np.asarray(raw_rand_smpl)
                if temp.shape[1] < self.numrawfeatures:
                    z = np.zeros((temp.shape[0], self.numrawfeatures - temp.shape[1]), dtype=temp.dtype)
                    temp = np.concatenate((temp, z), axis=1)
                    rawdata = np.append(rawdata, temp, axis=0)
                else:
                    # print temp.shape
                    ix = range(temp.shape[0])
                    iy = range(self.numrawfeatures)
                    # print self.numrawfeatures
                    newtemp = temp[:, :self.numrawfeatures]
                    # print newtemp.shape
                    # print rawdata.shape
                    rawdata = np.append(rawdata, newtemp, axis=0)
        # print rawdata.shape
        # print self.__rawtraindata.shape
        self.__rawtraindata = np.append(self.__rawtraindata, rawdata, axis=0)
        self.__traindata = np.append(self.__traindata, data, axis=0)
        self.__trainlabels = np.append(self.__trainlabels, np.zeros((data.shape[0],)), axis=0)
        self.__y = np.ones((self.__testdata.shape[0],))

    def classify(self):
        """Perform classification"""
        train_X = np.asarray(self.__rawtraindata)
        train_y = np.asarray(self.__trainlabels)
        test_X = np.asarray(self.__rawtestdata)

        train_feat_X = np.asarray(self.__traindata)
        test_feat_X = np.asarray(self.__testdata)
        # print train_feat_X.shape
        # print test_feat_X.shape

        scaler = StandardScaler().fit(np.r_[train_X, test_X])
        train_X = scaler.transform(train_X)
        test_X = scaler.transform(test_X)

        ## train a sparse filter on both train and test data
        sf = SparseFilter(n_features=20, n_iterations=1000)
        sf.fit(np.r_[train_X, test_X])
        train_sf_X = sf.transform(train_X)
        test_sf_X = sf.transform(test_X)
        print train_sf_X
        print test_sf_X

        ss = StandardScaler()
        train_combined_X = ss.fit_transform(np.c_[train_sf_X, train_feat_X])
        test_combined_X = ss.transform(np.c_[test_sf_X, test_feat_X])

        self.__clf.fit(train_combined_X, train_y.ravel())
        self.__y = self.__clf.predict(test_combined_X)
        feature_importance = self.__clf.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        print feature_importance

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
