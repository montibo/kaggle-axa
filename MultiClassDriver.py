import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from random import sample, seed
from sklearn.multiclass import OneVsRestClassifier
from collections import OrderedDict

class MultiClassDriver(object):
    """Class for Regression-based analysis of Driver traces"""

    def __init__(self, drivers, otherdrivers):
        """Initialize by providing a (positive) driver example and a dictionary of (negative) driver references."""
        seed(42)
        self.numfeatures = drivers[0].num_features
        self.__drivers = OrderedDict()
        # Initialize train and test np arrays
        self.__data = np.empty((0, self.numfeatures), float)
        self.__labels = np.empty((0,), float)
        for k in range(len(drivers)):
            driver = drivers[k]
            featurelist = []
            self.__drivers[driver.identifier] = []
            for trace in driver.traces:
                self.__drivers[driver.identifier].append(trace.identifier)
                featurelist.append(trace.features)
            newdata = np.asarray(featurelist)
            self.__data = np.append(self.__data, newdata, axis=0)
            self.__labels = np.append(self.__labels, k * np.ones((newdata.shape[0],)), axis=0)
        self.__addeddata = np.empty((0, self.numfeatures), float)
        self.__addedlabels = np.empty((0,), float)
        for driver in otherdrivers:
            if driver not in drivers:
                featurelist = []
                for trace in driver.traces:
                    featurelist.append(trace.features)
                newdata = np.asarray(featurelist)
                self.__addeddata = np.append(self.__addeddata, newdata, axis=0)
                self.__addedlabels = np.append(self.__addedlabels, -1 * np.ones((newdata.shape[0],)), axis=0)
        gbr = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42)
        self.__clf = OneVsRestClassifier(gbr, n_jobs=-1)
        self.__y = []

    def classify(self):
        """Perform classification"""
        train = np.concatenate((self.__data, self.__addeddata), axis=0)
        trainlabel = np.concatenate((self.__labels, self.__addedlabels), axis=0)
        # print train.shape
        # print trainlabel.shape
        self.__clf.fit(train,trainlabel)
        self.__y = self.__clf.predict_proba(self.__data)

    def toKaggle(self):
        """Return string in Kaggle submission format"""
        returnstring = ""
        for k in xrange(len(self.__drivers.keys())):
            key = self.__drivers.keys()[k]
            for i in xrange(len(self.__drivers[key])):
                returnstring += "%d_%d,%.3f\n" % (key, self.__drivers[key][i], self.__y[i,k])
        return returnstring[:-2]

    def validate(self, datadict):
        pass