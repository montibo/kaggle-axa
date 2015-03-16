import os
from SPTrace import Trace
import numpy as np

class Driver(object):
    """"
    Driver class assigns traces in sub-directories to the driver in a given foldername.
    """

    def __init__(self, foldername):
        """
        Initialize Driver with a foldername.
        """
        self._id = int(os.path.basename(foldername))
        self._traces = []
        files = [f for f in os.listdir(foldername) if f.endswith(".csv")]
        for filename in files:
            self._traces.append(Trace(os.path.join(foldername, filename)))

    def __str__(self):
        return "Driver {0} has {1} traces".format(self._id, len(self._traces))

    @property
    def identifier(self):
        """Returns driver identifier determined by its folder name."""
        return self._id

    @property
    def traces(self):
        """Returns all traces of a driver."""
        return self._traces

    @property
    def trace(self, index):
        """Returns trace specified by the index. (Note that in this competition each driver has 200 traces.)"""
        return self._traces[index]

    @property
    def num_features(self):
        """Returns the number of features based on trace 0."""
        return len(self._traces[0].features)

    @property
    def num_rawfeatures(self):
        """Returns the number of features based on trace 0."""
        return max([len(self._traces[i].rawfeatures) for i in range(len(self._traces))])

    @property
    def generate_data_model(self):
        """Returns a list of all features for all traces."""
        listoffeatures = {}
        listoffeatures['feat'] = []
        listoffeatures['raw'] = []
        for i in xrange(len(self._traces)):
            listoffeatures['feat'].append(self._traces[i].features)
            length = len(self._traces[i].rawfeatures)
            temp = self._traces[i].rawfeatures
            for k in range(length, self.num_rawfeatures):
                temp.append(-1e7)
            listoffeatures['raw'].append(temp)
        print np.asarray(listoffeatures['feat']).shape
        print np.asarray(listoffeatures['raw']).shape 
        return listoffeatures
