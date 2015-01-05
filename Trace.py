import os
from math import hypot


def smooth(x, y, steps):
    """
    Returns moving average using steps samples to generate the new trace

    Input: x-coordinates and y-coordinates as lists as well as an integer to indicate the size of the window (in steps)
    Output: list for smoothed x-coordinates and y-coordinates
    """
    xnew = []
    ynew = []
    for i in xrange(steps, len(x)):
        xnew.append(sum(x[i-steps:i]) / float(steps))
        ynew.append(sum(y[i-steps:i]) / float(steps))
    return xnew, ynew


def distance(x0, y0, x1, y1):
    """Computes 2D euclidean distance"""
    return hypot((x1 - x0), (y1 - y0))


def velocities_and_distance_covered(x, y):
    """
    Returns velocities just using difference in distance between coordinates as well as accumulated distances

    Input: x-coordinates and y-coordinates as lists
    Output: list of velocities
    """
    v = []
    distancesum = 0.0
    for i in xrange(1, len(x)):
        dist = distance(x[i-1], y[i-1], x[i], y[i])
        v.append(dist)
        distancesum += dist
    return v, distancesum


class Trace(object):
    """"
    Trace class reads a trace file and computes features of that trace.
    """

    def __init__(self, filename, filtering=10):
        """Input: path and name of the file of a trace; how many filtering steps should be used for sliding window filtering"""
        self.__id = int(os.path.basename(filename).split(".")[0])
        x = []
        y = []
        with open(filename, "r") as trainfile:
            trainfile.readline()  # skip header
            for line in trainfile:
                items = line.split(",", 2)
                x.append(float(items[0]))
                y.append(float(items[1]))
        self.__xn, self.__yn = smooth(x, y, filtering)
        v, self.distancecovered = velocities_and_distance_covered(self.__xn, self.__yn)
        self.maxspeed = max(v)
        self.triplength = distance(x[0], y[0], x[-1], y[-1])
        self.triptime = len(x)

    @property
    def features(self):
        """Returns a list that comprises all computed features of this trace."""
        features = []
        features.append(self.triplength)
        features.append(self.triptime)
        features.append(self.distancecovered)
        features.append(self.maxspeed)
        return features

    def __str__(self):
        return "Trace {0} has this many positions: \n {1}".format(self.__id, self.triptime)

    @property
    def identifier(self):
        """Driver identifier is its filename"""
        return self.__id
