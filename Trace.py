import os
from math import hypot, acos, pi


def median(mylist):
    '''Median as described here http://stackoverflow.com/questions/10482339/how-to-find-median'''
    sorts = sorted(mylist)
    length = len(sorts)
    if not length % 2:
        return (sorts[length / 2] + sorts[length / 2 - 1]) / 2.0
    return sorts[length / 2]


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


def bettervariance(x, y):
    "Variance in principal directions (since traces may be rotated rabitrarily)"
    from sklearn.decomposition import PCA
    import numpy as np
    column1 = np.asarray(x).T
    column2 = np.asarray(y).T
    X = np.column_stack((column1, column2))
    pca = PCA()
    pca.fit(X)
    # print pca.explained_variance_ratio_
    return pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]


def overlaps(x, y):
    """Overlaps in measurements"""
    count = 0
    for i in xrange(1, len(x)):
        if x[i-1] == x[i] and y[i-1] == y[i]:
            count += 1
    return count


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
        self.medianspeed = median(v)
        self.triplength = distance(x[0], y[0], x[-1], y[-1])
        self.triptime = len(x)
        self.xvar, self.yvar = bettervariance(x, y)
        self.overlaps = overlaps(x, y)
        self.jumps = 0.0
        angles = self.angles_and_jumps()
        self.totalangle = sum(angles)
        self.maxangle = max(angles)
        self.medianangle = max(angles)
        self.__acc = 0.0
        self.__dec = 0.0
        self.__stills = 0.0
        self.getratio(v)
        if (self.__acc + self.__dec) == 0:
            self.__accratio = 0.5
        else:
            self.__accratio = self.__acc / (self.__acc + self.__dec)

    @property
    def features(self):
        """Returns a list that comprises all computed features of this trace."""
        features = []
        features.append(self.triplength)
        features.append(self.triptime)
        features.append(self.distancecovered)
        features.append(self.maxspeed)
        features.append(self.medianspeed)
        features.append(self.totalangle)
        features.append(self.medianangle)
        features.append(self.maxangle)
        features.append(self.jumps)
        features.append(self.__accratio)
        features.append(self.__acc)
        features.append(self.__dec)
        features.append(self.__stills)
        features.append(self.xvar)
        features.append(self.yvar)
        features.append(self.overlaps)
        return features

    def angles_and_jumps(self):
        angles = []
        previousmax = distance(self.__xn[1], self.__yn[1], self.__xn[0], self.__yn[0])
        for i in xrange(1, len(self.__xn)):
            angles.append(self.getAngle(self.__xn[i], self.__yn[i], self.__xn[i-1], self.__yn[i-1]))
            current = distance(self.__xn[i], self.__yn[i], self.__xn[i-1], self.__yn[i-1])
            if current > previousmax:
                if current > 10 * previousmax:
                    self.jumps += 1
                previousmax = current
        return angles


    def getratio(self, v):
        threshold = 0.03 * self.medianspeed
        prev = v[0]
        for i in xrange(1, len(v)):
            if v[i] > (1+threshold) * prev:
                self.__acc += 1
            elif v[i] < (1-threshold) * prev:
                self.__dec += 1
            else:
                self.__stills += 1


    def getAngle(self, x1, y1, x0, y0):
        epsilon = 10e-2
        # get Length
        len1 = float(hypot(x1, y1))
        len2 = float(hypot(x0, y0))
        try:
            return acos((x1*y1+ x0*y0)/(len1*len2))
        except ValueError:
            # self.straights += 1
            if (x0/len1-x1/len2) < epsilon and (y0/len1-y1/len2) < epsilon:
                return 0.0
            else:
                return pi
        except ZeroDivisionError:
            return -1.0*pi

    def __str__(self):
        return "Trace {0} has this many positions: \n {1}".format(self.__id, self.triptime)

    @property
    def identifier(self):
        """Driver identifier is its filename"""
        return self.__id
