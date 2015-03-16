import os
import numpy as np
from math import hypot, acos, pi
from Trace import distance, getratio, velocities_and_distance_covered, bettervariance, overlap
from scipy.stats.mstats import mquantiles

def reject_outliers(data, m=2):
    '''http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list'''
    data = np.asarray(data)
    return data[abs(data - np.mean(data)) < m * np.std(data)]

class Trace(object):
    """"
    Trace class reads a trace file and computes features of that trace. Version that uses outlier detection and no smoothing
    """

    def __init__(self, filename, quantiles=[0.0, 0.25, 0.5, 0.75, 1.0]):
        """Input: path and name of the file of a trace; how many filtering steps should be used for sliding window filtering"""
        self.__id = int(os.path.basename(filename).split(".")[0])
        self._x = []
        self._y = []
        with open(filename, "r") as trainfile:
            trainfile.readline()  # skip header
            for line in trainfile:
                items = line.split(",", 2)
                self._x.append(float(items[0]))
                self._y.append(float(items[1]))
        self.rawfeaturelist = []
        self.rawfeaturelist.extend(self._x)
        self.rawfeaturelist.extend(self._y)
        triplength = distance(self._x[0], self._y[0], self._x[-1], self._y[-1])
        self.triptime = len(self._x)
        xvar, yvar = bettervariance(self._x, self._y)
        overlaps = overlap(self._x, self._y)
        self.featurelist = []
        self.featurelist.append(triplength)
        self.featurelist.append(self.triptime)
        self.featurelist.append(xvar)
        self.featurelist.append(yvar)
        self.featurelist.append(overlaps)

        v, distancecovered = velocities_and_distance_covered(self._x, self._y)
        vfiltered = reject_outliers(v, 3)
        self.rawfeaturelist.extend(vfiltered)
        speed_qs = mquantiles(vfiltered, prob=quantiles)
        angles, jumps = self.angles_and_jumps()
        self.rawfeaturelist.extend(angles)
        anglespeed = [speed * angle for (speed, angle) in zip(v, angles)]
        anglespeedfiltered = reject_outliers(anglespeed, 3)
        anglespeed_qs = mquantiles(anglespeedfiltered, prob=quantiles)
        totalangle = sum(angles)
        angle_qs = mquantiles(angles, prob=quantiles)
        acc, dec, stills = getratio(vfiltered, anglespeed_qs[2])

        self.featurelist.append(distancecovered)
        self.featurelist.extend(speed_qs)
        self.featurelist.extend(anglespeed_qs)
        self.featurelist.append(totalangle)
        self.featurelist.extend(angle_qs)
        self.featurelist.append(acc)
        self.featurelist.append(dec)
        self.featurelist.append(stills)

    @property
    def features(self):
        """Returns a list that comprises all computed features of this trace."""
        return self.featurelist

    @property
    def rawfeatures(self):
        """Returns a list that comprises all computed features of this trace."""
        return self.rawfeaturelist

    def angles_and_jumps(self):
        jumps = 0.0
        angles = []
        previousmax = distance(self._x[1], self._y[1], self._x[0], self._y[0])
        for i in xrange(1, len(self._x)):
            angles.append(self.getAngle(self._x[i], self._y[i], self._x[i-1], self._y[i-1]))
            current = distance(self._x[i], self._y[i], self._x[i-1], self._y[i-1])
            if current > previousmax:
                if current > 10 * previousmax:
                    jumps += 1
                previousmax = current
        return angles, jumps


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
