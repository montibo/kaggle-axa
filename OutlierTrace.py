import os
import numpy as np
from math import hypot, acos, pi
from Trace import median, distance, getratio, velocities_and_distance_covered, meanstdv, bettervariance, overlap


def reject_outliers(data, m=2):
    '''http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list'''
    return data[abs(data - np.mean(data)) < m * np.std(data)]

class Trace(object):
    """"
    Trace class reads a trace file and computes features of that trace. Version that uses outlier detection and no smoothing
    """

    def __init__(self, filename):
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
        triplength = distance(x[0], y[0], x[-1], y[-1])
        self.triptime = len(x)
        xvar, yvar = bettervariance(x, y)
        overlaps = overlap(x, y)
        self.featurelist = []
        self.featurelist.append(triplength)
        self.featurelist.append(self.triptime)
        self.featurelist.append(xvar)
        self.featurelist.append(yvar)
        self.featurelist.append(overlaps)

        v, distancecovered = velocities_and_distance_covered(x, y)
        vfiltered = reject_outliers(v, 3)
        maxspeed = max(vfiltered)
        medianspeed = median(vfiltered)
        meanspeed, varspeed = meanstdv(vfiltered)
        angles, jumps = self.angles_and_jumps()
        anglespeed = [speed * angle for (speed, angle) in zip(v, angles)]
        anglespeedfiltered  = reject_outliers(anglespeed, 3)
        maxanglespeed = max(anglespeedfiltered)
        mediananglespeed = median(anglespeedfiltered)
        meananglespeed, varanglespeed = meanstdv(anglespeedfiltered)
        totalangle = sum(angles)
        maxangle = max(angles)
        minangle = min(angles)
        medianangle = max(angles)
        acc, dec, stills = getratio(vfiltered, medianspeed)
        if (acc + dec) == 0:
            accratio = 0.5
        else:
            accratio = acc / (acc + dec)
        self.featurelist.append(distancecovered)
        self.featurelist.append(maxspeed)
        self.featurelist.append(medianspeed)
        self.featurelist.append(meanspeed)
        self.featurelist.append(varspeed)
        self.featurelist.append(maxanglespeed)
        self.featurelist.append(mediananglespeed)
        self.featurelist.append(meananglespeed)
        self.featurelist.append(varanglespeed)
        self.featurelist.append(totalangle)
        self.featurelist.append(medianangle)
        self.featurelist.append(maxangle)
        self.featurelist.append(minangle)
        self.featurelist.append(jumps)
        self.featurelist.append(accratio)
        self.featurelist.append(acc)
        self.featurelist.append(dec)
        self.featurelist.append(stills)

    @property
    def features(self):
        """Returns a list that comprises all computed features of this trace."""
        return self.featurelist

    def angles_and_jumps(self):
        jumps = 0.0
        angles = []
        previousmax = distance(self.__xn[1], self.__yn[1], self.__xn[0], self.__yn[0])
        for i in xrange(1, len(self.__xn)):
            angles.append(self.getAngle(self.__xn[i], self.__yn[i], self.__xn[i-1], self.__yn[i-1]))
            current = distance(self.__xn[i], self.__yn[i], self.__xn[i-1], self.__yn[i-1])
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
