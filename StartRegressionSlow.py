"""Main module for Kaggle AXA Competition

Uses the logistic regression idea described by Stephane Soulier: https://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/11299/score-0-66-with-logistic-regression
Hence, we use the traces from every driver as positive examples and build a set of references that we use as negative examples. Note that our set is larger by one driver, in case the reference set includes the driver that we are currently using as positive.
"""

from datetime import datetime
from Driver import Driver
from RegressionDriver import RegressionDriver
import os
import sys
from random import sample, seed

REFERENCE_DATA = {}


def generatedata(drivers):
    """
    Generates reference data for regression

    Input: List of driver folders that are read.
    Returns: Nothing, since this data is stored in global variable ReferenceData
    """
    global REFERENCE_DATA
    for driver in drivers:
        REFERENCE_DATA[driver.identifier] = driver.generate_data_model


def perform_analysis(folder):
    print "Working on {0}".format(folder)
    sys.stdout.flush()
    temp = Driver(folder)
    cls = RegressionDriver(temp, REFERENCE_DATA)
    cls.classify()
    return cls.toKaggle()


def analysis(foldername, outdir, referencenum):
    """
    Start the analysis

    Input:
        1) Path to the driver directory
        2) Path where the submission file should be written
        3) Number of drivers to compare against
    """
    seed(42)
    start = datetime.now()
    submission_id = datetime.now().strftime("%H_%M_%B_%d_%Y")
    folders = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isdir(os.path.join(foldername, f))]
    referencefolders = [folders[i] for i in sorted(sample(xrange(len(folders)), referencenum))]
    referencedrivers = []
    for referencefolder in referencefolders:
        referencedrivers.append(Driver(referencefolder))
    generatedata(referencedrivers)
    results = [perform_analysis(folder) for folder in folders]
    with open(os.path.join(outdir, "pyRegression_{0}.csv".format(submission_id)), 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for item in results:
            writefile.write("%s\n" % item)
    print 'Done, elapsed time: %s' % str(datetime.now() - start)

if __name__ == '__main__':
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    analysis(os.path.join(MyPath, "..", "axa-telematics", "data", "drivers_small"), MyPath, 5)
