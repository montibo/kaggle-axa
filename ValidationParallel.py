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
from joblib import Parallel, delayed
from Trace import median

REFERENCE_DATA = {}
TEST_DATA = {}


def generatedata(drivers):
    """
    Generates reference data for regression

    Input: List of driver folders that are read.
    Returns: Nothing, since this data is stored in global variable ReferenceData
    """
    global REFERENCE_DATA
    for driver in drivers:
        REFERENCE_DATA[driver.identifier] = driver.generate_data_model

def generatetestdata(drivers):
    """
    Generates reference data for regression

    Input: List of driver folders that are read.
    Returns: Nothing, since this data is stored in global variable ReferenceData
    """
    global TEST_DATA
    for driver in drivers:
        TEST_DATA[driver.identifier] = driver.generate_data_model


def perform_analysis(trainfolder):
    print "Working on {0}".format(trainfolder)
    sys.stdout.flush()
    temp = Driver(trainfolder)
    cls = RegressionDriver(temp, REFERENCE_DATA)
    cls.classify()
    return cls.validate(TEST_DATA)


def analysis(trainfoldername, testfoldername, outdir, referencenum):
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
    trainfolders = [os.path.join(trainfoldername, f) for f in os.listdir(trainfoldername) if os.path.isdir(os.path.join(trainfoldername, f))]
    referencefolders = [trainfolders[i] for i in sorted(sample(xrange(len(trainfolders)), referencenum))]
    referencedrivers = []
    for referencefolder in referencefolders:
        referencedrivers.append(Driver(referencefolder))
    generatedata(referencedrivers)
    testdrivers = []
    testfolders = [os.path.join(testfoldername, f) for f in os.listdir(testfoldername) if os.path.isdir(os.path.join(testfoldername, f))]
    for testfolder in testfolders:
        testdrivers.append(Driver(testfolder))
    generatetestdata(testdrivers)
    results = Parallel(n_jobs=10)(delayed(perform_analysis)(trainfolder) for trainfolder in trainfolders)
    with open(os.path.join(outdir, "testing_results_{0}.txt".format(submission_id)), 'w') as writefile:
        for item in results:
            writefile.write("%.4f\n" % item)
        mean = sum(results)/len(results)
        writefile.write("Mean: %.4f\n" % mean)
        writefile.write("Median: %.4f\n" % median(results))
        writefile.write("Min: %.4f\n" % min(results))
            
    print 'Done, elapsed time: %s' % str(datetime.now() - start)

if __name__ == '__main__':
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    valpath = os.path.join(MyPath,"..", "axa-telematics", "data")
    analysis(os.path.join(valpath, "train"), os.path.join(valpath, "test"), MyPath, 21)
