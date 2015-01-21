"""Main module for Kaggle AXA Competition

Uses the logistic regression idea described by Stephane Soulier: https://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/11299/score-0-66-with-logistic-regression
Hence, we use the traces from every driver as positive examples and build a set of references that we use as negative examples. Note that our set is larger by one driver, in case the reference set includes the driver that we are currently using as positive.
"""

from datetime import datetime
from Driver import Driver
from MultiClassDriver import MultiClassDriver
import os
import sys
from random import sample, seed

def perform_analysis(folders, referencedrivers):
    print folders
    sys.stdout.flush()
    temp = []
    for folder in folders:
        temp.append(Driver(folder))
    cls = MultiClassDriver(temp, referencedrivers)
    cls.classify()
    return cls.toKaggle()


def chunks(l, n):
    """ Yield successive n-sized chunks from l.

    http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def analysis(foldername, outdir, partitionssize, referencenum, maxsize=None):
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
    #for testing
    if maxsize:
        folders = folders[:maxsize]
    results = []
    for folderlist in chunks(folders, partitionssize):
        results.append(perform_analysis(folderlist, referencedrivers))
    with open(os.path.join(outdir, "pyMultiClass_{0}.csv".format(submission_id)), 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for item in results:
            writefile.write("%s\n" % item)
    print 'Done, elapsed time: %s' % str(datetime.now() - start)

if __name__ == '__main__':
    MyPath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    # analysis(os.path.join(MyPath, "..", "axa-telematics", "data", "drivers"), MyPath, 40, 11, 40)
    analysis(os.path.join(MyPath, "..", "axa-telematics", "data", "drivers"), MyPath, 40, 11)