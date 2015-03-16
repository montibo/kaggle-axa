import matplotlib.pyplot as plt
import os

def plot_np(nparray, title=None):
    plt.scatter(nparray[:,0], nparray[:,1])
    if title:
        fig.suptitle(title, fontsize=20)
    plt.plot(nparray[0,:], 'rD')
    plt.plot(nparray[-1,:], 'ks')
    plt.show()

def plot_list(x,y, titlename=None):
    plt.scatter(x, y)
    if titlename:
        plt.title(titlename, fontsize=20)
    plt.plot(x[0], y[0], 'rD')
    plt.plot(x[-1], y[-1], 'ks')
    plt.show()


def plot_file(filename, title=None):
    x = []
    y = []
    with open(filename, "r") as trainfile:
        trainfile.readline()  # skip header
        for line in trainfile:
            items = line.split(",")
            x.append(items[0])
            y.append(items[1])
    fig = plt.figure()
    plt.scatter(x, y)
    if title:
        fig.suptitle(title, fontsize=20)
    plt.plot(x[0], y[0], 'rD')
    plt.plot(x[-1], y[-1], 'ks')
    plt.show()

if __name__ == '__main__':
    from random import sample, seed
    seed(42)
    # plot_file(os.path.join("data", "drivers_small", "3600", "100.csv"))
    # plot_file(os.path.join("data", "drivers_small", "3600", "102.csv"))
    #
    #
    # plot_file(os.path.join("data", "drivers_small", "3600", "1.csv"))
    # plot_file(os.path.join("data", "drivers_small", "3600", "10.csv"))
    # plot_file(os.path.join("data", "drivers_small", "3600", "104.csv"))
    #
    # plot_file(os.path.join("data", "drivers_small", "3600", "101.csv"))
    # plot_file(os.path.join("data", "drivers_small", "3600", "108.csv"))

    # Funky plots --Faked
    # plot_file(os.path.join("data", "drivers_small", "3600", "107.csv"))
    # plot_file(os.path.join("data", "drivers_small", "3600", "112.csv"))
    #
    # # Missing GPS values
    # plot_file(os.path.join("data", "drivers_small", "3600", "117.csv"))
    #
    # plot_file(os.path.join("data", "drivers_small", "3600", "92.csv"))
    foldername = os.path.join("data", "drivers", "191")
    files = [os.path.join(foldername, f) for f in os.listdir(foldername) if os.path.isfile(os.path.join(foldername, f))]
    referencenum = 10
    referencefiles = [files[i] for i in sorted(sample(xrange(len(files)), referencenum))]
    for file in referencefiles:
        plot_file(file)