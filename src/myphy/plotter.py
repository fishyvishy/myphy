import matplotlib.pyplot as plt


def makePlot(x, y, label):
    plt.plot(x, y, label=label)


def makeErrorbar(x, y, yerr, xerr, label=""):
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=".", label=label)


def makeResidual(x, y, yerr, xerr):
    makeErrorbar(x, y, yerr=yerr, xerr=xerr)
    plt.axhline(0, color="black", linestyle="--")
