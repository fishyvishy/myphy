import numpy as np
from myphy.mata import Mata
from myphy.plotter import makePlot, makeResidual
from scipy.optimize import curve_fit
from scipy.stats import chi2


class Meval:
    """
    A class to evaluate a model with given data and uncertainties.

    Attributes
    ----------
    model : callable
        Model function to be evaluated.

    popt : optimal model parameters
    pcov : covariance matrix of popt params
    perr : error of popt params
    """

    def __init__(self, model, mata: Mata):
        """
        Initializes the instance with the given model and Mata data.
        Args:
            model: The model to be used.
            mata (Mata): The Mata data object.
        """
        self.model = model
        self.data = mata

    def fit(self, p0):
        x, y, yerr, xerr = self.data.unpack()
        popt, pcov = curve_fit(self.model, x, y, p0, sigma=yerr, absolute_sigma=True)

        self.popt = popt
        self.pcov = pcov
        self.perr = np.sqrt(np.diag(pcov))
        self.yfit = self.model(self.data.xdata, *self.popt)

    def plotModel(self, samples=100, label="Fit", plotData=True):
        x, y, yerr, xerr = self.data.unpack()
        xspace = np.linspace(np.min(x), np.max(x), samples)
        yfit = self.model(xspace, *self.popt)
        makePlot(xspace, yfit, label=label)
        if plotData:
            self.data.plotData("Data")

    def stats(self):
        """
        Calculate the chi-squared statistic, reduced chi-squared, and chi-squared probability.
        Returns:
            tuple: chi-squared statistic, reduced chi-squared, and chi-squared probability.
        """
        x, y, yerr, xerr = self.data.unpack()
        chi = np.sum((self.residuals() / yerr) ** 2)
        dof = len(y) - len(self.popt)
        reduced_chi2 = chi / dof
        chi2_prob = 1 - chi2.cdf(chi, dof)

        return chi, reduced_chi2, chi2_prob

    def residuals(self):
        residuals = self.data.ydata - self.yfit
        return residuals

    def plotResiduals(self):
        x, y, yerr, xerr = self.data.unpack()
        makeResidual(x, self.residuals(), yerr, xerr)
