import numpy as np
from myphy.mata import Mata
from myphy.plotter import get_plotter
from scipy.optimize import curve_fit
from scipy.stats import chi2


class NotFittedError(Exception):
    """Raised when a method requiring fit() is called before fitting the model."""

    pass


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
        self.fitted = False  # Ensure the model is not fitted initially

    def fit(self, p0):
        """
        Fit the model to the data using curve fitting.

        Args:
            p0: Initial guess for the parameters.
        """
        x, y, yerr, _ = self.data.unpack()
        popt, pcov = curve_fit(self.model, x, y, p0, sigma=yerr, absolute_sigma=True)

        self.popt = popt
        self.pcov = pcov
        self.perr = np.sqrt(np.diag(pcov))
        self.yfit = self.model(self.data.xdata, *self.popt)
        self.fitted = True

    def plot_model(self, samples=100, label="Fit", plotData=True):
        """
        Plot the model fit along with the data.

        Args:
            samples: Number of points to sample for the fit line.
            label: Label for the fit line in the plot.
            plotData: Whether to plot the original data points as well.
        """
        self._check_fitted()  # Ensure the model is fitted before plotting
        if plotData:
            self.data.plot_data("Data")

        x = self.data.xdata
        xspace = np.linspace(np.min(x), np.max(x), samples)
        yfit = self.model(xspace, *self.popt)
        get_plotter().make_plot(xspace, yfit, label=label)

    def stats(self):
        """
        Calculate chi-squared statistic, reduced chi-squared, and chi-squared probability.

        Returns:
            tuple: chi-squared statistic, reduced chi-squared, and chi-squared probability.
        """
        _, y, yerr, _ = self.data.unpack()
        chi = np.sum((self.residuals() / yerr) ** 2)
        dof = len(y) - len(self.popt)
        reduced_chi2 = chi / dof
        chi2_prob = 1 - chi2.cdf(chi, dof)

        return chi, reduced_chi2, chi2_prob

    def residuals(self):
        """
        Calculate the residuals of the model fit.

        Returns:
            np.ndarray: The residuals (differences between data and model).
        """
        residuals = self.data.ydata - self.yfit
        return residuals

    def plot_residuals(self):
        """
        Plot the residuals of the model fit.

        The residuals are plotted against the x-values with error bars.
        """
        x, _, yerr, xerr = self.data.unpack()
        get_plotter().make_residual(x, self.residuals(), yerr, xerr)

    def _check_fitted(self):
        """
        Ensure that the model has been fitted before calling methods that depend on it.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        if not self.fitted:
            raise NotFittedError(
                "The model has not been fitted yet. Please run the 'fit' method first."
            )
