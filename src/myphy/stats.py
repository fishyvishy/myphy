import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2


def calculate_residuals(x, y, model, popt):
    """
    Calculate the residuals of a model fit.
    Parameters:
    x (array-like): The independent variable data.
    y (array-like): The dependent variable data.
    model (callable): The model function that takes x and parameters as input and returns the predicted y values.
    popt (array-like): The optimal parameters for the model function.
    Returns:
    array-like: The residuals, which are the differences between the observed y values and the model's predicted y values.
    """
    model_fit = model(x, *popt)
    residuals = y - model_fit
    return residuals


# def plot_residuals(x, y, y_fit, y_unc, xlabel="", ylabel="Residuals", save_as=""):
#     residuals = y - y_fit
#     plt.errorbar(
#         x, residuals, yerr=y_unc, fmt="o", ecolor="lightgray", elinewidth=2, capsize=1
#     )
#     plt.axhline(0, color="red", linestyle="--")
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     if save_as:
#         plt.savefig(plotpath / save_as, bbox_inches="tight")
#     plt.show()


def chi2vals(x, y, model, popt, y_unc):
    """
    Calculate the chi-squared statistic, reduced chi-squared, and chi-squared probability.
    Parameters:
    x (array-like): Observed data points.
    y_fit (array-like): Fitted data points.
    y_unc (array-like): Uncertainties in the observed data points.
    k (int): Number of fitted parameters.
    Returns:
    tuple: A tuple containing:
        - chi (float): The chi-squared statistic.
        - reduced_chi2 (float): The reduced chi-squared statistic.
        - chi2_prob (float): The probability of obtaining a chi-squared value at least as extreme as the one computed, given the degrees of freedom.
    """
    y_fit = model(x, *popt)
    chi = np.sum(((y - y_fit) / y_unc) ** 2)
    dof = len(y) - len(popt)
    reduced_chi2 = chi / dof
    chi2_prob = 1 - chi2.cdf(chi, dof)

    return chi, reduced_chi2, chi2_prob


def calculate_chi2(y, y_fit, y_unc, k):
    """
    Calculate the chi-squared statistic, reduced chi-squared, and chi-squared probability.
    Parameters:
    y (array-like): Observed data points.
    y_fit (array-like): Fitted data points.
    y_unc (array-like): Uncertainties in the observed data points.
    k (int): Number of fitted parameters.
    Returns:
    tuple: A tuple containing:
        - chi (float): The chi-squared statistic.
        - reduced_chi2 (float): The reduced chi-squared statistic.
        - chi2_prob (float): The probability of obtaining a chi-squared value at least as extreme as the one computed, given the degrees of freedom.
    """

    chi = np.sum(((y - y_fit) / y_unc) ** 2)
    dof = len(y) - k
    reduced_chi2 = chi / dof
    chi2_prob = 1 - chi2.cdf(chi, dof)
    return chi, reduced_chi2, chi2_prob
