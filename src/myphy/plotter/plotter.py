from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class BasePlotter(ABC):
    """Abstract base class for a plotting backend."""

    @abstractmethod
    def make_plot(self, x, y, label):
        pass

    @abstractmethod
    def make_errorbar(self, x, y, yerr, xerr, label):
        pass

    @abstractmethod
    def make_residual(self, x, y, yerr, xerr):
        pass


class DefaultPlotter(BasePlotter):
    """Default implementation using Matplotlib."""

    def make_plot(self, x, y, label):
        plt.plot(x, y, label=label)

    def make_errorbar(self, x, y, yerr, xerr, label):
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=".", label=label)

    def make_residual(self, x, y, yerr, xerr):
        self.make_errorbar(x, y, yerr, xerr, label=None)
        plt.axhline(0, color="black", linestyle="--")
