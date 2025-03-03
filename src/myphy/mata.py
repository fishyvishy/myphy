import numpy as np
from myphy.plotter import get_plotter


class Mata:
    """
    A class to handle data with uncertainties.
    """

    def __init__(self, x, y, yerr, xerr=None, copy=False):
        x, y = map(np.asarray, (x, y))

        # Broadcast uncertainties if scalar
        if np.isscalar(yerr):
            yerr = np.full_like(x, yerr, dtype=float)
        else:
            yerr = np.asarray(yerr)

        self.xerr_flag = xerr is not None
        if self.xerr_flag:
            if np.isscalar(xerr):
                xerr = np.full_like(x, xerr, dtype=float)
            else:
                xerr = np.asarray(xerr)

        # Ensure all inputs have the same shape
        shape_flag = x.shape == y.shape == yerr.shape and (
            not self.xerr_flag or x.shape == xerr.shape
        )
        if not shape_flag:
            raise ValueError("All input arrays must have the same shape.")

        # Store as a stacked array (copying only if necessary)
        self.data = np.stack((x, y, yerr, xerr) if self.xerr_flag else (x, y, yerr))
        if copy:
            self.data = self.data.copy()

        self.mask = None

    def updateMask(self, mask):
        "update mask for data"
        mask = np.asarray(mask)
        if mask.shape != self.data.shape[1:]:
            raise ValueError("Mask must have the same shape as one data array.")
        self.mask = mask

    def clearMask(self):
        self.mask = None

    @property
    def xdata(self):
        if self.mask is not None:
            return self.data[0][self.mask]
        return self.data[0]

    @property
    def ydata(self):
        if self.mask is not None:
            return self.data[1][self.mask]
        return self.data[1]

    @property
    def yerr(self):
        if self.mask is not None:
            return self.data[2][self.mask]
        return self.data[2]

    @property
    def xerr(self):
        if self.xerr_flag:
            if self.mask is not None:
                return self.data[3][self.mask]
            return self.data[3]
        return None

    def unpack(self):
        return self.xdata, self.ydata, self.yerr, self.xerr

    def plot_data(self, label="Data"):
        get_plotter().make_errorbar(*self.unpack(), label=label)

    def __repr__(self):
        out_dict = {"xdata": self.xdata, "ydata": self.ydata, "yerr": self.yerr}
        if self.xerr_flag:
            out_dict["xerr"] = self.xerr
        return str(out_dict)
