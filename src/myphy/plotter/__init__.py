from .plotter import DefaultPlotter, BasePlotter

# Global plotter instance
_global_plotter = DefaultPlotter()


def get_plotter():
    """Returns the current global plotter."""
    return _global_plotter


def set_plotter(plotter):
    """Allows users to set a custom plotter."""
    global _global_plotter
    if not isinstance(plotter, BasePlotter):
        raise TypeError("Custom plotter must be a subclass of BasePlotter")
    _global_plotter = plotter


# Expose BasePlotter so users can subclass it
__all__ = ["BasePlotter", "DefaultPlotter", "get_plotter", "set_plotter"]
