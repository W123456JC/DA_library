import pylab


def scatter(x, y, size=20):
    """
        Parameters
        ----------
        x : numpy.ndarray
            feature data
        y : int
            domain labels
        size : int
            point size
    """
    pylab.scatter(x[:, 0], x[:, 1], size, y)
    pylab.show()