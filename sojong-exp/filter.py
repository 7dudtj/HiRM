import numpy as np
import math

class Filter(object):
    def __init__(self) -> None:
        pass
class LinearFilter(Filter):
    def __init__(self) -> None:
        super(LinearFilter, self).__init__()
    def __call__(self, s: np.array) -> np.array:
        return s
class IdealLowPassFilter(Filter):
    def __init__(self) -> None:
        super(IdealLowPassFilter, self).__init__()
    def __call__(self, s: np.array) -> np.array:
        return np.ones(shape=s.shape, dtype=float)
class GaussianFilter(Filter):
    def __init__(self, *nargs) -> None:
        super(GaussianFilter, self).__init__()
        self.alpha = 0.2 if len(nargs) == 0 else nargs[0]
    def __call__(self, s: np.array) -> np.array:
        return np.exp(-self.alpha * (s**2))
class HeatKernelFilter(Filter):
    def __init__(self, *nargs) -> None:
        super(HeatKernelFilter, self).__init__()
        self.alpha = 0.1 if len(nargs) == 0 else nargs[0]
    def __call__(self, s: np.array) -> np.array:
        return np.exp(-self.alpha * s)
class ButterWorthFilter(Filter):
    def __init__(self, *nargs) -> None:
        super(ButterWorthFilter, self).__init__()
        self.order = 1 if len(nargs)==0 else nargs[0]
        self.order_list = [1,2,3]
        if self.order == 1:
            self.butterworth = lambda s: 1 / (s + 1)
        elif self.order == 2:
            self.butterworth = lambda s: 1 / (s**2 + math.sqrt(2)*s + 1)
        elif self.order == 3:
            self.butterworth = lambda s: 1 / ((s+1) * (s**2 + s + 1))
        else:
            print("We only use filter order value in [1, 2, 3]")
            raise NotImplementedError
    def __call__(self, s: np.array) -> np.array:
        return self.butterworth(s)
class GFCFLinearAutoencoderFilter(Filter):
    def __init__(self, *nargs) -> None:
        super(GFCFLinearAutoencoderFilter, self).__init__()
        self.mu = 0.1 if len(nargs)==0 else nargs[0]
    def __call__(self, s: np.array) -> np.array:
        return (1 - s) / (1 - s + self.mu)
class GFCFNeighborhoodBasedFilter(Filter):
    def __init__(self) -> None:
        super(GFCFNeighborhoodBasedFilter, self).__init__()
    def __call__(self, s: np.array) -> np.array:
        return (1 - s)
class InverseFilter(Filter):
    def __init__(self) -> None:
        super(InverseFilter, self).__init__()
    def __call__(self, s: np.array) -> np.array:
        return 1/s
