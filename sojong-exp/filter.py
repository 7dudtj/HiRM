import numpy as np
import math

class Filter(object):
    def __init__(self, option) -> None:
        pass
class LinearFilter(Filter):
    def __init__(self, option) -> None:
        super(LinearFilter, self).__init__(option)
    def __call__(self, s: np.array) -> np.array:
        return s
class IdealLowPassFilter(Filter):
    def __init__(self, option) -> None:
        super(IdealLowPassFilter, self).__init__(option)
    def __call__(self, s: np.array) -> np.array:
        return np.ones(shape=s.shape, dtype=float)
class GaussianFilter(Filter):
    def __init__(self, alpha=0.2) -> None:
        super(GaussianFilter, self).__init__(alpha)
        if alpha < 0:
            self.alpha = 0.2
        else:
            self.alpha = alpha
    def __call__(self, s: np.array) -> np.array:
        return np.exp(-self.alpha * (s**2))
class HeatKernelFilter(Filter):
    def __init__(self, alpha=0.1) -> None:
        super(HeatKernelFilter, self).__init__(alpha)
        if alpha < 0:
            self.alpha = 0.1
        else:
            self.alpha = alpha
    def __call__(self, s: np.array) -> np.array:
        return np.exp(-self.alpha * s)
class ButterWorthFilter(Filter):
    def __init__(self, order=1) -> None:
        super(ButterWorthFilter, self).__init__(order)
        self.order = order
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
    def __init__(self, mu=0.1) -> None:
        super(GFCFLinearAutoencoderFilter, self).__init__(mu)
        if mu < 0:
            self.mu = 0.1
        else:
            self.mu = mu
    def __call__(self, s: np.array) -> np.array:
        return (1 - s) / (1 - s + self.mu)
class GFCFNeighborhoodBasedFilter(Filter):
    def __init__(self, option) -> None:
        super(GFCFNeighborhoodBasedFilter, self).__init__(option)
    def __call__(self, s: np.array) -> np.array:
        return (1 - s)
class InverseFilter(Filter):
    def __init__(self, option) -> None:
        super(InverseFilter, self).__init__(option)
    def __call__(self, s: np.array) -> np.array:
        return 1/s
