import numpy as np
import matplotlib.pyplot as plt
from myphy import Mata, Meval


x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
yu = np.random.rand(x.shape[0]) / 10
y = y + yu - 0.05

xu = np.random.rand(x.shape[0]) / 100

mata = Mata(x, y, yu, xu)

mata.plotData()
plt.legend()
plt.show()


def mysin(x, A):
    return A * np.sin(x)


sinfit = Meval(mysin, mata)
sinfit.fit((1))

sinfit.plotModel()
plt.legend()
plt.show()

sinfit.plotResiduals()
plt.show()


print(sinfit.stats())
print(sinfit.popt)
