from myphy import Mata
import numpy as np
import matplotlib.pyplot as plt

a = np.array([i for i in range(1, 11)])
b = a + 10
c = a / 10
d = a / 10

mata = Mata(a, b, c, d)

mata.plot_data()
plt.show()
