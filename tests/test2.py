from myphy import Mata
import numpy as np

a = np.array([i for i in range(1, 11)])
b = a + 10
c = a / 10

mata = Mata(a, b, c)

print(mata)
