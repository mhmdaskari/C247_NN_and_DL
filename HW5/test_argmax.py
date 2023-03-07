import numpy as np
a = np.random.randint(0,9,(2,2,3))
print(a)
print(np.argmax(a, axis=[1,2]))