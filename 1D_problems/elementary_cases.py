import numpy as np
import matplotlib.pyplot as plt

Q = -1
L = 1
EI = 1

q1 = lambda x: Q / (24*EI) * (x**4 - 4*L*x**3 + 6*L**2*x**2) # Fixed, free, constant q
q2 = lambda x: Q / (120*EI) * (x**5/L - 10*L*x**3 + 20*L**2*x**2) # Fixed, free, increasing q
q3 = lambda x: Q / (120*EI) * (-x**5/L + 5*x**4 - 10*L*x**3 + 10*L**2*x**2) # Fixed, free, decreasing q
q4 = lambda x: Q / (24*EI) * (x**4/L - 2*x**3 + x/L**2) # Pinned, pinned, constant q
q5 = lambda x: Q / (180*EI) * (3*x**5/L**2 - 10*x**3 + 7*L**2*x) # Pinned, pinned, increasing q
q6 = lambda x: Q / (180*EI) * (-3*x**5/L**2 + 15*x**4/L - 20*x**3 + 8*L**2*x) # Pinned, pinned, decreasing q
q7 = lambda x: Q / (48*EI) * (2*x**4 - 5*L*x**3 + 3*L**2*x**2) # Fixed, pinned, constant q

x = np.linspace(0, L, 100)

y1 = q7(x)

plt.plot(x, y1)
plt.show()


