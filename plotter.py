import matplotlib.pyplot as plt
import numpy as np

bands = 3
rows = 8

x = np.arange(0, 1, 0.01)
plt.plot(x, 1-pow(1-pow(x,rows), bands))
plt.title(f"bands: {bands}, rows: {rows}")

# plt.show()
plt.savefig(f"{bands}-{rows}.png")