import matplotlib.pyplot as plt
import numpy as np

bands = 4
rows = 6

x = np.arange(0, 1, 0.01)
plt.plot(x, 1-pow(1-pow(x,rows), bands))
plt.title(f"bands: {bands}, rows: {rows}")
plt.xlabel("doc sim")
plt.ylabel("chance they are candidate pairs")

# plt.show()
plt.savefig(f"{bands}-{rows}.png")