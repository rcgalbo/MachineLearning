import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn_theano.datasets import fetch_mnist_generated

X = fetch_mnist_generated(n_samples=1600, random_state=1999)

# plotting based on
# http://stackoverflow.com/questions/4098131/matplotlib-update-a-plot
num_updates = len(X) // 16
f, axarr = plt.subplots(4, 4)
objarr = np.empty_like(axarr)
for n, ax in enumerate(axarr.flat):
    objarr.flat[n] = ax.imshow(X[n], cmap='gray', interpolation='nearest')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
plt.show(block=False)

for i in range(num_updates):
    for n, obj in enumerate(objarr.flat):
        obj.set_data(X[i * len(objarr.flat) + n])
    plt.draw()
    time.sleep(.08)
    if (i % 20) == 0:
        print("Iteration %i" % i)
plt.show()
