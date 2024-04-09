import os
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2 as cv
from PAUT_Data import PAUT_Data
from matplotlib import pyplot as plt
from plots_utils import plot_Cscan, plot_Dscan, plot_Ascan

from scipy.optimize import curve_fit
from scipy.stats import poisson

data = []
for f in os.listdir("C:/users/dalmonte/data/temp"):
    path = os.path.join("C:/users/dalmonte/data/temp", f)
    data.append(np.load(path))
data = np.array(data)

print(data.shape)


def fit_function(k, lamb):
    # The parameter lamb will be used as the fit parameter
    return poisson.pmf(k, lamb)

n = 1

entries, bin_edges, _ = plt.hist(data[n], bins=np.arange(100)-0.5, density=True, alpha=0.75, label="Data")
middles_bins = (bin_edges[1:] + bin_edges[:-1]) * 0.5

parameters, cov_matrix = curve_fit(fit_function, middles_bins, entries)
print(f"parameters: {parameters}")
print(f"cov_matrix: {cov_matrix}")

x_plot = np.arange(0, 100, 1)

plt.plot(
    x_plot,
    fit_function(x_plot, *parameters),
    linestyle="-",
    color="red",
    label="Fit result",
)
plt.legend()
plt.show()


# convert data to shannon entropy
probability = poisson.pmf(np.round(data[n], 0).astype(np.int32), parameters[0])
print(probability)
shannon_entropy = -np.log(probability+1e-10)

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6,8), dpi=150, tight_layout=True)
ax[0].plot(data[n])
ax[0].set_title("Data")
ax[1].plot(probability)
ax[1].set_title("Probability")
ax[2].plot(shannon_entropy)
ax[2].set_title("Shannon entropy")
plt.show()


