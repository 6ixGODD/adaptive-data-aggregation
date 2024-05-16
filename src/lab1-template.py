import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

import lab1_ndpcm_library

# General parameters
n_bits = 16
n = 100  # number of iterations
h_depth = 3  # for now hardcode size of history to last 3 values

# Generate sample ADC data 
x = np.linspace(0, 2 * pi, n)
# useful to evaluate function at lots of points 
f_original = np.sin(x)
# Scale to range 0-4095
# f = (f_original+1)*4095
# Scale to range 0-100
# f = (f_original+1)*100
# Scale to range 0-100
f = (f_original + 1) * 10000

tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)

e = np.zeros(n)  # Prepare array for saving true error (f - y_hat)

for k in range(1, n - 1):
    # TX side
    # Run compression part with update of coefficients
    lab1_ndpcm_library.prepare_params_for_prediction(tx_data, k)
    y_hat = lab1_ndpcm_library.predict(tx_data, k)
    eq = lab1_ndpcm_library.calculate_error(tx_data, k, f[k])
    y_rec = lab1_ndpcm_library.reconstruct(tx_data, k)

    e[k] = f[k] - y_hat  # Save 
    # communication (e.g. inject error)
    rx_data.eq[k] = tx_data.eq[k]

    # RX side 
    # receiver side - recreate
    lab1_ndpcm_library.prepare_params_for_prediction(rx_data, k)
    y_hat_rx = lab1_ndpcm_library.predict(rx_data, k)
    # No need to calculate ERROR - since it was received
    y_rec_rx = lab1_ndpcm_library.reconstruct(rx_data, k)

# print(tx_data)
# print(rx_data)
e_all = np.abs(rx_data.y_recreated - f)
print(
    "cumulative error (reconstructed y - ADC data) =", e_all.sum(), " average error = ", np.average(e_all)
)
print("Bits transmitted = ", n_bits * n)
print("Assuming 1kSamples/second bitrate [bits/sec]=", n_bits * n / 1000)

tt = np.arange(1, n + 1)

plt.subplot(3, 1, 1)
plt.title("N_bits=" + str(n_bits))
plt.xlabel("Iteration")
plt.ylabel("Sensor value")

plt.plot(tt, f, label="ADC data")
plt.plot(tt, tx_data.y_hat, label="TX y_hat")
plt.plot(tt, tx_data.y_recreated, label="TX recreated data")
plt.legend()

plt.subplot(3, 1, 2)
# error of the final reconstruction
plt.plot(tt, rx_data.y_recreated - f, label="reconstruction error")
plt.legend()

plt.subplot(3, 1, 3)
# error of the final reconstruction
plt.plot(tt, rx_data.eq, label="quantized error")
plt.plot(tt, e, label="true error")
plt.legend()

plt.show()
