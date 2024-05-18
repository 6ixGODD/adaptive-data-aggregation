import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

import lab1_ndpcm_library

# General parameters =====================================================================================================
n_bits = 16
n = 1000  # number of iterations
# h_depth = 3  # for now hardcode size of history to last 3 values
h_depth = 6

# Generate sample ADC data 
x = np.linspace(0, 2 * pi, n + h_depth)  # add h_depth to initialize history
# useful to evaluate function at lots of points
f_original: np.ndarray = np.sin(x)
# Scale to range 0-4095
# f = (f_original+1)*4095
# Scale to range 0-100
# f = (f_original+1)*100
# Scale to range 0-10000
f: np.ndarray = (f_original + 1) * 10000

tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)

e = np.zeros(n)  # Prepare array for saving true error (f - y_hat)

# Initialization ========================================================================================================
# >> STEP-1: Initialize \phi(k) with `h_depth` data points
for i in range(h_depth):
    lab1_ndpcm_library.init_params(tx_data, f[i])
    lab1_ndpcm_library.init_params(rx_data, f[i])

# Iterations ============================================================================================================
# for k in range(1, n - 1):
for k in range(n):
    print(">> Iteration", k)
    # TX side (transmitter)
    # >> Transmitter STEP-2: Calculate estimate y_hat(k)
    y_hat = lab1_ndpcm_library.predict(tx_data, k)
    # >> Transmitter STEP-3: Calculate estimation error e(k) and quantize it
    eq = lab1_ndpcm_library.calculate_error(tx_data, k, f[k + h_depth])
    # >> Transmitter STEP-4: Calculate \theta(k) update and update \phi(k)
    lab1_ndpcm_library.update_params(tx_data, k)
    t_re = lab1_ndpcm_library.reconstruct(tx_data, k)

    e[k] = f[k] - y_hat  # Save

    # communication (e.g. inject error)

    rx_data.eq[k + 1] = eq

    # RX side (receiver)
    # >> Receiver STEP-2: Calculate estimate y_hat(k)
    _ = lab1_ndpcm_library.predict(rx_data, k)
    # >> Receiver STEP-4: Update coefficients
    lab1_ndpcm_library.update_params(rx_data, k)
    # >> Receiver STEP-3: Reconstruct data
    r_re = lab1_ndpcm_library.reconstruct(rx_data, k)


# Plotting ==============================================================================================================
f = f[h_depth:]  # remove history
e_all = np.abs(rx_data.y_recreated[1:] - f)  # :1 for removing initial value
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

plt.plot(tt, f, label="ADC data", color="red")
plt.plot(tt, tx_data.y_hat[1:], label="TX y_hat", color="green")  # same above
plt.plot(tt, tx_data.y_recreated[:-1], label="TX recreated data", color="blue")  # :-1 for removing zero at the end
plt.legend()

plt.subplot(3, 1, 2)
# error of the final reconstruction
plt.plot(tt, rx_data.y_recreated[:-1] - f, label="reconstruction error")  # same above
# plt.plot(tt, rx_data.y_recreated[1:], label="RX recreated data")
# plt.plot(tt, f, label="ADC data", color="red")
# plt.plot(tt, rx_data.eq[1:], label="RX eq")
plt.legend()

plt.subplot(3, 1, 3)
# error of the final reconstruction
plt.plot(tt, rx_data.eq[:-1], label="quantized error")
plt.plot(tt, e, label="true error")
plt.legend()

plt.show()
