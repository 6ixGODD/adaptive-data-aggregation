import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

import lab1_ndpcm_library
import time


# General parameters
# =====================================================================================================
# n_bits = 12
# n = 200  # number of iterations
# h_depth = 3  # for now hardcode size of history to last 3 values
# alpha = 1e-9
# k_v = 1e-3

def run(n_bits: int, n: int, h_depth: int, alpha: float, k_v: float, with_noise: bool = False):
    # lab1_ndpcm_library.hyb_param = lab1_ndpcm_library.HybParam(alpha=alpha, k_v=k_v)

    # Generate sample ADC data
    x = np.linspace(0, 6 * pi, n + h_depth)  # add h_depth to initialize history
    # useful to evaluate function at lots of points
    f_original: np.ndarray = np.sin(x)
    # Scale to range 0-4095
    # f = (f_original+1)*4095
    # Scale to range 0-100
    # f = (f_original+1)*100
    # Scale to range 0-10000
    f: np.ndarray = (f_original + 1) * 10000
    noise = np.random.normal(0, 100, size=f.shape)
    print(noise)
    f += noise if with_noise else 0

    tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
    rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
    e = np.zeros(n)  # Prepare array for saving true error (f - y_hat)

    avg_errs = []
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    # plt.figure(figsize=(6, 6), dpi=200)
    # plt.title(f"Alpha Comparison")
    # for a in [1e-09, 1e-10, 1e-11, 1e-12]:
    #     lab1_ndpcm_library.hyb_param = lab1_ndpcm_library.HybParam(alpha=a, k_v=k_v)
    #     _tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
    #     _rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
    #     _e = np.zeros(n)
    #     for _i in range(h_depth):
    #         lab1_ndpcm_library.init_params(_tx_data, f[_i])
    #         lab1_ndpcm_library.init_params(_rx_data, f[_i])
    #     for _k in range(n):
    #         _y_hat = lab1_ndpcm_library.predict(_tx_data, _k)
    #         _eq = lab1_ndpcm_library.calculate_error(_tx_data, _k, f[_k + h_depth])
    #         lab1_ndpcm_library.update_params(_tx_data, _k)
    #         _t_re = lab1_ndpcm_library.reconstruct(_tx_data, _k)
    #         _e[_k] = f[_k] - _y_hat
    #         _rx_data.eq[_k + 1] = _eq
    #         _ = lab1_ndpcm_library.predict(_rx_data, _k)
    #         lab1_ndpcm_library.update_params(_rx_data, _k)
    #         _r_re = lab1_ndpcm_library.reconstruct(_rx_data, _k)
    #     _f = f[h_depth:]
    #     _e_all = np.abs(_rx_data.y_recreated[1:] - _f)
    #     avg_errs.append(np.average(_e_all))
    #     _tt = np.arange(1, n + 1)
    #     plt.plot(_tt, _rx_data.y_recreated[:-1] - _f, label=f"alpha={a}", linewidth=.5)
    # plt.xlabel("Iteration")
    # plt.ylabel("Error")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    # plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    # plt.savefig(
    #     f"figure.n_bits={n_bits}.alpha_comparison.png.iter={n}.h_depth={h_depth}.with_noise={with_noise}.alpha_comparison.png",
    #     dpi=300,
    #     bbox_inches='tight'
    # )
    # plt.close()
    # # plt.show()
    #
    # avg_errs = []
    # plt.figure(figsize=(6, 6), dpi=200)
    # plt.title(f"Kv Comparison")
    # for kv in [1, 5e-1, 2e-1, 1e-1, 1e-2, 1e-3]:
    #     lab1_ndpcm_library.hyb_param = lab1_ndpcm_library.HybParam(alpha=alpha, k_v=kv)
    #     _tx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
    #     _rx_data = lab1_ndpcm_library.init(n, h_depth, n_bits)
    #     _e = np.zeros(n)
    #     for _i in range(h_depth):
    #         lab1_ndpcm_library.init_params(_tx_data, f[_i])
    #         lab1_ndpcm_library.init_params(_rx_data, f[_i])
    #     for _k in range(n):
    #         _y_hat = lab1_ndpcm_library.predict(_tx_data, _k)
    #         _eq = lab1_ndpcm_library.calculate_error(_tx_data, _k, f[_k + h_depth])
    #         lab1_ndpcm_library.update_params(_tx_data, _k)
    #         _t_re = lab1_ndpcm_library.reconstruct(_tx_data, _k)
    #         _e[_k] = f[_k] - _y_hat
    #         _rx_data.eq[_k + 1] = _eq
    #         _ = lab1_ndpcm_library.predict(_rx_data, _k)
    #         lab1_ndpcm_library.update_params(_rx_data, _k)
    #         _r_re = lab1_ndpcm_library.reconstruct(_rx_data, _k)
    #     _f = f[h_depth:]
    #     _e_all = np.abs(_rx_data.y_recreated[1:] - _f)
    #     avg_errs.append(np.average(_e_all))
    #     _tt = np.arange(1, n + 1)
    #     plt.plot(_tt, _rx_data.y_recreated[:-1] - _f, label=f"kv={kv}", linewidth=.5)
    # plt.xlabel("Iteration")
    # plt.ylabel("Error")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    # plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    # plt.savefig(
    #     f"figure.n_bits={n_bits}.alpha_comparison.png.iter={n}.h_depth={h_depth}.with_noise={with_noise}.kv_comparison.png",
    #     dpi=300,
    #     bbox_inches='tight'
    # )
    # plt.close()
    # plt.show()

    # Initialization
    # ========================================================================================================
    # >> STEP-1: Initialize \phi(k) with `h_depth` data points
    lab1_ndpcm_library.hyb_param = lab1_ndpcm_library.HybParam(alpha=alpha, k_v=k_v)
    for i in range(h_depth):
        lab1_ndpcm_library.init_params(tx_data, f[i])
        lab1_ndpcm_library.init_params(rx_data, f[i])

    # Iterations
    # ============================================================================================================
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
        # >> Receiver STEP-3: Update coefficients
        lab1_ndpcm_library.update_params(rx_data, k)
        # >> Receiver STEP-4: Reconstruct data
        r_re = lab1_ndpcm_library.reconstruct(rx_data, k)

    # print(rx_data.y_hat)
    # Plotting
    # ==============================================================================================================
    f = f[h_depth:]  # remove history
    e_all = np.abs(rx_data.y_recreated[1:] - f)  # 1: for removing initial value
    print(
        "cumulative error (reconstructed y - ADC data) =", e_all.sum(), " average error = ", np.average(e_all)
    )
    print("Bits transmitted = ", n_bits * n)
    print("Assuming 1kSamples/second bitrate [bits/sec]=", n_bits * n / 1000)

    tt = np.arange(1, n + 1)

    plt.figure(figsize=(6, 6), dpi=200)
    plt.title(f"n_bits={n_bits}, alpha={alpha}, k_v={k_v}, iter={n}, h_depth={h_depth}, with_noise={with_noise}")
    plt.subplot(3, 1, 1)
    plt.ylabel("Sensor value")
    plt.plot(tt, f, label="ADC data", color="red", linewidth=.5)
    plt.plot(tt, tx_data.y_hat[1:], label="TX y_hat", color="green", linewidth=.5)  # same above
    plt.plot(
        tt,
        tx_data.y_recreated[:-1],
        label="TX recreated data",
        color="blue",
        linewidth=.5
    )  # :-1 for removing zero at the end
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    # plt.savefig('figure.1.png', dpi=300, bbox_inches='tight')

    # plt.figure(figsize=(6, 6), dpi=200)
    plt.subplot(3, 1, 2)
    # error of the final reconstruction
    # plt.xlabel("Iteration")
    # plt.ylabel("Sensor value")
    plt.plot(tt, rx_data.y_recreated[:-1] - f, label="reconstruction error", linewidth=.5)  # same above
    # plt.plot(tt, rx_data.y_recreated[1:], label="RX recreated data")
    # plt.plot(tt, f, label="ADC data", color="red")
    # plt.plot(tt, rx_data.eq[1:], label="RX eq")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    # plt.savefig('figure.2.png', dpi=300, bbox_inches='tight')

    plt.subplot(3, 1, 3)
    plt.plot(tt, rx_data.eq[:-1], label="quantized error", linewidth=.5)
    plt.plot(tt, e, label="true error", linewidth=.5)
    plt.xlabel("Iteration")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    # plt.legend()
    plt.savefig(
        f"figure.n_bits={n_bits}.alpha={alpha}.k_v={k_v}.png.iter={n}.h_depth={h_depth}.with_noise={with_noise}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(figsize=(6, 6), dpi=200)
    plt.title(f"Error distribution")
    plt.hist(e_all, bins=100)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    plt.savefig(
        f"figure.n_bits={n_bits}.alpha={alpha}.k_v={k_v}.png.iter={n}.h_depth={h_depth}.with_noise={with_noise}.hist.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    plt.figure(figsize=(6, 6), dpi=200)
    plt.title(f"Theta")
    plt.plot(tt, rx_data.theta[1:], label="theta", linewidth=.5)

    plt.xlabel("Iteration")
    plt.ylabel("Theta")
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    plt.savefig(
        f"figure.n_bits={n_bits}.alpha={alpha}.k_v={k_v}.png.iter={n}.h_depth={h_depth}.with_noise={with_noise}.theta.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


if __name__ == '__main__':
    # run(12, 800, 5, 1e-9, 1e-2, with_noise=True)
    # run(12, 800, 3, 1e-9, 1e-2, with_noise=True)
    # run(12, 800, 3, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(15, 800, 3, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(16, 800, 3, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(12, 800, 4, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(15, 800, 4, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(16, 800, 4, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(12, 1200, 3, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(15, 1200, 3, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(16, 1200, 3, 1e-9, 1e-2, with_noise=True)
    time.sleep(1)
    run(12, 3, 3, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(15, 100, 3, 1e-9, 1e-2, with_noise=True)
    # time.sleep(1)
    # run(16, 100, 3, 1e-9, 1e-2, with_noise=True)
    #
    # # time.sleep(1)
    # # run(12, 800, 3, 1e-9, 1e-2, with_noise=False)
    # # time.sleep(1)
    # # run(15, 800, 3, 1e-9, 1e-2, with_noise=False)
    # # time.sleep(1)
    # # run(16, 800, 3, 1e-9, 1e-2, with_noise=False)
    # # time.sleep(1)
    # # run(12, 800, 4, 1e-9, 1e-2, with_noise=False)
    # # time.sleep(1)
    # # run(15, 800, 4, 1e-9, 1e-2, with_noise=False)
    # # time.sleep(1)
    # # run(16, 800, 4, 1e-9, 1e-2, with_noise=False)
    # # time.sleep(1)
    # # run(12, 1200, 3, 1e-9, 1e-2, with_noise=False)
    # # time.sleep(1)
    # # run(15, 1200, 3, 1e-9, 1e-2, with_noise=False)
    # # time.sleep(1)
    # # run(16, 1200, 3, 1e-9, 1e-2, with_noise=False)
    # time.sleep(1)
    # run(12, 100, 3, 1e-9, 1e-2, with_noise=False)
    # time.sleep(1)
    # run(15, 100, 3, 1e-9, 1e-2, with_noise=False)
    # time.sleep(1)
    # run(16, 100, 3, 1e-9, 1e-2, with_noise=False)
