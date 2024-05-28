import copy
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from NADPCMC import NADPCMC

plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Times New Roman'

RANDOM_SEED = 42


def plot(
        nadpcmc_tx: NADPCMC,
        nadpcmc_rx: NADPCMC,
        y: np.ndarray,
        dpi: int = 200,
        save_dir: str = './output'
):
    """
    Plot the original signal, the predicted signal, the reconstructed signal, the error and the error distribution

    Args:
        nadpcmc_tx (NADPCMC): transmitter
        nadpcmc_rx (NADPCMC): receiver
        y (np.ndarray): original signal
        dpi (int): resolution of the plot
        save_dir (str): directory to save the plot

    """
    if not Path(save_dir).joinpath('plots').exists():
        Path(save_dir).joinpath('plots').mkdir(parents=True, exist_ok=True)
    save_dir = Path(save_dir).joinpath('plots')
    plt.figure(figsize=(10, 6), dpi=dpi)
    plt.subplot(3, 1, 1)
    plt.title('Signal Prediction and Reconstruction')
    plt.ylabel('Value')
    plt.plot(np.arange(1, nadpcmc_tx.n_iter + 1), y, label='Original Signal', color='yellow', linewidth=0.5)
    plt.plot(
        np.arange(1, nadpcmc_tx.n_iter + 1),
        nadpcmc_tx.y_hat[1:],
        label='Predicted Signal',
        color='red',
        linewidth=0.5
    )
    plt.plot(
        np.arange(1, nadpcmc_tx.n_iter + 1),
        nadpcmc_rx.y_recreated[:-1],
        label='Reconstructed Signal',
        color='blue',
        linewidth=0.5
    )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')

    plt.subplot(3, 1, 2)
    plt.plot(nadpcmc_rx.y_recreated[:-1] - y, label='Reconstruction Error', color='blue', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')

    plt.subplot(3, 1, 3)
    plt.plot(nadpcmc_rx.eq[:-1], label='Quantization Error', color='blue', linewidth=0.5)
    plt.plot(y - nadpcmc_tx.y_hat[1:], label='Prediction Error', color='red', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(
        Path(save_dir) /
        f'signal.n_bits={nadpcmc_tx.n_bits}.n_iter={nadpcmc_tx.n_iter}.h_depth={nadpcmc_tx.h_depth}'
        f'.alpha={nadpcmc_tx.alpha}.k_v={nadpcmc_tx.k_v}.with_noise={nadpcmc_tx.with_noise}.png',
        dpi=dpi,
        bbox_inches='tight'
    )
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 6), dpi=dpi)
    plt.title('Error Distribution')
    plt.hist(nadpcmc_rx.eq[:-1], bins=100, alpha=0.5, label='Quantization Error', color='blue')
    plt.hist(nadpcmc_tx.y_hat[1:] - y, bins=100, alpha=0.5, label='Prediction Error', color='red')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(
        Path(save_dir) /
        f'error.n_bits={nadpcmc_tx.n_bits}.n_iter={nadpcmc_tx.n_iter}.h_depth={nadpcmc_tx.h_depth}'
        f'.alpha={nadpcmc_tx.alpha}.k_v={nadpcmc_tx.k_v}.with_noise={nadpcmc_tx.with_noise}.png',
        dpi=dpi,
        bbox_inches='tight'
    )
    # plt.show()
    plt.close()

    plt.figure(figsize=(10, 6), dpi=dpi)
    plt.title('Theta Update')
    plt.plot(nadpcmc_tx.theta, label='Theta', color='black', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    plt.xlabel('Iteration')
    plt.ylabel('Theta')
    plt.tight_layout()
    plt.savefig(
        Path(save_dir) /
        f'theta.n_bits={nadpcmc_tx.n_bits}.n_iter={nadpcmc_tx.n_iter}.h_depth={nadpcmc_tx.h_depth}'
        f'.alpha={nadpcmc_tx.alpha}.k_v={nadpcmc_tx.k_v}.with_noise={nadpcmc_tx.with_noise}.png'
    )
    # plt.show()
    plt.close()


def iterate(nadpcmc_tx: NADPCMC, nadpcmc_rx: NADPCMC, y: np.ndarray, save_dir: str = './output') -> tuple:
    if not Path(save_dir).joinpath('metrics').exists():
        Path(save_dir).joinpath('metrics').mkdir(parents=True, exist_ok=True)
    for k in range(nadpcmc_tx.n_iter):
        print(f'== Iteration: {k + 1}/{nadpcmc_tx.n_iter}')
        # Transmitting
        nadpcmc_tx.predict(k)  # predict y_hat
        eq = nadpcmc_tx.calculate_error(k, y[k])  # calculate quantization error
        nadpcmc_tx.update_theta(k)  # update theta
        nadpcmc_tx.reconstruct(k)  # reconstruct the signal
        # nadpcmc_tx.update_phi(k, y[k])  # update phi with the real value
        nadpcmc_tx.update_phi(k, nadpcmc_tx.y_recreated[k])  # update phi with the real value

        print(f'-- Quantization Error: {eq}')

        # Receiving
        nadpcmc_rx.receive_error(k, eq)  # receive quantization error from transmitter
        nadpcmc_rx.predict(k)  # predict y_hat
        nadpcmc_rx.update_theta(k)  # update theta
        nadpcmc_rx.reconstruct(k)  # reconstruct the signal
        nadpcmc_rx.update_phi(k, nadpcmc_rx.y_recreated[k])  # update phi with the reconstructed value

    # Calculate Metrics
    print('=' * 50)
    # Distortion = |(y - y_recreated) / y| * 100%
    distortion = np.mean(np.abs(nadpcmc_rx.y_recreated[:-1] - y) / y * 100)
    print(f'Mean Distortion: {distortion} %')
    # Compression Ratio = total bits in y / total bits in eq and some y for initialization
    # assuming original signal is 16-bit float
    total_bits_y = nadpcmc_tx.n_iter * 16
    total_bits_eq = nadpcmc_tx.n_iter * nadpcmc_tx.n_bits + nadpcmc_tx.h_depth * nadpcmc_tx.n_bits
    compression_ratio = total_bits_y / total_bits_eq
    print(f'Compression Ratio: {compression_ratio}')
    # Save the results
    metrics = pd.DataFrame(
        {
            'distortion':        [distortion],
            'compression_ratio': [compression_ratio]
        }
    )
    metrics.to_csv(
        Path(save_dir) /
        f'metrics/metrics.n_bits={nadpcmc_tx.n_bits}.n_iter={nadpcmc_tx.n_iter}.h_depth={nadpcmc_tx.h_depth}'
        f'.alpha={nadpcmc_tx.alpha}.k_v={nadpcmc_tx.k_v}.with_noise={nadpcmc_tx.with_noise}.csv'
    )
    return nadpcmc_tx, nadpcmc_rx


def alphas_comparison(
        n_iter: int,
        h_depth: int,
        n_bits: int,
        k_v: float = 1e-02,
        with_noise: bool = False,
        alphas: list = None,
        save_dir: str = './output'
):
    if not Path(save_dir).joinpath('plots').exists():
        Path(save_dir).joinpath('plots').mkdir(parents=True, exist_ok=True)
    save_dir = Path(save_dir).joinpath('plots')

    if alphas is None:
        alphas = [1e-07, 1e-08, 1e-09, 1e-10, 1e-11]
    plt.figure(figsize=(10, 6), dpi=200)
    plt.title('Alpha Comparison')
    for alpha in alphas:
        print(f'Evaluating Alpha: {alpha}')
        nadpcmc_tx = NADPCMC(n_iter, h_depth, n_bits, alpha, k_v)
        nadpcmc_rx = copy.deepcopy(nadpcmc_tx)
        y = nadpcmc_tx.generate_signal(seed=RANDOM_SEED, with_noise=with_noise)
        nadpcmc_tx.init_with_signal(y)
        nadpcmc_rx.init_with_signal(y)
        y = y[h_depth:]  # remove the values that are used for initialization
        try:  # to catch the exception when OVERFLOW
            nadpcmc_tx, nadpcmc_rx = iterate(nadpcmc_tx, nadpcmc_rx, y)
        except Exception as e:
            print(f'Overflow: {e}')
            continue
        __error = np.abs(nadpcmc_rx.y_recreated[:-1] - y)
        plt.plot(__error, label=f'Alpha={alpha}', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.tight_layout()
    plt.savefig(
        Path(save_dir) /
        f'alpha_comparison.n_bits={n_bits}.n_iter={n_iter}.h_depth={h_depth}.k_v={k_v}.with_noise={with_noise}.png',
        dpi=200,
        bbox_inches='tight'
    )
    plt.show()
    plt.close()


def k_vs_comparison(
        n_iter: int,
        h_depth: int,
        n_bits: int,
        alpha: float = 1e-09,
        with_noise: bool = False,
        k_vs: list = None,
        save_dir: str = './output'
):
    if not Path(save_dir).joinpath('plots').exists():
        Path(save_dir).joinpath('plots').mkdir(parents=True, exist_ok=True)
    save_dir = Path(save_dir).joinpath('plots')
    if k_vs is None:
        k_vs = [1e-01, 1e-02, 1e-03, 1e-04, 1e-05]
    plt.figure(figsize=(10, 6), dpi=200)
    plt.title('K_v Comparison')
    for k_v in k_vs:
        print(f'Evaluating K_v: {k_v}')
        nadpcmc_tx = NADPCMC(n_iter, h_depth, n_bits, alpha, k_v)
        nadpcmc_rx = copy.deepcopy(nadpcmc_tx)
        y = nadpcmc_tx.generate_signal(seed=RANDOM_SEED, with_noise=with_noise)
        nadpcmc_tx.init_with_signal(y)
        nadpcmc_rx.init_with_signal(y)
        y = y[h_depth:]  # remove the values that are used for initialization
        try:  # to catch the exception when OVERFLOW
            nadpcmc_tx, nadpcmc_rx = iterate(nadpcmc_tx, nadpcmc_rx, y)
        except Exception as e:
            print(f'Overflow: {e}')
            continue
        __error = np.abs(nadpcmc_rx.y_recreated[:-1] - y)
        plt.plot(__error, label=f'K_v={k_v}', linewidth=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.grid(True, which='both', linestyle='-', linewidth=0.2, color='grey')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.tight_layout()
    plt.savefig(
        Path(save_dir) /
        f'k_v_comparison.n_bits={n_bits}.n_iter={n_iter}.h_depth={h_depth}.alpha={alpha}.with_noise={with_noise}.png'
    )
    # plt.show()
    plt.close()


def run(
        n_iter: int,
        h_depth: int,
        n_bits: int,
        alpha: float = 1e-09,
        k_v: float = 1e-02,
        with_noise: bool = False,
        save_dir: str = './output'
):
    nadpcmc_tx = NADPCMC(n_iter, h_depth, n_bits, alpha, k_v)
    nadpcmc_rx = copy.deepcopy(nadpcmc_tx)
    f = nadpcmc_tx.generate_signal(seed=RANDOM_SEED, with_noise=with_noise)
    nadpcmc_tx.init_with_signal(f)
    nadpcmc_rx.init_with_signal(f)
    f = f[h_depth:]  # remove the values that are used for initialization
    nadpcmc_tx, nadpcmc_rx = iterate(nadpcmc_tx, nadpcmc_rx, f, save_dir=save_dir)
    plot(nadpcmc_tx, nadpcmc_rx, f, save_dir=save_dir)
    return nadpcmc_tx, nadpcmc_rx, f


if __name__ == '__main__':
    @dataclasses.dataclass
    class Config:
        n_iter: int = 1000
        h_depth: int = 10
        n_bits: int = 8
        alpha: float = 1e-09
        k_v: float = 1e-02
        with_noise: bool = False
        save_dir: str = './output'


    configs = [
        Config(n_iter=1000, h_depth=3, n_bits=16, alpha=1e-09, k_v=1e-02, with_noise=False),
        Config(n_iter=1000, h_depth=3, n_bits=16, alpha=1e-09, k_v=1e-02, with_noise=True),
        Config(n_iter=1000, h_depth=3, n_bits=12, alpha=1e-09, k_v=1e-02, with_noise=False),
        Config(n_iter=1000, h_depth=3, n_bits=12, alpha=1e-09, k_v=1e-02, with_noise=True),
        Config(n_iter=1000, h_depth=3, n_bits=8, alpha=1e-09, k_v=1e-02, with_noise=True),
        Config(n_iter=1000, h_depth=3, n_bits=8, alpha=1e-09, k_v=1e-02, with_noise=True),
    ]

    for config in configs:
        run(
            config.n_iter,
            config.h_depth,
            config.n_bits,
            config.alpha,
            config.k_v,
            config.with_noise,
            save_dir=config.save_dir
        )
        alphas_comparison(
            config.n_iter,
            config.h_depth,
            config.n_bits,
            config.k_v,
            config.with_noise,
            save_dir=config.save_dir
        )
        k_vs_comparison(
            config.n_iter,
            config.h_depth,
            config.n_bits,
            config.alpha,
            config.with_noise,
            save_dir=config.save_dir
        )
