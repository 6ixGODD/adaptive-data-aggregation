from dataclasses import dataclass

import numpy as np

# Import quantizer for error
import lab1_library


@dataclass
class NADPCMC:
    """
    Data block for NADPCMC algorithm

    Attributes:
        # n (int): total length of the simulation (number of samples/iterations)
        h_depth (int): number of history elements in \phi and corresponding coefficients (length of vectors)
        n_bits (int): number of bits to be transmitted (resolution of encoded error value)
        phi (np.ndarray): vector of vectors of samples history (reproduced!!)
                          first index = iteration; second index = current time vector element
        theta (np.ndarray): vector of vectors of coefficients
                            first index = iteration; second index = current time vector element
        y_hat (np.ndarray): vector of all predicted (from = theta * phi + k_v * eq)
        e (np.ndarray): exact error between the sample and the predicted value (y_hat)
        eq (np.ndarray): quantized value of error (see n_bits!!)
        y_recreated (np.ndarray): vector of all recreated/regenerated samples (used in the prediction!!)

    """
    # n: int
    h_depth: int
    n_bits: int
    phi: np.ndarray
    theta: np.ndarray
    y_hat: np.ndarray
    e: np.ndarray
    eq: np.ndarray
    y_recreated: np.ndarray


@dataclass
class HybParam:
    """
    Hybrid parameters for NADPCMC algorithm

    Attributes:
        alpha (int): adaptation gain / learning rate
        k_v (int): step size for error correction

    """
    alpha: float
    k_v: float


# Global parameters. if alpha is too high, it will diverge (thetas will be NaN or Inf)
hyb_param = HybParam(alpha=1e-9, k_v=1e-1)


def _enqueue(queue: np.ndarray, value) -> np.ndarray:
    """
    Enqueue value to the queue in the first position

    Args:
        queue (np.ndarray): queue to be updated
        value (any): value to be enqueued

    Returns:
        np.ndarray: updated queue

    """
    q = np.roll(queue, 1)  # Shift all elements to the right
    q[0] = value
    return q


def init(n: int, h_depth: int, n_bits: int) -> NADPCMC:
    n += 1  # for including the initial values of phi and theta in index 0
    data_block = NADPCMC(
        h_depth=h_depth,
        n_bits=n_bits,
        phi=np.zeros((n, h_depth)),
        theta=np.zeros((n, h_depth)),
        y_hat=np.zeros(n),
        e=np.zeros(n),
        eq=np.zeros(n),
        y_recreated=np.zeros(n)
    )
    # data_block.theta[0] = np.random.rand(h_depth)
    return data_block


def init_params(data_block: NADPCMC, y: float):
    """
    Initialize phi, y_hat, y_recreated with the first value of the signal

    Args:
        data_block (NADPCMC): data block with all necessary data
        y (float): first value of the signal

    """
    phi = data_block.phi[0]
    data_block.phi[0] = _enqueue(phi, y)
    # data_block.phi[0] = _enqueue(phi, np.sin(y))
    data_block.y_recreated[0], data_block.y_hat[0] = y, y


def update_params(data_block: NADPCMC, k: int):
    """
    Update weights for next round (k) based on previous k-1, k-2,...

    Args:
        data_block (NADPCMC): data block with all necessary data
        k (int): current iteration

    """
    data_block.theta[k + 1] = (
            data_block.theta[k] + hyb_param.alpha * data_block.phi[k] * data_block.eq[k + 1]
    )  # TODO: `e` or `eq`. in the original paper `e^T`, how a scalar can be transposed?


def predict(data_block: NADPCMC, k: int) -> float:
    data_block.y_hat[k + 1] = (
            data_block.theta[k] @ data_block.phi[k] - hyb_param.k_v * data_block.eq[k]
    )  # TODO: `e` or `eq`. In ori paper `e` is used but in ppt `eq`
    return data_block.y_hat[k + 1]


def calculate_error(data_block: NADPCMC, k: int, real_y: float) -> float:
    data_block.e[k + 1] = real_y - data_block.y_hat[k + 1]
    data_block.eq[k + 1] = lab1_library.quantize(
        data_block.e[k + 1], data_block.n_bits
    )
    # phi_k = data_block.phi[k]
    # data_block.phi[k + 1] = _enqueue(phi_k, real_y)
    # data_block.phi[k + 1] = _enqueue(phi_k, real_y)
    return data_block.eq[k + 1]


def reconstruct(data_block, k):
    data_block.y_recreated[k + 1] = data_block.y_hat[k + 1] + data_block.eq[k + 1]
    phi_k = data_block.phi[k]
    data_block.phi[k + 1] = _enqueue(phi_k, data_block.y_recreated[k])
    # data_block.phi[k + 1] = np.sin(data_block.y_recreated[k])
    # temp = data_block.y_recreated[k+1]
    return data_block.y_recreated[k + 1]
