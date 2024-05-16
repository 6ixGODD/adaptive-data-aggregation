from dataclasses import dataclass

import numpy as np

# Import quantizer for error
import lab1_library


# Declaring namedtuple()
# n - total length of the simulation (number of samples/iterations)
# h_depth - number of history elements in \phi and corresponding coefficients (length of vectors)
# n_bits - number of bits to be transmitted (resolution of encoded error value)
# phi   - vector of vectors of samples history (reproduced!!)
#       - first index = iteration; second index = current time vector element
# theta - vector of vectors of coefficients 
#       - first index = iteration; second index = current time vector element
# y_hat - vector of all predicted (from = theta * phi + k_v * eq)
# e - exact error between the sample and the predicted value (y_hat)
# eq - quantized value of error (see n_bits!!)
# y_recreated - vector of all recreated/regenerated samples (used in the prediction!!)
# NDPCM = namedtuple(
#     'NDPCM',
#     ['n', 'h_depth', 'n_bits', 'phi', 'theta', 'y_hat', 'e', 'eq', 'y_recreated']
# )

@dataclass
class NADPCMC:
    """
    Data block for NADPCMC algorithm

    Attributes:
        n: int                  - total length of the simulation (number of samples/iterations)
        h_depth: int            - number of history elements in \phi and corresponding coefficients (length of vectors)
        n_bits: int             - number of bits to be transmitted (resolution of encoded error value)
        phi: np.ndarray         - vector of vectors of samples history (reproduced!!)
                                - first index = iteration; second index = current time vector element
        theta: np.ndarray       - vector of vectors of coefficients
                                - first index = iteration; second index = current time vector element
        y_hat: np.ndarray       - vector of all predicted (from = theta * phi + k_v * eq)
        e: np.ndarray           - exact error between the sample and the predicted value (y_hat)
        eq: np.ndarray          - quantized value of error (see n_bits!!)
        y_recreated: np.ndarray - vector of all recreated/regenerated samples (used in the prediction!!)

    """
    n: int
    h_depth: int
    n_bits: int
    phi: np.ndarray
    theta: np.ndarray
    y_hat: np.ndarray
    e: np.ndarray
    eq: np.ndarray
    y_recreated: np.ndarray


# Define the hyb param of the NDPCM
@dataclass
class HybParam:
    """
    Hybrid parameters for NADPCMC algorithm

    Attributes:
        alpha: int  - adaptation gain, or called it learning rate :)
        k_v: int    - step size for error correction
    """
    alpha: float = 0.01
    k_v: float = 0.01


def init(n, h_depth, n_bits):
    # Adding values
    # data_block = NDPCM(
    #     n, h_depth, n_bits, np.zeros((n, h_depth)), np.zeros(
    #         (n, h_depth)
    #     ), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    # )
    # Modify initial value for any component, parameter:
    # ...

    data_block = NADPCMC(
        n=n,
        h_depth=h_depth,
        n_bits=n_bits,
        phi=np.zeros((n, h_depth)),
        theta=np.zeros((n, h_depth)),
        y_hat=np.zeros(n),
        e=np.zeros(n),
        eq=np.zeros(n),
        y_recreated=np.zeros(n)
    )
    # Initialize
    data_block.phi[0] = np.zeros(h_depth)
    data_block.theta[0] = np.zeros(h_depth)
    return data_block


def prepare_params_for_prediction(data_bloc, k):
    # Update weights for next round (k) based on previous k-1, k-2,...
    # TODO: for first iteration INITIALIZE 'phi' and 'theta'
    if k <= data_bloc.h_depth:
        data_bloc.phi[k] = np.array(
            [data_bloc.y_recreated[k],  # Add last recreated value (y(k-1)
             data_bloc.y_recreated[k - 1],  # Copy shifted from previous history (y(k-2))
             data_bloc.y_recreated[k - 2]]
        )
        data_bloc.theta[k] = np.array(
            [0.1, 0.1, 0.1]
        )
        return
    # TODO: Fill 'phi' history for 'h_depth' last elements (k-1, k-2,...)
    data_bloc.phi[k] = np.array(
        [data_bloc.y_recreated[k],  # Add last recreated value (y(k-1)
         data_bloc.y_recreated[k - 1],  # Copy shifted from previous history (y(k-2))
         data_bloc.y_recreated[k - 2]]
    )
    print("e=", data_bloc.eq[k])
    print("eT=", data_bloc.eq[k].transpose())
    # TODO: Update weights/coefficients 'theta'

    return


def predict(data_bloc, k):
    if k > 0:
        data_bloc.phi[k] = data_bloc.phi[k - 1]
    # TODO: calculate 'hat y(k)' based on (k-1) parameters
    # data_block.y_hat[k] = ...
    # if (k==1):
    # data_block.y_hat[k] = ...
    print(data_bloc.theta[k] @ data_bloc.phi[k])
    # TODO: Return prediction - fix:
    # return data_bloc.y_recreated[k-1];
    return data_bloc.phi[k][0]


def calculate_error(data_block, k, real_y):
    data_block.e[k] = real_y - data_block.y_hat[k]
    data_block.eq[k] = lab1_library.quantize(
        data_block.e[k], data_block.n_bits
    )
    return data_block.eq[k]


def reconstruct(data_block, k):
    data_block.y_recreated[k] = data_block.y_hat[k] + data_block.eq[k]
