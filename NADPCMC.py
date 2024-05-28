import numpy as np


class NADPCMC:
    def __init__(self, n_iter: int, h_depth: int, n_bits: int, alpha: float = 1e-09, k_v: float = 1e-02):
        self.h_depth = h_depth
        self.n_bits = n_bits
        self.n_iter = n_iter  # Number of iterations
        n_iter += 1  # for including the initial values of phi and theta in index 0
        self.phi = np.zeros((n_iter, h_depth))
        self.theta = np.zeros((n_iter, h_depth))
        self.y_hat = np.zeros(n_iter)
        self.e = np.zeros(n_iter)
        self.eq = np.zeros(n_iter)
        self.y_recreated = np.zeros(n_iter)
        self.with_noise = False
        self.alpha = alpha  # Learning rate
        self.k_v = k_v  # Step size for error correction

    def __str__(self):
        return f'NADPCMC(n_iter={self.n_iter}, h_depth={self.h_depth}, n_bits={self.n_bits}, alpha={self.alpha}, k_v={self.k_v})'

    def __repr__(self):
        return self.__str__()

    def init_params(self, y: float):
        """
        Initialize phi, y_hat, y_recreated with the first value of the signal

        Args:
            y (float): first value of the signal

        """
        phi = self.phi[0]
        self.phi[0] = self.__enqueue(phi, y)
        self.y_recreated[0], self.y_hat[0] = y, y

    def receive_error(self, k: int, eq: float):
        """
        Receive the quantized error from the transmitter

        Args:
            k (int): current iteration
            eq (float): quantized error

        """
        self.eq[k + 1] = eq

    def update_theta(self, k: int):
        """
        Update weights for next round (k) based on previous k-1, k-2,...

        Args:
            k (int): current iteration

        """
        self.theta[k + 1] = (
                self.theta[k] + self.alpha * self.phi[k] * self.eq[k + 1]
        )

    def predict(self, k: int) -> float:
        """
        Predict the next value of the signal based on the current weights

        Args:
            k (int): current iteration

        Returns:
            float: predicted value

        """
        self.y_hat[k + 1] = (
                self.theta[k] @ self.phi[k] - self.k_v * self.eq[k]
        )
        return self.y_hat[k + 1]

    def calculate_error(self, k: int, real_y: float) -> float:
        """
        Calculate the error and quantize it

        Args:
            k (int): current iteration
            real_y (float): real value of the signal

        Returns:
            float: quantized error

        """
        self.e[k + 1] = real_y - self.y_hat[k + 1]
        self.eq[k + 1] = self.__quantize(self.e[k + 1], self.n_bits)
        return self.eq[k + 1]

    def update_phi(self, k: int, y: float):
        """
        Update phi with the real value of the signal

        Args:
            k (int): current iteration
            y (float): updated value

        """
        phi_k = self.phi[k]
        self.phi[k + 1] = self.__enqueue(phi_k, y)

    def reconstruct(self, k: int):
        """
        Reconstruct the signal based on the quantized error

        Args:
            k (int): current iteration

        """
        self.y_recreated[k + 1] = self.y_hat[k + 1] + self.eq[k + 1]

    @staticmethod
    def __enqueue(queue: np.ndarray, value) -> np.ndarray:
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

    @staticmethod
    def __quantize(a, n_bits: int = 4) -> int:
        """
        Quantize a into integer with n_bits

        Args:
            a (float): value to be quantized
            n_bits (int): number of bits for quantization

        Returns:
            int: quantized value

        """
        max_q = (2 ** (n_bits - 1)) - 1
        min_q = -2 ** (n_bits - 1)
        if a > max_q:
            return max_q
        if a < min_q:
            return min_q
        return int(a)

    def generate_signal(self, seed: int = 42, scale: float = 10000, with_noise: bool = True) -> np.ndarray:
        """
        Generate a signal with a sin function and some noise
        Args:
            seed (int): random seed
            scale (float): scale of the noise
            with_noise (bool): add noise to the signal

        Returns:
            np.ndarray: generated signal

        """
        y = np.sin(np.linspace(0, 6 * np.pi, self.n_iter + self.h_depth)) * scale
        np.random.seed(seed)
        if with_noise:
            y += np.random.normal(0, scale / 100, self.n_iter + self.h_depth)
            self.with_noise = True
        return y

    def init_with_signal(self, y: np.ndarray):
        for k in range(self.h_depth):
            self.init_params(y[k])

