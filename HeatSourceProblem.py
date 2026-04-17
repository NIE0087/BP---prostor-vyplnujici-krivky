import numpy as np
from scipy.sparse import diags, kron, eye, csc_matrix
from scipy.sparse.linalg import splu


class HeatSourceInverseProblem3D:
    """
    Inverzní problém:
    hledáme parametry zdroje theta = (x0, y0, A)
    v úloze

        -Δu = f(x,y; x0,y0,A)   v (0,1)x(0,1)
         u = 0                  na hranici

    kde f je Gaussovský zdroj.

    Cenová funkce:
        J(theta) = ||u(theta) - u_ref||^2
    """

    def __init__(
        self,
        grid_size=50,
        sigma=0.08,
        noise_level=0.01,
        true_params=(0.35, 0.70, 5.0),   # (x0, y0, A)
        random_seed=123,
    ):
        self.grid_size = grid_size
        self.sigma = sigma
        self.noise_level = noise_level
        self.true_params = true_params
        self.random_seed = random_seed

        # síť v oblasti (0,1)x(0,1)
        self.x = np.linspace(0.0, 1.0, self.grid_size)
        self.y = np.linspace(0.0, 1.0, self.grid_size)
        self.h = 1.0 / (self.grid_size - 1)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # počet vnitřních bodů
        self.n_inner = self.grid_size - 2
        self.shape_inner = (self.n_inner, self.n_inner)

        # matice diskrétního Laplaceova operátoru
        self.Amat = self._build_poisson_matrix()
        self.A_lu = splu(self.Amat)

        # cache pro už spočtené hodnoty J(theta)
        self.cache = {}

        # referenční data
        self.u_ref_clean = self.solve_forward(self.true_params)
        self.u_ref = self._generate_noisy_reference()

    def _build_poisson_matrix(self):
        n = self.n_inner

        main = 4.0 * np.ones(n)
        off = -1.0 * np.ones(n - 1)

        T = diags([off, main, off], [-1, 0, 1], shape=(n, n))
        I = eye(n, format="csc")
        S = diags(
            [-1.0 * np.ones(n - 1), -1.0 * np.ones(n - 1)],
            [-1, 1],
            shape=(n, n)
        )

        A = (kron(I, T) + kron(S, I)) / (self.h ** 2)
        return csc_matrix(A)

    def source_term(self, x0, y0, A):
        return A * np.exp(
            -(((self.X - x0) ** 2 + (self.Y - y0) ** 2) / (self.sigma ** 2))
        )

    def solve_forward(self, params):
        x0, y0, A = map(float, params)

        f = self.source_term(x0, y0, A)
        rhs = f[1:-1, 1:-1].reshape(-1)

        u_inner = self.A_lu.solve(rhs)

        u = np.zeros((self.grid_size, self.grid_size), dtype=float)
        u[1:-1, 1:-1] = u_inner.reshape(self.shape_inner)
        return u

    def _generate_noisy_reference(self):
        rng = np.random.default_rng(self.random_seed)
        noise = (
            self.noise_level
            * np.max(np.abs(self.u_ref_clean))
            * rng.standard_normal(self.u_ref_clean.shape)
        )
        return self.u_ref_clean + noise

    def cost(self, params):
        key = tuple(round(float(p), 20) for p in params)

        if key in self.cache:
            return self.cache[key]

        u = self.solve_forward(params)
        value = np.linalg.norm(u - self.u_ref) ** 2

        self.cache[key] = value
        return value

    def parameter_error(self, x_est, y_est, A_est):
        x_true, y_true, A_true = self.true_params
        return np.sqrt(
            (x_est - x_true) ** 2 +
            (y_est - y_true) ** 2 +
            (A_est - A_true) ** 2
        )