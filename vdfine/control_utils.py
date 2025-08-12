import torch


def solve_discrete_lyapunov(A, Q):
    """
        solves W = A @ W @ A.T + Q
        using systems of equations
    """

    x_dim = A.shape[0]
    I = torch.eye(
        x_dim * x_dim,
        dtype=A.dtype,
        device=A.device
    )
    K = torch.kron(A, A)
    Q_vec = Q.reshape(-1, 1)
    W_vec = torch.linalg.solve(I - K, Q_vec)
    return W_vec.reshape(x_dim, x_dim)


def compute_gramians(A, B, C):
    Wc = solve_discrete_lyapunov(A, B @ B.T)
    Wo = solve_discrete_lyapunov(A, C.T @ C)

    return Wc, Wo

