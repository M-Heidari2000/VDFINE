import torch
import torch.nn as nn
from typing import Optional
import monotonicnetworks as lmn
from torch.distributions import MultivariateNormal


class Encoder(nn.Module):

    """
        q(a_t|y_t)
    """

    def __init__(
        self,
        y_dim: int,
        a_dim: int,
        hidden_dim: int,
        dropout_p: Optional[float] = 0.2,
        min_var: Optional[float] = 1e-3,
    ):
        super().__init__()

        self.min_var = min_var

        self.mlp_layers = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, a_dim)
        # TODO: later try to learn full covariance matrix
        self.cov_head = nn.Sequential(
            nn.Linear(hidden_dim, a_dim),
            nn.Softplus()
        )

    def forward(self, y):
        hidden = self.mlp_layers(y)
        mean = self.mean_head(hidden)
        cov = torch.diag_embed(self.cov_head(hidden) + self.min_var)
        return MultivariateNormal(loc=mean, covariance_matrix=cov)
    

class Decoder(nn.Module):
    """
        p(y_t|a_t)
    """

    def __init__(
        self,
        a_dim: int,
        y_dim: int,
        hidden_dim: Optional[int],
        dropout_p: Optional[float] = 0.2,
        min_var: Optional[float] = 1e-3,
    ):
        
        super().__init__()
        self.min_var = min_var

        self.mlp_layers = nn.Sequential(
            nn.Linear(a_dim, hidden_dim),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout_p),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, y_dim)
        self.cov_head = nn.Sequential(
            nn.Linear(hidden_dim, y_dim),
            nn.Softplus()
        )

    def forward(self, a):
        hidden = self.mlp_layers(a)
        mean = self.mean_head(hidden)
        cov = torch.diag_embed(self.cov_head(hidden) + self.min_var)
        return MultivariateNormal(loc=mean, covariance_matrix=cov)
    

class CostModel(nn.Module):
    """
        Learnable quadratic cost function in the latent space
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        device: str,
        hidden_dim: Optional[int]=16,
    ):
        
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        
        self.device = device
        self.A = nn.Parameter(
            torch.eye(x_dim, device=self.device, dtype=torch.float32),
        )
        self.B = nn.Parameter(
            torch.eye(u_dim, device=self.device, dtype=torch.float32)
        )
        self.q = nn.Parameter(
            torch.randn((x_dim, 1), device=self.device, dtype=torch.float32)
        )

        # monotonic increasing function
        self.F = lmn.MonotonicWrapper(
            nn.Sequential(
                lmn.LipschitzLinear(1, hidden_dim, kind="one-inf"),
                lmn.GroupSort(2),
                lmn.LipschitzLinear(hidden_dim, hidden_dim, kind="inf"),
                lmn.GroupSort(2),
                lmn.LipschitzLinear(hidden_dim, 1, kind="inf")
            ),
            monotonic_constraints=[1],
        ).to(device=self.device)

    @property
    def Q(self):
        return self.A @ self.A.T
    
    @property
    def R(self):
        L = torch.tril(self.B)
        diagonals = nn.functional.softplus(L.diagonal()) + 1e-4
        X = 1 - torch.eye(self.u_dim, device=self.device, dtype=torch.float32)
        L = L * X + diagonals.diag()
        return L @ L.T
    
    def forward(self, x, u):
        # x: b x
        # u: b u
        # TODO: use torch.einsum for efficieny
        cost = 0.5 * x @ self.Q @ x.T + 0.5 * u @ self.R @ u.T
        cost = cost.diagonal().unsqueeze(1) + x @ self.q
        return self.F(cost)
        

class Dynamics(nn.Module):
    
    """
        dynamics including prior rollouts and posterior inference
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        a_dim: int,
        device: str,
        min_var: float=1e-4,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.u_dim = u_dim
        self.a_dim = a_dim
        self.device = device
        self._min_var = min_var

        # Dynamics matrices
        self.M = nn.Parameter(
            torch.eye(self.x_dim, device=self.device)
        )
        self.N = nn.Parameter(
            torch.eye(self.x_dim, device=self.device)
        )
        self.d = nn.Parameter(
            torch.randn(self.x_dim, device=self.device)
        )
        self.B = nn.Parameter(
            torch.randn(self.x_dim, self.u_dim, device=self.device),
        )
        self.C = nn.Parameter(
            torch.randn(self.a_dim, self.x_dim, device=self.device)
        )

        # Transition noise covariance (diagonal)
        self.nx = nn.Parameter(
            torch.randn(self.x_dim, device=self.device)
        )
        # Observation noise covariance (diagonal)
        self.na = nn.Parameter(
            torch.randn(self.a_dim, device=device)
        )

    @property
    def A(self):
        # constructing a stable A matrix
        # softplus ensures positive entries
        d = nn.functional.softplus(self.d)
        # QR decomposition to obtain a unitary matrix
        # why sign correction of the columns?
        Q, R = torch.linalg.qr(self.M, mode="reduced")
        Q = Q @ R.diagonal().sign().diag()

        U, R2 = torch.linalg.qr(self.N, mode="reduced")
        U = U @ R2.diagonal().sign().diag()

        return U @ d.sqrt().diag() @ Q @ (1 / (1+d).sqrt()).diag() @ U.T
    
    def make_pd(self, P, eps=1e-6):
        P = 0.5 * (P + P.transpose(-1, -2))
        P = P + eps * torch.eye(P.size(-1), device=P.device)
        return P
    
    def dynamics_update(
        self,
        dist: MultivariateNormal,
        u: torch.Tensor,
    ):
        """
            infer q(x_t|a_{1:t-1}, u_{0:t-1})
            inputs:
                dist:
                    q(x_{t-1}|a_{1:t-1}, u_{0:t-2})
                    - mean: b x
                    - cov: b x x
                u:
                    - u_{t-1}
                    - b u
        """

        mean = dist.loc
        cov = dist.covariance_matrix

        Nx = torch.diag(nn.functional.softplus(self.nx) + self._min_var)    # shape: x x
        next_mean = mean @ self.A.T + u @ self.B.T
        next_cov = self.A @ cov @ self.A.T + Nx
        next_cov = self.make_pd(next_cov)

        return MultivariateNormal(loc=next_mean, covariance_matrix=next_cov)
    
    def measurement_update(
        self,
        dist: MultivariateNormal,
        a: torch.Tensor,
    ):
        """
            infer q(x_t|a_{1:t}, u_{0:t-1})
            dist:
                - q(x_t|a_{1:t-1}, u_{0:t-1})
                - mean: b x
                - cov: b x x
            a: 
                - a_t
                - b a
        """

        mean = dist.loc
        cov = dist.covariance_matrix

        Na = torch.diag(nn.functional.softplus(self.na) + self._min_var)    # shape: a a

        K = cov @ self.C.T @ torch.linalg.pinv(self.C @ cov @ self.C.T + Na)
        next_mean = mean + ((a - mean @ self.C.T).unsqueeze(1) @ K.transpose(1, 2)).squeeze(1)
        next_cov = (torch.eye(self.x_dim, device=self.device) - K @ self.C) @ cov
        next_cov = self.make_pd(next_cov)

        return MultivariateNormal(loc=next_mean, covariance_matrix=next_cov)
        
    def prior_step(        
        self,
        dist: MultivariateNormal,
        u: torch.Tensor,
    ):
        """
            p(x_t|x_{t-d}, u_{t-d:t-1})

            inputs:
                - dist: posterior distribution of of x_{t-d}
                - u: u_{t-d:t-1}
        """
        Nx = torch.diag(nn.functional.softplus(self.nx) + self._min_var)    # shape: x x

        mean = dist.loc
        cov = dist.covariance_matrix
        steps = u.shape[0]

        for d in range(0, steps):
            mean = mean @ self.A.T + u[d] @ self.B.T
            cov = self.A @ cov @ self.A.T + Nx

        return MultivariateNormal(mean, cov)
    
    def posterior_step(
        self,
        dist: MultivariateNormal,
        u,
        a,
    ):
        """
            infer q(x_t|a_{1:t}, u_{0:t-1})
            inputs:
                dist: q(x_{t-1}|a_{1:t-1}, u_{0:t-2})
                u: u_{t-1}
                a: a_t
        """

        dist = self.dynamics_update(dist=dist, u=u)
        dist = self.measurement_update(dist=dist, a=a)

        return dist
    
    def compute_a_prior(
        self,
        x
    ):
        
        Na = torch.diag(nn.functional.softplus(self.na) + self._min_var)    # shape: a a
        mean = x @ self.C.T
        cov = Na.repeat(x.shape[0], 1, 1)
        dist = MultivariateNormal(
            loc=mean,
            covariance_matrix=cov
        )

        return dist