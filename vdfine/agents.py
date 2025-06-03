import torch
import numpy as np
from mpc import mpc
from mpc.mpc import QuadCost, LinDx
from torch.distributions import MultivariateNormal


class LQRAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        dynamics_model,
        cost_model,
        planning_horizon: int,
    ):
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.cost_model = cost_model
        self.planning_horizon = planning_horizon

        self.device = next(encoder.parameters()).device
        self.Ks, self.ks = self._compute_policy()
        self.step = 0

        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)
        )

    def __call__(self, y, u, sample: bool=False):

        """
            inputs: y_t, u_{t-1}
            outputs: planned u_t
        """

        # convert y_t to a torch tensor and add a batch dimension
        y = torch.as_tensor(y, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.dynamics_model.eval()
        
            a = self.encoder(y).sample() if sample else self.encoder(y).loc

            self.dist = self.dynamics_model.posterior_step(
                dist=self.dist,
                u=torch.as_tensor(u, device=self.device).unsqueeze(0),
                a=a,
            )

            x = self.dist.sample() if sample else self.dist.loc

            planned_u = x @ self.Ks[self.step].T + self.ks[self.step].T
        
        self.step += 1
        return np.clip(planned_u.cpu().numpy(), a_min=-1.0, a_max=1.0)
    
    def _compute_policy(self):
        x_dim, u_dim = self.dynamics_model.B.shape

        Ks = []
        ks = []

        V = torch.zeros((x_dim, x_dim), device=self.device)
        v = torch.zeros((x_dim, 1), device=self.device)

        C = torch.block_diag(self.cost_model.Q, self.cost_model.R)
        c = torch.cat([
            self.cost_model.q,
            torch.zeros((u_dim, 1), device=self.device)
        ])

        F = torch.cat((self.dynamics_model.A, self.dynamics_model.B), dim=1)
        f = torch.zeros((x_dim, 1), device=self.device)

        for _ in range(self.planning_horizon-1, -1, -1):
            Q = C + F.T @ V @ F
            q = c + F.T @ V @ f + F.T @ v
            Qxx = Q[:x_dim, :x_dim]
            Qxu = Q[:x_dim, x_dim:]
            Qux = Q[x_dim:, :x_dim]
            Quu = Q[x_dim:, x_dim:]
            qx = q[:x_dim, :]
            qu = q[x_dim:, :]

            K = - torch.linalg.pinv(Quu) @ Qux
            k = - torch.linalg.pinv(Quu) @ qu
            V = Qxx + Qxu @ K + K.T @ Qux + K.T @ Quu @ K
            v = qx + Qxu @ k + K.T @ qu + K.T @ Quu @ k

            Ks.append(K)
            ks.append(k)
        
        return Ks[::-1], ks[::-1]
    
    def reset(self):
        self.step = 0
        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)
        )


class MPCAgent:
    """
        action planning by the LQR method
    """
    def __init__(
        self,
        encoder,
        dynamics_model,
        cost_model,
        planning_horizon: int,
    ):
        self.encoder = encoder
        self.dynamics_model = dynamics_model
        self.cost_model = cost_model
        self.planning_horizon = planning_horizon

        self.device = next(encoder.parameters()).device

        x_dim, u_dim = self.dynamics_model.B.shape

        C = torch.block_diag(self.cost_model.Q, self.cost_model.R).repeat(
            self.planning_horizon, 1, 1, 1,
        )

        c = torch.cat([
            self.cost_model.q.reshape(1, -1),
            torch.zeros((1, u_dim), device=self.device)
        ], dim=1).repeat(self.planning_horizon, 1, 1)

        F = torch.cat((self.dynamics_model.A, self.dynamics_model.B), dim=1).repeat(
            self.planning_horizon, 1, 1, 1
        )
        f = torch.zeros((1, x_dim), device=self.device).repeat(
            self.planning_horizon, 1, 1
        )

        self.quadcost = QuadCost(C, c)
        self.lindx = LinDx(F, f)

        self.planner = mpc.MPC(
            n_batch=1,
            n_state=x_dim,
            n_ctrl=u_dim,
            T=self.planning_horizon,
            u_lower=-1.0,
            u_upper=1.0,
            lqr_iter=50,
            backprop=False,
            exit_unconverged=False,
        )

        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)
        )

    def __call__(self, y, u, sample: bool=False):

        """
            inputs: y_t, u_{t-1}
            outputs: planned u_t
        """

        # convert y_t to a torch tensor and add a batch dimension
        y = torch.as_tensor(y, device=self.device).unsqueeze(0)

        # no learning takes place here
        with torch.no_grad():
            self.encoder.eval()
            self.dynamics_model.eval()
        
            a = self.encoder(y).sample() if sample else self.encoder(y).loc

            self.dist = self.dynamics_model.posterior_step(
                dist=self.dist,
                u=torch.as_tensor(u, device=self.device).unsqueeze(0),
                a=a,
            )

            x = self.dist.sample() if sample else self.dist.loc
            planned_x, planned_u, _ = self.planner(
                x,
                self.quadcost,
                self.lindx
            )
        
        return np.clip(planned_u.squeeze(1).cpu().numpy(), a_min=-1.0, a_max=1.0)
    
    def reset(self):
        self.dist = MultivariateNormal(
            loc=torch.zeros((1, self.dynamics_model.x_dim), device=self.device),
            covariance_matrix=torch.eye(self.dynamics_model.x_dim, device=self.device).unsqueeze(0)
        )