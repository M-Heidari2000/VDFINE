import os
import json
import torch
import einops
import torch.nn as nn
import numpy as np
from pathlib import Path
from .memory import ReplayBuffer
from .configs import TrainConfig
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    CostModel,
    ZDecoder,
)
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal


def evaluate(
    backbone_dir: Path,
    z_decoder_dir: Path,
    train_replay_buffer: ReplayBuffer,
    test_replay_buffer: ReplayBuffer,
):

    with open(backbone_dir / "args.json", "r") as f:
        config = TrainConfig(**json.load(f))

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define models and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
        min_var=config.min_var
    ).to(device)

    decoder = Decoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
        min_var=config.min_var
    ).to(device)

    dynamics_model = Dynamics(
        x_dim=config.x_dim,
        u_dim=train_replay_buffer.u_dim,
        a_dim=config.a_dim,
        device=device,
        min_var=config.min_var
    ).to(device)

    cost_model = CostModel(
        x_dim=config.x_dim,
        u_dim=train_replay_buffer.u_dim,
        device=device
    ).to(device)

    z_decoder = ZDecoder(
        x_dim=config.x_dim,
        z_dim=train_replay_buffer.z_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
        min_var=config.min_var,
    ).to(device)

    # load the backbone
    encoder.load_state_dict(torch.load(backbone_dir / "encoder.pth", weights_only=True))
    dynamics_model.load_state_dict(torch.load(backbone_dir / "dynamics.pth", weights_only=True))
    decoder.load_state_dict(torch.load(backbone_dir / "decoder.pth", weights_only=True))
    z_decoder.load_state_dict(torch.load(z_decoder_dir / "z_decoder.pth", weights_only=True))

    encoder.eval()
    dynamics_model.eval()
    decoder.eval()
    z_decoder.eval()

    with torch.no_grad():

        # train and test loop
        for _ in range(config.num_updates):

            y, u, c, z, _ = train_replay_buffer.sample(
                batch_size=config.batch_size,
                chunk_length=config.chunk_length,
            )

            # convert to tensor, transform to device, reshape to time-first
            y = torch.as_tensor(y, device=device)
            y = einops.rearrange(y, "b l y -> l b y")
            u = torch.as_tensor(u, device=device)
            u = einops.rearrange(u, "b l u -> l b u")
            c = torch.as_tensor(c, device=device)
            c = einops.rearrange(c, "b l 1 -> l b 1")
            z = torch.as_tensor(z, device=device)
            z = einops.rearrange(z, "b l z -> l b z")

            q_a_samples = encoder(einops.rearrange(y, "l b y -> (l b) y")).rsample()
            q_a_samples = einops.rearrange(
                q_a_samples,
                "(l b) a -> l b a",
                b=config.batch_size
            )

            # Initial distribution N(0, I)
            q_x = [MultivariateNormal(
                loc=torch.zeros((config.batch_size, config.x_dim), dtype=torch.float32, device=device),
                covariance_matrix=torch.diag_embed(torch.ones((config.batch_size, config.x_dim), device=device, dtype=torch.float32)),
            ) for _ in range(config.chunk_length)] 

            # Kalman filtering
            for t in range(1, config.chunk_length):
                q_x[t] = dynamics_model.posterior_step(
                    dist=q_x[t-1],
                    u=u[t-1],
                    a=q_a_samples[t],
                )
            
            prediction_loss = 0.0

            for t in range(config.overshoot_d+1, config.chunk_length):
                # q(x_{t-d}|a_{1:t-d}, u_{0:t-d-1})
                past_q_x = q_x[t-config.overshoot_d]
                # p(x_t|x_{t-d, u_{t-d:t-1}})
                current_p_x = dynamics_model.prior_step(
                    dist=past_q_x,
                    u=u[t-config.overshoot_d: t]
                )
                current_p_x_sample = current_p_x.sample()