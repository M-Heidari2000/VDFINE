import os
import json
import torch
import einops
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from .memory import ReplayBuffer
from .configs import TrainConfig
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    CostModel,
)
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal


def train(
    config: TrainConfig,
    train_replay_buffer: ReplayBuffer,
    test_replay_buffer: ReplayBuffer,
):

    # prepare logging
    log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(config.dict(), f)
    
    writer = SummaryWriter(log_dir=log_dir)

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

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(dynamics_model.parameters()) +
        list(cost_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # train and test loop
    for update in range(config.num_updates):

        # train
        encoder.train()
        decoder.train()
        dynamics_model.train()
        cost_model.train()

        y, u, c, _ = train_replay_buffer.sample(
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
        ) for _ in range(config.chunk_length) ] 

        # Kalman filtering
        for t in range(1, config.chunk_length):
            q_x[t] = dynamics_model.posterior_step(
                dist=q_x[t-1],
                u=u[t-1],
                a=q_a_samples[t],
            )
        
        loss1 = 0.0
        loss2 = 0.0
        loss3 = 0.0
        cost_loss = 0.0

        for t in range(config.overshoot_d+1, config.chunk_length):
            # first loss term
            p_y = decoder(q_a_samples[t])
            loss1 -= p_y.log_prob(y[t]).mean()

            # second loss term
            # q(x_{t-d}|a_{1:t-d}, u_{0:t-d-1})
            past_q_x_sample = q_x[t-config.overshoot_d].rsample()

            # p(x_t|x_{t-d, u_{t-d:t-1}})
            current_p_x = dynamics_model.prior(
                x_sample=past_q_x_sample,
                u=u[t-config.overshoot_d: t]
            )

            # q(x_t|a_{1:t}, u_{0:t-1})
            current_q_x = q_x[t]
            loss2 += kl_divergence(current_q_x, current_p_x).clamp(min=config.kl_free_nats).mean()

            # third loss term
            current_q_x_sample = current_q_x.rsample()
            # q_a
            current_q_a = encoder(y[t])
            # p_a
            current_p_a = dynamics_model.compute_a_prior(
                x=current_q_x_sample,
            )
            
            loss3 += (
                current_q_a.log_prob(q_a_samples[t])
                - current_p_a.log_prob(q_a_samples[t])
            ).clamp(min=config.a_free_nats).mean()

            cost_loss += nn.MSELoss()(cost_model(x=current_q_x_sample, u=u[t]), c[t])

        loss1 /= (config.chunk_length - config.overshoot_d - 1)
        loss2 /= (config.chunk_length - config.overshoot_d - 1)
        loss3 /= (config.chunk_length - config.overshoot_d - 1)
        cost_loss /= (config.chunk_length - config.overshoot_d - 1)

        loss = loss1 + config.kl_beta * loss2 + config.a_beta * loss3 + config.cost_reconstruction_weight * cost_loss
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()

        writer.add_scalar("train/loss1", loss1.item(), update+1)
        writer.add_scalar("train/loss2", loss2.item(), update+1)
        writer.add_scalar("train/loss3", loss3.item(), update+1)
        writer.add_scalar("train/cost_loss", cost_loss.item(), update+1)
        writer.add_scalar("train/total loss", loss.item(), update+1)
        print(f"update step: {update+1}, train_loss: {loss.item()}")

        # test
        if update % config.test_interval == 0:
            # test
            encoder.eval()
            decoder.eval()
            dynamics_model.eval()
            cost_model.eval()

            with torch.no_grad():

                y, u, c, _ = test_replay_buffer.sample(
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
                ) for _ in range(config.chunk_length) ] 

                # Kalman filtering
                for t in range(1, config.chunk_length):
                    q_x[t] = dynamics_model.posterior_step(
                        dist=q_x[t-1],
                        u=u[t-1],
                        a=q_a_samples[t],
                    )
                
                loss1 = 0.0
                loss2 = 0.0
                loss3 = 0.0
                cost_loss = 0.0

                for t in range(config.overshoot_d+1, config.chunk_length):
                    # first loss term
                    p_y = decoder(q_a_samples[t])
                    loss1 -= p_y.log_prob(y[t]).mean()

                    # second loss term
                    # q(x_{t-d}|a_{1:t-d}, u_{0:t-d-1})
                    past_q_x_sample = q_x[t-config.overshoot_d].rsample()

                    # p(x_t|x_{t-d, u_{t-d:t-1}})
                    current_p_x = dynamics_model.prior(
                        x_sample=past_q_x_sample,
                        u=u[t-config.overshoot_d: t]
                    )

                    # q(x_t|a_{1:t}, u_{0:t-1})
                    current_q_x = q_x[t]
                    loss2 += kl_divergence(current_q_x, current_p_x).clamp(min=config.kl_free_nats).mean()

                    # third loss term
                    current_q_x_sample = current_q_x.rsample()
                    # q_a
                    current_q_a = encoder(y[t])
                    # p_a
                    current_p_a = dynamics_model.compute_a_prior(
                        x=current_q_x_sample,
                    )
                    
                    loss3 += (
                        current_q_a.log_prob(q_a_samples[t])
                        - current_p_a.log_prob(q_a_samples[t])
                    ).clamp(min=config.a_free_nats).mean()

                    cost_loss += nn.MSELoss()(cost_model(x=current_q_x_sample, u=u[t]), c[t])

                loss1 /= (config.chunk_length - config.overshoot_d - 1)
                loss2 /= (config.chunk_length - config.overshoot_d - 1)
                loss3 /= (config.chunk_length - config.overshoot_d - 1)
                cost_loss /= (config.chunk_length - config.overshoot_d - 1)

                loss = loss1 + config.kl_beta * loss2 + config.a_beta * loss3 + config.cost_reconstruction_weight * cost_loss

                writer.add_scalar("test/loss1", loss1.item(), update+1)
                writer.add_scalar("test/loss2", loss2.item(), update+1)
                writer.add_scalar("test/loss3", loss3.item(), update+1)
                writer.add_scalar("test/cost_loss", cost_loss.item(), update+1)
                writer.add_scalar("test/total loss", loss.item(), update+1)
                print(f"update step: {update+1}, test_loss: {loss.item()}")

    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(dynamics_model.state_dict(), log_dir / "dynamics.pth")
    torch.save(cost_model.state_dict(), log_dir / "cost_model.pth")

    return {"model_dir": log_dir}