import os
import json
import wandb
import torch
import einops
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleAction, DtypeObservation
from pathlib import Path
from tqdm import tqdm
from argparse import Namespace
from datetime import datetime
from torch.nn.utils import clip_grad_norm_
from .agents import MPCAgent
from .memory import ReplayBuffer
from .env_utils import ActionRepeatWrapper
from .models import (
    Encoder,
    Decoder,
    Dynamics,
    CostModel,
)
from torch.distributions import MultivariateNormal


def train(
    args: Namespace,
):

    # prepare logging
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(vars(args), f)

    wandb.init(
        project="Controlling from high-dimensional observations",
        name="VDFINE",
        config=vars(args),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("*",step_metric="global_step")

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # make environments
    env = gym.make(id=args.env, g=3.0)
    env = DtypeObservation(env=env, dtype=np.float32)
    env = RescaleAction(env=env, min_action=-1.0, max_action=1.0)
    env = ActionRepeatWrapper(env=env, repeat=args.action_repeat)

    # define models and optimizer
    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    encoder = Encoder(
        y_dim=env.observation_space.shape[0],
        a_dim=args.a_dim,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
        min_var=args.min_var
    ).to(device)

    decoder = Decoder(
        y_dim=env.observation_space.shape[0],
        a_dim=args.a_dim,
        hidden_dim=args.hidden_dim,
        dropout_p=args.dropout_p,
        min_var=args.min_var
    ).to(device)

    dynamics_model = Dynamics(
        x_dim=args.x_dim,
        u_dim=env.action_space.shape[0],
        a_dim=args.a_dim,
        device=device,
        min_var=args.min_var
    ).to(device)

    cost_model = CostModel(
        x_dim=args.x_dim,
        u_dim=env.action_space.shape[0],
        device=device,
        hidden_dim=args.hidden_dim,
    ).to(device)

    wandb.watch([encoder, dynamics_model, decoder, cost_model], log="all", log_freq=10)

    all_params = (
        list(encoder.parameters()) +
        list(decoder.parameters()) + 
        list(dynamics_model.parameters()) +
        list(cost_model.parameters())
    )

    optimizer = torch.optim.Adam(all_params, lr=args.lr, eps=args.eps)

    # agent
    agent = MPCAgent(
        encoder=encoder,
        dynamics_model=dynamics_model,
        cost_model=cost_model,
        planning_horizon=args.planning_horizon
    )

    # replay buffer
    buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        y_dim=env.observation_space.shape[0],
        u_dim=env.action_space.shape[0],
    )

    # collect seed episodes
    print("collecting seed episodes")
    for s in tqdm(range(1, args.seed_episodes+1)):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action=action)
            done = terminated or truncated
            buffer.push(
                y=obs,
                u=action,
                c=-reward,
                done=done
            )
            obs = next_obs


    # train and test loop
    for episode in tqdm(range(args.all_episodes)):

        # model fit
        for s in range(args.collect_interval):

            encoder.train()
            decoder.train()
            dynamics_model.train()
            cost_model.train()

            y, u, c, _ = buffer.sample(
                batch_size=args.batch_size,
                chunk_length=args.chunk_length,
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
                b=args.batch_size
            )

            # Initial distribution N(0, I)
            q_x = [MultivariateNormal(
                loc=torch.zeros((args.batch_size, args.x_dim), dtype=torch.float32, device=device),
                covariance_matrix=torch.diag_embed(torch.ones((args.batch_size, args.x_dim), device=device, dtype=torch.float32)),
            ) for _ in range(args.chunk_length)] 

            # Kalman filtering
            for t in range(1, args.chunk_length):
                q_x[t] = dynamics_model.posterior_step(
                    dist=q_x[t-1],
                    u=u[t-1],
                    a=q_a_samples[t],
                )
            
            loss1 = 0.0
            loss2 = 0.0
            loss3 = 0.0
            cost_loss = 0.0

            for t in range(args.overshoot_d+1, args.chunk_length):
                # first loss term
                y_recon = decoder(q_a_samples[t])
                loss1 += nn.MSELoss()(y_recon, y[t])

                # second loss term
                # q(x_{t-d}|a_{1:t-d}, u_{0:t-d-1})
                past_q_x = q_x[t-args.overshoot_d]

                # q(x_t|a_{1:t}, u_{0:t-1})
                current_q_x = q_x[t]

                loss2 += dynamics_model.compute_kl_loss(
                    past_q_x=past_q_x,
                    current_q_x=current_q_x,
                    u=u[t-args.overshoot_d: t],
                ).clamp(min=args.kl_free_nats).mean()

                # third loss term
                # q_a
                current_q_a = encoder(y[t])
                
                loss3 += dynamics_model.compute_logratio_loss(
                    current_q_x=current_q_x,
                    current_q_a=current_q_a,
                    current_q_a_sample=q_a_samples[t],
                ).clamp(min=args.a_free_nats).mean()

                # cost loss
                current_q_x_sample = current_q_x.rsample()
                cost_loss += nn.MSELoss()(cost_model(x=current_q_x_sample, u=u[t]), c[t])

            loss1 /= (args.chunk_length - args.overshoot_d - 1)
            loss2 /= (args.chunk_length - args.overshoot_d - 1)
            loss3 /= (args.chunk_length - args.overshoot_d - 1)
            cost_loss /= (args.chunk_length - args.overshoot_d - 1)

            loss = loss1 + args.kl_beta * loss2 + args.a_beta * loss3 + cost_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, args.clip_grad_norm)
            optimizer.step()

            global_step = episode * args.collect_interval + s
            wandb.log({
                "train/loss1": loss1.item(),
                "train/loss2": loss2.item(),
                "train/loss3": loss3.item(),
                "train/cost loss": cost_loss.item(),
                "train/total loss": loss.item(),
                "global_step": global_step,
            })
        
        # data collection
        with torch.no_grad():
            obs, info = env.reset()
            agent.reset()
            action = env.action_space.sample()
            done = False
            while not done:
                planned_actions = agent(y=obs, u=action, explore=True)
                action = planned_actions[0]
                next_obs, reward, terminated, truncated, _ = env.step(action=action)
                done = terminated or truncated
                buffer.push(
                    y=obs,
                    u=action,
                    c=-reward,
                    done=done
                )
                obs = next_obs
    
        # test
        if episode % args.test_interval == 0:
            rewards = []
            print("testing ...")
            for _ in tqdm(range(args.num_test_envs)):
                encoder.eval()
                decoder.eval()
                dynamics_model.eval()
                cost_model.eval()
                with torch.no_grad():
                    obs, info = env.reset()
                    agent.reset()
                    action = env.action_space.sample()
                    done = False
                    total_reward = 0.0
                    while not done:
                        planned_actions = agent(y=obs, u=action, explore=False)
                        action = planned_actions[0]
                        next_obs, reward, terminated, truncated, _ = env.step(action=action)
                        done = terminated or truncated
                        obs = next_obs
                        total_reward += reward
                rewards.append(total_reward)
                
            avg_mean = np.array(rewards).mean()
            wandb.log({
                "average reward": avg_mean
            })

    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(decoder.state_dict(), log_dir / "decoder.pth")
    torch.save(dynamics_model.state_dict(), log_dir / "dynamics.pth")
    torch.save(cost_model.state_dict(), log_dir / "cost_model.pth")
    wandb.finish()

    return {"model_dir": log_dir}