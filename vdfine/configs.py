from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = "log"
    x_dim: int = 30
    a_dim: int = 100
    hidden_dim: int = 32
    min_var: float = 1e-3
    dropout_p: float=0.4
    test_interval: int = 10
    num_updates: int = 100
    chunk_length: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    eps: float = 1e-8
    clip_grad_norm: int = 1000
    cost_reconstruction_weight: float = 1.0
    overshoot_d: int = 3
    # TODO: add weighting or free nats
    kl_free_nats: int = 3
    a_free_nats: int = 3
    kl_beta: float = 1.0
    a_beta: float = 1.0

    dict = asdict