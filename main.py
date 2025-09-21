import argparse
from vdfine.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DFINE")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--env", type=str, default="Pendulum-v1", help="name of the gym environment")
    parser.add_argument("--log-dir", type=str, default="log", help="logging directory")
    parser.add_argument("--x-dim", type=int, default=30, help="x(state) dimension")
    parser.add_argument("--a-dim", type=int, default=100, help="a(intermediate state) dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden layer dimension for encoder and decoder")
    parser.add_argument("--min-var", type=float, default=0.01, help="minimum var for states")
    parser.add_argument("--dropout-p", type=float, default=0.4, help="dropout ratio for encoder and decoder")
    parser.add_argument("--test-interval", type=int, default=10, help="test interval")
    parser.add_argument("--action-repeat", type=int, default=2, help="action repeat")
    parser.add_argument("--num-test-envs", type=int, default=3, help="number of envs used for testing")
    parser.add_argument("--buffer-capacity", type=int, default=50000, help="capacity of experience buffer")
    parser.add_argument("--seed-episodes", type=int, default=5, help="number of seed episodes (for initial data collection)")
    parser.add_argument("--all-episodes", type=int, default=1000, help="number of all episodes")
    parser.add_argument("--collect-interval", type=int, default=50, help="number of updates between data collection steps")
    parser.add_argument("--chunk-length", type=int, default=50, help="length of chunks used for the update step")
    parser.add_argument("--overshoot-d", type=int, default=3, help="latent overshooting distance")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--planning-horizon", type=int, default=12, help="planning horizon used for MPC")
    parser.add_argument("--action-noise-std", type=float, default=0.3, help="std of randomness added to actions for exploration")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon for optimizer")
    parser.add_argument("--clip-grad-norm", type=float, default=1000.0, help="clip gradients to this value")
    parser.add_argument("--kl-free-nats", type=float, default=3.0, help="free nats for kl term")
    parser.add_argument("--a-free-nats", type=float, default=0.0, help="free nats for a logratio term")
    parser.add_argument("--kl-beta", type=float, default=1.0, help="kl term weight in the loss")
    parser.add_argument("--a-beta", type=float, default=1.0, help="a logratio term weight in the loss")
    parser.add_argument("--disable-gpu", action="store_true", default=False, help="disable using gpu")

    args = parser.parse_args()
    
    train(args=args)