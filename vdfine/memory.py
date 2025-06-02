import numpy as np


class ReplayBuffer:
    """
        Replay buffer holds sample trajectories
    """
    def __init__(
        self,
        capacity: int,
        y_dim: int,
        u_dim: int,
    ):
        self.capacity = capacity

        self.y_dim = y_dim
        self.u_dim = u_dim

        self.ys = np.zeros((capacity, y_dim), dtype=np.float32)
        self.us = np.zeros((capacity, u_dim), dtype=np.float32)
        self.cs = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)

        self.index = 0
        self.is_filled = False

    def __len__(self):
        return self.capacity if self.is_filled else self.index

    def push(
        self,
        y,
        u,
        c,
        done,
    ):
        """
            Add experience (single step) to the replay buffer
        """
        self.ys[self.index] = y
        self.us[self.index] = u
        self.cs[self.index] = c
        self.done[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.is_filled = self.is_filled or self.index == 0

    def sample(
        self,
        batch_size: int,
        chunk_length: int
    ):
        done = self.done.copy()
        done[-1] = 1
        episode_ends = np.where(done)[0]

        all_indexes = np.arange(len(self))
        distances = episode_ends[np.searchsorted(episode_ends, all_indexes)] - all_indexes + 1
        valid_indexes = all_indexes[distances >= chunk_length]

        sampled_indexes = np.random.choice(valid_indexes, size=batch_size)
        sampled_ranges = np.vstack([
            np.arange(start, start + chunk_length) for start in sampled_indexes
        ])

        sampled_ys = self.ys[sampled_ranges].reshape(
            batch_size, chunk_length, self.ys.shape[1]
        )
        sampled_us = self.us[sampled_ranges].reshape(
            batch_size, chunk_length, self.us.shape[1]
        )
        sampled_cs = self.cs[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )
        sampled_done = self.done[sampled_ranges].reshape(
            batch_size, chunk_length, 1
        )

        return sampled_ys, sampled_us, sampled_cs, sampled_done