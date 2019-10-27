# The replay buffer class is responsible for storing transitions.

from collections import deque
import numpy as np
import random 

class ReplayBuffer(object):

    # Construct the buffer using the deque structure
    def __init__(self, buffer_size, random_seed=123):

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    # Add transition to buffer.  If buffer has no room, pop out oldest transition.
    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    # This function will be used to generate the minibatch of transitions.  The minibatch of transitons will be used to
    # update the Q-network in the "Experimentation and Evaluation" phase of the learning cycle.
    def sample_batch(self, batch_size):
        batch = []

        # If there is less than minibatch size number of transitions available, sample all transitions
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        # Transition includes current state, action, reward, finished boolean, and next state
        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0
