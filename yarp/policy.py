import numpy as np


class Policy(object):
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, s):
        return self.agent.policy(s)


class EpsilonGreedyQPolicy(Policy):
    def __init__(self, agent, epsilon):
        super(EpsilonGreedyQPolicy, self).__init__(agent)
        self.epsilon = epsilon

    def __call__(self, s):
        if np.random.uniform(0., 1.) < self.epsilon:
            return np.array([
                self.agent.action_from_native(
                    self.agent.environment.action_space.sample()) for _ in range(s.shape[0])
            ])
        else:
            return super(EpsilonGreedyQPolicy, self).__call__(s)

    def step(self, steps=1):
        self.epsilon -= self.d_epsilon * steps
        self.epsilon = max(self.epsilon, self.min_epsilon)


class AnnealedGreedyQPolicy(Policy):
    def __init__(self, agent, start_epsilon, end_epsilon, steps):
        super(AnnealedGreedyQPolicy, self).__init__(agent)
        self.epsilon = start_epsilon
        self.d_epsilon = (start_epsilon - end_epsilon) / float(steps)
        self.end_epsilon = end_epsilon

    def __call__(self, s):
        if np.random.uniform(0., 1.) < self.epsilon:
            return np.array([
                self.agent.action_from_native(
                    self.agent.environment.action_space.sample()) for _ in range(s.shape[0])
            ])
        else:
            return super(AnnealedGreedyQPolicy, self).__call__(s)

    def step(self, steps=1):
        self.epsilon -= self.d_epsilon * steps
        self.epsilon = max(self.epsilon, self.end_epsilon)
