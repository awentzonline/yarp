import numpy as np


class Policy(object):
    def __init__(self, agent):
        self.agent = agent

    def __call__(self, s):
        pass


class GreedyQPolicy(Policy):
    def q(self, s, use_target=False):
        if use_target:
            model = self.agent.target_model
        else:
            model = self.agent.model
        return model.predict(s)

    def __call__(self, s):
        return np.argmax(self.q(s), axis=1)


class EpsilonGreedyQPolicy(GreedyQPolicy):
    def __init__(self, agent, epsilon):
        super(EpsilonGreedyQPolicy, self).__init__(agent)
        self.epsilon = epsilon

    def __call__(self, s):
        if np.random.uniform(0., 1.) < self.epsilon:
            return np.array([
                self.agent.environment.action_space.sample() for _ in range(s.shape[0])
            ])
        else:
            return super(EpsilonGreedyQPolicy, self).__call__(s)

    def step(self, steps=1):
        self.epsilon -= self.d_epsilon * steps
        self.epsilon = max(self.epsilon, self.min_epsilon)


class AnnealedGreedyQPolicy(GreedyQPolicy):
    def __init__(self, agent, start_epsilon, end_epsilon, steps):
        super(AnnealedGreedyQPolicy, self).__init__(agent)
        self.epsilon = start_epsilon
        self.d_epsilon = (start_epsilon - end_epsilon) / float(steps)
        self.end_epsilon = end_epsilon

    def __call__(self, s):
        if np.random.uniform(0., 1.) < self.epsilon:
            return np.array([
                self.agent.environment.action_space.sample() for _ in range(s.shape[0])
            ])
        else:
            return super(AnnealedGreedyQPolicy, self).__call__(s)

    def step(self, steps=1):
        self.epsilon -= self.d_epsilon * steps
        self.epsilon = max(self.epsilon, 0.)
