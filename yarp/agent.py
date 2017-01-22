import os
from collections import deque

import numpy as np
from keras.models import load_model
from tqdm import tqdm


class QAgent(object):
    def __init__(self, config, environment, memory, name='qagent', ignore_existing=True):
        self.config = config
        self.environment = environment
        self.memory = memory
        self.name = name
        self.ignore_existing = ignore_existing
        self.setup_model()

    def setup_model(self):
        if self.ignore_existing or not self.try_load_model():
            self.model = self.build_model()
            self.target_model = self.build_model()
            self.train_target_model(tau=1.0)  # copy the random initial weights
        print self.model.summary()

    def build_model(self):
        pass

    def try_load_model(self):
        filename = self.model_filename
        if os.path.exists(filename):
            self.model = load_model(filename, custom_objects=self.model_custom_objects())
            self.target_model = load_model(filename, custom_objects=self.model_custom_objects())
            return True
        return False

    def model_custom_objects(self, **kwargs):
        return kwargs

    @property
    def model_filename(self):
        return '{}.h5'.format(self.name)

    def save_model(self):
        self.model.save(self.model_filename)

    def policy(self, states):
        q = self.q(states)
        return np.argmax(q, axis=1)

    def q(self, states, target=False):
        if target:
            model = self.target_model
        else:
            model = self.model
        return model.predict(states)

    def train(
            self, environment, policy, max_episodes=100, max_steps=np.inf,
            epsilon=0.05, train_p=0.0, hard_step_stop=False):
        last_state = None
        losses = []
        rewards = []
        episode_rewards = []
        episode_reward = 0
        num_episodes = 0
        is_running = True
        environment.reset()
        def range_until_done():
            i = 0
            while is_running:
                yield i
                i += 1
        last_action = last_native = None
        state_history = deque([], self.memory.sample_length)
        for step_i in tqdm(range_until_done()):
            terminal = step_i == max_steps - 1
            if len(state_history) < self.memory.sample_length:
                native_action = environment.action_space.sample()
                action = self.action_from_native(native_action)
            else:
                if self.memory.sample_length == 1:
                    full_state = last_state
                else:
                    full_state = np.stack(state_history)[None, ...]
                action = policy(full_state)[0]
                native_action = self.action_to_native(action)
            this_state, reward, terminal, info = environment.step(native_action)
            state_history.append(this_state)
            this_state = this_state[None, ...]

            rewards.append(reward)
            episode_reward += reward
            environment.render()

            if not last_state is None:
                self.memory.add(last_state, action, reward, terminal, this_state)
            if self.memory.size > 100 and np.random.uniform(0., 1.) < train_p:
                train_loss = self.train_from_memory()
                self.train_target_model(tau=self.config.target_update)
                losses += train_loss

            last_state = this_state
            last_action = action
            if terminal:
                environment.reset()
                episode_rewards.append(episode_reward)
                episode_reward = 0
                num_episodes += 1
                if step_i >= max_steps:
                    break
            is_running = num_episodes < max_episodes and (step_i < max_steps or not hard_step_stop)
        return losses, rewards, episode_rewards

    def train_from_memory(self, num_batches=1):
        full_history = []
        for i in range(num_batches):
            s, a, r, s2, t = self.memory.sample_batch(32)
            full_history.append(
                self.train_step(s, a, r, s2, t, discount=self.config.discount)
            )
        return full_history

    def action_from_native(self, action):
        return action

    def action_to_native(self, action):
        return action

    def train_step(self, s, a, r, s1, t, discount=0.9):
        q0 = self.q(s, target=True)
        q1 = self.q(s1, target=True)
        max_q1 = np.max(q1, axis=1)
        #original_q0 = np.copy(q0)
        #print q0.shape, a.shape, r.shape, t.shape, max_q1.shape, q0.shape, q1.shape
        q0[np.arange(q0.shape[0]), a.squeeze()] = r + np.logical_not(t) * discount * max_q1
        #print q0 - original_q0
        return self.model.train_on_batch(s, q0)

    def train_target_model(self, tau=0.001):
        '''Good article: https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html'''
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = tau * actor_weights[i] + (1 - tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
