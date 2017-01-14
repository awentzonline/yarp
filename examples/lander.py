import argparse
import os
from collections import deque

import gym
import gym.wrappers
import numpy as np
from keras import backend as K
from keras.layers import (
    Activation, Convolution2D, Dense, Dropout, Flatten, GlobalAveragePooling2D,
    Input, LeakyReLU, merge, RepeatVector
)
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop

from yarp.advantage import AdvantageAggregator
from yarp.agent import QAgent
from yarp.memory import Memory
from yarp.policy import AnnealedGreedyQPolicy, EpsilonGreedyQPolicy



def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


class DiscreteToContinuousMixin(object):
    num_actions = 4  # none, up, left, right

    def action_to_native(self, action, default=0.):
        result = np.ones((2,)) * default
        if action[1]:
            result[0] = 1.
        if action[2]:
            result[1] -= 1.
        if action[3]:
            result[1] += 1
        #print action, result
        return result

    def action_from_native(self, action, default=0.):
        result = np.ones((4,)) * default
        if action[0] >= 0.5:
            result[1] = action[0]
        if action[1] < -0.5:
            result[2] = np.abs(action[1])
        elif action[1] > 0.5:
            result[3] = action[1]
        # TODO: better?
        best = np.argmax(result)
        result = (np.zeros((4,)) * default)
        result[best] = 1.
        return result

    def policy(self, states, default=0.):
        q = self.q(states)
        best = np.argmax(q, axis=1)
        result = (np.zeros((best.shape[0],4,)) * default)
        result[np.arange(best.shape[0]), best] = 1.
        #print result,'fart'
        return result


class LanderQAgent(DiscreteToContinuousMixin, QAgent):
    def build_model(self):
        obs_space = self.environment.observation_space
        input = Input(shape=self.environment.observation_space.low.shape)
        x = Dense(self.config.num_hidden)(input)
        x = LeakyReLU()(x)
        x = Dense(self.num_actions)(x)
        model = Model([input], [x])
        optimizer = RMSprop(lr=self.config.lr, rho=0.99, clipnorm=10., epsilon=0.01)
        model.compile(optimizer=optimizer, loss='mse')
        return model


class LanderAdvantageAgent(DiscreteToContinuousMixin, QAgent):
    def build_model(self):
        obs_space = self.environment.observation_space
        input = Input(shape=self.environment.observation_space.low.shape)
        x = Dense(self.config.num_hidden)(input)
        x = LeakyReLU()(x)
        v_hat = Dense(self.config.num_hidden)(x)
        v_hat = LeakyReLU()(v_hat)
        v_hat = Dense(1)(v_hat)
        a_hat = Dense(self.config.num_hidden)(x)
        a_hat = LeakyReLU()(a_hat)
        a_hat = Dense(self.num_actions)(a_hat)
        x = AdvantageAggregator()([v_hat, a_hat])
        model = Model([input], [x])
        optimizer = RMSprop(lr=self.config.lr, rho=0.99, clipnorm=10., epsilon=0.01)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def model_custom_objects(self, **kwargs):
        return super(LanderAdvantageAgent, self).model_custom_objects(
            AdvantageAggregator=AdvantageAggregator, **kwargs)


AGENT_REGISTRY = dict(
    dqn=LanderQAgent, duel=LanderAdvantageAgent
)

def print_stats(label, arr):
    if arr:
        print('{} // min: {} mean: {} max: {}'.format(
            label, np.min(arr), np.mean(arr), np.max(arr)
        ))


def main(config, api_key):
    print('creating environment')
    environment = gym.make(config.env)
    environment = gym.wrappers.Monitor(
        environment, config.monitor_path, force=True
    )
    environment.reset()

    print('creating agent')
    agent_class = AGENT_REGISTRY[config.agent]
    memory = Memory(config.memory)
    agent = agent_class(
        config, environment, memory, name=config.model_name,
        ignore_existing=config.ignore_existing,
    )
    train_policy = AnnealedGreedyQPolicy(
        agent, config.epsilon, config.min_epsilon,
        config.anneal_steps
    )
    eval_policy = EpsilonGreedyQPolicy(agent, 0.01)
    print('simulating...')
    epsilon = config.epsilon
    d_epsilon = 1. / float(config.anneal_steps) * config.epsilon
    needs_training = True
    recent_episode_rewards = deque([], 100)
    try:
        for epoch_i in range(config.epochs):
            print('epoch {} / epsilon = {}'.format(epoch_i, train_policy.epsilon))
            if needs_training:
                losses, rewards, episode_rewards = agent.train(
                    environment, train_policy, train_p=1.0, max_steps=config.learn_steps,
                    max_episodes=config.learn_episodes
                )
                train_policy.step(len(rewards))
                recent_episode_rewards += episode_rewards
                print_stats('Loss', losses)
                print_stats('All rewards', rewards)
                print_stats('Episode rewards', episode_rewards)
            else:
                print('skipping training...')
            # Evaluate
            losses, rewards, episode_rewards = agent.train(
                environment, eval_policy, train_p=0.,
                max_steps=config.sim_steps, max_episodes=config.sim_episodes
            )
            recent_episode_rewards += episode_rewards
            needs_training = np.mean(episode_rewards) < 196.
            print_stats('Loss', losses)
            print_stats('All rewards', rewards)
            print_stats('Episode rewards', episode_rewards)
            if (epoch_i + 1) % config.save_rate == 0:
                agent.save_model()
            if len(recent_episode_rewards) == 100 and np.mean(recent_episode_rewards) >= 195. and np.min(recent_episode_rewards) >= 195.:
                break
    except KeyboardInterrupt:
        pass
    environment.monitor.close()
    # if api_key:
    #     print('uploading')
    #     gym.upload('monitor-data', api_key=api_key)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('lander solver')
    arg_parser.add_argument('--discount', type=float, default=0.99)
    arg_parser.add_argument('--episodes', type=int, default=5)
    arg_parser.add_argument('--epsilon', type=float, default=1.0)
    arg_parser.add_argument('--min-epsilon', type=float, default=0.05)
    arg_parser.add_argument('--epochs', type=int, default=100)
    arg_parser.add_argument('--anneal-steps', type=int, default=30000)
    arg_parser.add_argument('--sim-steps', type=int, default=1000)
    arg_parser.add_argument('--sim-episodes', type=int, default=10)
    arg_parser.add_argument('--learn-steps', type=int, default=300)
    arg_parser.add_argument('--learn-episodes', type=int, default=100)
    arg_parser.add_argument('--ignore-existing', action='store_true')
    arg_parser.add_argument('--model-name', default='eas_agent')
    arg_parser.add_argument('--save-rate', type=int, default=10)
    arg_parser.add_argument('--num-hidden', type=int, default=32)
    arg_parser.add_argument('--memory', type=int, default=30000)
    arg_parser.add_argument('--agent', default='duel')
    arg_parser.add_argument('--lr', type=float, default=0.000625)
    arg_parser.add_argument('--monitor-path', default='monitor-data')
    arg_parser.add_argument('--env', default='LunarLanderContinuous-v2')
    config = arg_parser.parse_args()

    api_key = os.environ.get('AIGYM_API_KEY' ,'').strip()
    print('api key:' + api_key)
    main(config, api_key)
