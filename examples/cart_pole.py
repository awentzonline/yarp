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
from yarp.policy import AnnealedGreedyQPolicy, GreedyQPolicy


class CartPoleAgent(QAgent):
    def build_model(self):
        obs_space = self.environment.observation_space
        input = Input(shape=self.environment.observation_space.low.shape)
        x = Dense(self.config.num_hidden)(input)
        x = LeakyReLU()(x)
        x = Dense(self.environment.action_space.n)(x)
        model = Model([input], [x])
        optimizer = RMSprop(lr=self.config.lr, rho=0.99, clipnorm=10., epsilon=0.01)
        model.compile(optimizer=optimizer, loss='mse')
        return model


class CartPoleAdvantageAgent(QAgent):
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
        a_hat = Dense(self.environment.action_space.n)(a_hat)
        x = AdvantageAggregator()([v_hat, a_hat])
        model = Model([input], [x])
        optimizer = RMSprop(lr=self.config.lr, rho=0.99, clipnorm=10., epsilon=0.01)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def model_custom_objects(self, **kwargs):
        return super(CartPoleAdvantageAgent, self).model_custom_objects(
            AdvantageAggregator=AdvantageAggregator, **kwargs)


AGENT_REGISTRY = dict(
    dqn=CartPoleAgent, duel=CartPoleAdvantageAgent
)

def print_stats(label, arr):
    if arr:
        print('{} // min: {} mean: {} max: {}'.format(
            label, np.min(arr), np.mean(arr), np.max(arr)
        ))


def main(config, api_key):
    print('creating environment')
    environment = gym.make('CartPole-v0')
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
    eval_policy = GreedyQPolicy(agent)

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
            print needs_training
            # print_stats('Loss', losses)
            # print_stats('All rewards', rewards)
            # print_stats('Episode rewards', episode_rewards)
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
    arg_parser = argparse.ArgumentParser('cart/pole solver')
    arg_parser.add_argument('--discount', type=float, default=0.95)
    arg_parser.add_argument('--episodes', type=int, default=5)
    arg_parser.add_argument('--epsilon', type=float, default=1.0)
    arg_parser.add_argument('--min-epsilon', type=float, default=0.05)
    arg_parser.add_argument('--epochs', type=int, default=100)
    arg_parser.add_argument('--anneal-steps', type=int, default=10000)
    arg_parser.add_argument('--sim-steps', type=int, default=300)
    arg_parser.add_argument('--sim-episodes', type=int, default=10)
    arg_parser.add_argument('--learn-steps', type=int, default=300)
    arg_parser.add_argument('--learn-episodes', type=int, default=100)
    arg_parser.add_argument('--num-canvases', type=int, default=3)
    arg_parser.add_argument('--ignore-existing', action='store_true')
    arg_parser.add_argument('--model-name', default='eas_agent')
    arg_parser.add_argument('--save-rate', type=int, default=10)
    arg_parser.add_argument('--num-hidden', type=int, default=16)
    arg_parser.add_argument('--memory', type=int, default=30000)
    arg_parser.add_argument('--agent', default='duel')
    arg_parser.add_argument('--lr', type=float, default=0.000625)
    arg_parser.add_argument('--monitor-path', default='monitor-data')
    config = arg_parser.parse_args()

    api_key = os.environ.get('AIGYM_API_KEY' ,'').strip()
    print('api key:' + api_key)
    main(config, api_key)
