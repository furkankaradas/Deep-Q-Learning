import gym
import logging
import numpy as np

from agent import Agent

# Create and configure logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('training.log')
                    ])

# Creating an object
logger = logging.getLogger()


class TrainGame:
    def __init__(self, game_name, n_games, layer_dims=[128, 128]):
        self.game_name = game_name
        self.n_games = n_games
        self.environment = gym.make(self.game_name)
        self.scores = []
        self.eps_history = []
        self.layer_dims = layer_dims
        self.agent = Agent(input_dims=self.environment.observation_space.shape,
                           n_actions=self.environment.action_space.n, layer_dims=self.layer_dims)

    def training(self):
        for i in range(self.n_games):
            score = 0
            done = False
            obs = self.environment.reset()

            while done is False:
                action = self.agent.choose_action(obs)
                obs_, reward, done, info = self.environment.step(action)
                score += reward
                self.agent.learn(obs, action, reward, obs_)
                obs = obs_
            self.scores.append(score)
            self.eps_history.append(self.agent.epsilon)

            if i % 100 == 0 or i == self.n_games - 1:
                avg_score = np.mean(self.scores[-100:])
                logger.info(
                    "Episode: {} Score: {:.2f} Average Score: {:.2f} Epsilon: {:.4f}".format(i, score, avg_score,
                                                                                             self.agent.epsilon))
