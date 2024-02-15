import gym
import numpy as np

class StockTradingEnvironment(gym.Env):
    def __init__(self, dataset):
        super(StockTradingEnvironment, self).__init__()
        self.dataset = dataset.drop(columns=['Date']) 
        self.action_space = gym.spaces.Discrete(3) 
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        self.current_step = 0
        self.max_steps = len(dataset) - 1
        self.initial_balance = 10000 
        self.balance = self.initial_balance
        self.shares = 0  
        self.total_sales = 0 
        self.total_purchases = 0 

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.total_sales = 0
        self.total_purchases = 0
        return self._next_observation()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        else:
            done = False
        obs = self._next_observation()
        reward = self._get_reward()
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _next_observation(self):
        return self.dataset.iloc[self.current_step].values

    def _take_action(self, action):
        current_price = self.dataset.iloc[self.current_step]['Close/Last']
        if action == 0: 
            if self.balance > current_price:
                self.shares += 1
                self.balance -= current_price
                self.total_purchases += 1
        elif action == 1:
            if self.shares > 0:
                self.balance += current_price
                self.shares -= 1
                self.total_sales += 1

    def _get_reward(self):
        current_price = self.dataset.iloc[self.current_step]['Close/Last']
        previous_price = self.dataset.iloc[self.current_step - 1]['Close/Last']
        reward = 0
        if self.total_sales > 0: 
            profit = (current_price - previous_price) * self.total_sales
            if profit > 0:
                reward = profit / self.total_sales 
            else:
                reward = profit / self.total_sales 
        return reward
