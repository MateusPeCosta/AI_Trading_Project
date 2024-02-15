import ambiente
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

data = pd.read_csv('data/META_DATA.csv')

data['Close/Last'] = data['Close/Last'].str.replace('$', '', regex=False).astype(float)
data['Open'] = data['Open'].str.replace('$', '', regex=False).astype(float)
data['High'] = data['High'].str.replace('$', '', regex=False).astype(float)
data['Low'] = data['Low'].str.replace('$', '', regex=False).astype(float)

data['Date'] = data['Date'].astype(str)

data = data.iloc[::-1].reset_index(drop=True)

env = ambiente.StockTradingEnvironment(data) 

env = DummyVecEnv([lambda: env])

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

profits = 0  

obs = env.reset()
for _ in range(len(data)):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break
    
    current_price = env.envs[0].dataset.iloc[env.envs[0].current_step]['Close/Last']
    previous_price = env.envs[0].dataset.iloc[env.envs[0].current_step - 1]['Close/Last']
    profits += (current_price - previous_price) * env.envs[0].total_sales

print("Lucro total obtido pelo modelo:", profits)
