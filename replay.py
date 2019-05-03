from agent import DQNAgent
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrappers import wrapper


# Build env
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
env = wrapper(env)

# Parameters
states = (84, 84, 4)
actions = env.action_space.n

# Agent:q
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)

# Replay
agent.replay(env=env, model_path='./models/final-vm-1', n_replay=1, plot=True)
