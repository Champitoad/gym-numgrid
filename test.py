import gym
from gym.wrappers import Monitor
from gym.scoreboard.scoring import score_from_local

from gym_numgrid.wrappers import DirectionWrapper

red = '\033[91m'
yellow = '\033[93m'
green = '\033[32m'
endc = '\033[0m'

numgrid = DirectionWrapper(gym.make('NumGrid-v0'))
experiment_path = '/tmp/numgrid-direction-random'
env = Monitor(numgrid, experiment_path, force=True)
env.configure(size=(5,5), render_scale=4, draw_grid=True)

for i_episode in range(env.spec.trials):
    print("\n********* EPISODE", i_episode, "**********\n")
    observation = env.reset()
    done = False
    while not done:
        env.render()
        digit = numgrid.env.digit_space.sample()
        direction = numgrid.direction_space.sample()
        action = (digit, direction)
        color = ''
        observation, reward, done, info = env.step(action)
        if info['out_of_bounds']:
            print(yellow + "Can't get out of the world!" + endc)
        if digit != 10:
            color = green if digit == info['digit'] else red
        print(color + 'action:', action, endc)

env.close()
print(score_from_local(experiment_path))
