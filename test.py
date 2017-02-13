from time import sleep
from gym_numgrid.envs import NumGrid

env = NumGrid()
env.configure(size=(5,5), render_scale=4, draw_grid=True)
for i_episode in range(20):
    print("\n********* EPISODE", i_episode, "**********\n")
    observation = env.reset()
    for t in range(100):
        env.render()
        digit = env.digit_space.sample()
        pos = env.cursor_move(env.direction_space.sample(), distance=1)
        action = (digit, tuple(pos))
        print('action:', action)
        observation, reward, done, info = env.step(action)
        if info['out_of_bounds']:
            print("Can't get out of the world!")
        if info['digit'] is not None:
            print("digit:", info['digit'])
            if digit == info['digit']:
                print("You found the right digit!")
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        sleep(0.05)
