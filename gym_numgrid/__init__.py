from gym.envs.registration import register

register(
    id='numgrid-v0',
    entry_point='gym_numgrid.envs:NumGrid',
)
