from gym.envs.registration import register

register(
    id='NumGrid-v0',
    entry_point='gym_numgrid.envs:NumGrid',
    trials=20,
    reward_threshold=0
)
