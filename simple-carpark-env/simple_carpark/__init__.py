from gym.envs.registration import register
register(
    id='SimpleDriving-v0',
    entry_point='simple_carpark.envs:SimpleCarparkEnv'
)