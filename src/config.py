# Environment setup variables
ENV_NAME = "Hopper-v5"      # name of the mujoco environment to run the agent
N_ENVS = 4                  # number of environment to run in parallel
STATE_DIM = 11              # continuous space ranging [-inf, +inf], includes body part positions and velocities
ACTION_DIM = 3              # continuous space ranging [−1, 1], represents the torques applied at the hinge joints
HEALTHY_REWARD = 0.75          # reward given at every step agent is healthy
FORWARD_REWARD_WEIGHT = 1.25   # reward for moving foward, which is positive when agent is moving forward and negative when moving side-ways or backwards
CTRL_COST_WEIGHT = 1e-3     # negative reward to penalize agent taking large actions

# History tracking and checkpoint variables
CHECKPOINT_EPISODE_FREQUENCY = 200 # checkpoint frequency of the training history and the agent
CHECKPOINT_FOLDER = './checkpoint' # directory to store checkpoints
HISTORY_FOLDER = './history'       # directory to store history dictionaries

# Miscellaneous variables
VERBOSE = True                     # flag to toggle verbose messages for debugging purposes
FIG_PATH = './fig'                 # directory to store figures, such as plots
VIDEO_FOLDER = './video'           # directory to store a video recording of the agent in the environment
VIDEO_NAME_PREFIX = 'hopper-agent' # name prefix of video recording of the agent in the environment