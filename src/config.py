# TODO SAME CONFIG AS HALFCHEETAH

# Environment setup variables
ENV_NAME = "HalfCheetah-v5" # name of the mujoco environment to run the agent
N_ENVS = 4                  # number of environment to run in parallel
STATE_DIM = 17              # continuous space ranging [-inf, +inf], includes body part positions and velocities
ACTION_DIM = 6              # continuous space ranging [−1, 1], represents the torques applied at the hinge joints

# Training variables
N_TRAINING_STEPS = 2000000    # number of training steps the agents interacts with the environment, considering all environments
N_POLICY_ROLLOUT_STEPS = 1024 # number of policy rollout steps to run for data collection (NOTE: this amount of steps are run for each environment)
N_MINIBATCHES = 64            # size of minibatches to optimize policy and value network parameters
POLICY_UPDATE_EPOCHS = 10     # number of epochs to optimize agent after collection batch data during policy rollout
LEARNING_RATE = 3e-4          # learning rate starting value
LEARNING_RATE_END = 1e-5      # learning rate final value after linear decay annealing
CLIP_EPSILON = 0.1            # epsilon used for clipped surrogate objective function
GAMMA = 0.98                  # discount factor for late rewards
GAE_LAMBDA = 0.9              # general advantage estimation lambda 
MAX_GRADIENT_NORM = 0.6       # max gradient norm, considering all parameters as one vector, to clip gradient norm
VALUE_LOSS_COEFFICIENT = 0.3  # coefficient that weights loss of value network in the agent lsos

# History tracking and checkpoint variables
N_CHECKPOINTS = 10                 # number of checkpoints of the training history and the agent made during training
CHECKPOINT_FOLDER = './checkpoint' # directory to store checkpoints
HISTORY_FOLDER = './history'       # directory to store history dictionaries

# Miscellaneous variables
CHECK_TENSOR_SHAPES = False     # flag to toggle checking if tensors have expected shape outputs
VERBOSE = True                  # flag to toggle verbose messages for debugging purposes
FIG_PATH = './fig'              # directory to store figures, such as plots
VIDEO_FOLDER = './video'        # directory to store a video recording of the agent in the environment
VIDEO_NAME_PREFIX = 'ppo-agent' # name prefix of video recording of the agent in the environment