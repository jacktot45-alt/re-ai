"""Central configuration for the simulator, cameras and neural network.

Everything is in SI units (metres, seconds, radians) unless noted.
Tweak values here to experiment; the rest of the code reads from this module.
"""

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
DT = 0.05                  # physics timestep (s) -> 20 Hz control loop
MAX_STEPS = 4000           # safety cap on steps per episode

# ----------------------------------------------------------------------------
# Vehicle (loosely Tesla-ish)
# ----------------------------------------------------------------------------
WHEELBASE = 2.9            # m, distance between axles
MAX_STEER = 0.60           # rad, max front-wheel angle (~34 deg)
ACCEL_MAX = 3.5            # m/s^2, throttle authority
BRAKE_MAX = 7.0            # m/s^2, braking authority
CAR_LENGTH = 4.7           # m, for drawing / off-road checks
CAR_WIDTH = 2.0            # m

# ----------------------------------------------------------------------------
# Cameras  (4 of them, like the request: front / left / right / rear)
# Each camera renders a small grayscale image. What the network sees IS what
# the viewer shows, so the images are intentionally low resolution.
# ----------------------------------------------------------------------------
CAM_W = 20                 # pixels wide
CAM_H = 12                 # pixels tall
CAM_NEAR = 3.0             # m, nearest ground distance the camera sees
CAM_FAR = 45.0             # m, farthest ground distance (front)
CAM_FOV = 1.30             # rad horizontal field of view (~75 deg)

# The four camera mounts: (name, yaw offset from car heading, fov, far range)
# The front camera has a wider FOV so curves stay in view through tight turns.
CAMERAS = [
    ("front", 0.0,            1.55,    CAM_FAR),
    ("left",  +1.5707963,     CAM_FOV, 28.0),
    ("right", -1.5707963,     CAM_FOV, 28.0),
    ("rear",  3.1415927,      CAM_FOV, 28.0),
]

# Down-sampling applied before the pixels enter the network (average pooling).
# 1 = network sees full camera resolution. 2 = halves each dimension, etc.
NET_POOL = 2

# In addition to the 4 cameras, the network gets the car's own speed (its
# "speedometer"), normalised by this reference. Cameras show *where* the road
# goes but not *how fast* you are going, which is needed for throttle/braking.
EGO_SPEED_REF = 30.0
# The speed is fed as several copies so it carries real weight against the
# hundreds of camera pixels (otherwise the net ignores it and the throttle
# control becomes unreliable -- the car crawls or brakes for no reason).
EGO_REPLICAS = 8

# ----------------------------------------------------------------------------
# Neural network
# ----------------------------------------------------------------------------
HIDDEN_LAYERS = [64, 32]   # sizes of hidden layers (MLP)
LEARNING_RATE = 0.04
LR_DECAY = 0.93            # multiply LR by this each epoch
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
EPOCHS = 18                # epochs for the initial behavioural-cloning fit
SEED = 7

MODEL_PATH = "selfdrive_model.json"
