# Installation instructions

## On Ubuntu:

`apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig`

`pip install -r requirements.txt`
`pip install -e .`

or install nix and run `nix develop`

# Learning the dynamics
	`python -m sd.dynamics_learning`

# Running the test
	`python -m sd.test --random_actor`

# Running the lyapunov function and actor training
	`python -m sd.lyapunov`

# Training an RL agent
	`python -m sd.rl.sac`