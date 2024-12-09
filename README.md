# Here's an example of a pendulum's evolution over time on a parameterized lyapunov function:
[Screencast From 2024-12-09 00-29-48.webm](https://github.com/user-attachments/assets/8b208b17-cb24-4e68-9e59-0dd2b0e4c992)

# Here's an example of how the pendulum behaves (both in the NN model case and the idealized system):


[Pendulums](https://github.com/user-attachments/assets/cc5a26a3-331f-4410-8f3f-35f4ec74047c)



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
