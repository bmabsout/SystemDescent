# Here's an example of a pendulum's evolution over time on a parameterized lyapunov function:
[Screencast From 2024-12-09 00-29-48.webm](https://github.com/user-attachments/assets/8b208b17-cb24-4e68-9e59-0dd2b0e4c992)

# Here's an example of how the same pendulum system behaves (using parameterized lyapunov function learning):
[Screencast from 2023-08-22 02-17-25.webm](https://github.com/user-attachments/assets/0e3e52f1-282d-404e-88e3-ad8e1a6c2716)


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
