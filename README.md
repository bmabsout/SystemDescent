# Installation instructions

## On Ubuntu:

`apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig`

`pip install -r requirements.txt`

or install nix and run `nix-shell`

## On Nixos:
	run `nix-shell`


# Running training
	`python src/GAN_dynamics_learning.py --env_name CustomBipedalWalker-v0 --learning_rate 1e-3 --direct --latent_size 0`

# Running examples
	`python src/test.py --random_actor`