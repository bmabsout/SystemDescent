with import (
  (import <nixpkgs> {}).fetchFromGitHub {
    owner = "nixos";
    repo = "nixpkgs";
    rev = "f8929dce13e729357f31d5b2950cbb097744bed7";
    sha256 = "sha256:06ikqdb5038vkkyx4hi5lw4gksjjndjg7mz0spawnb1gpzhqkavs";
   }) {}; 
with python37Packages;

buildPythonPackage rec {
  name = "rl_smoothness";
  src = ./.;
  buildInputs = [ numpy matplotlib scipy tensorflow_2 gym noise mpi4py joblib pyglet 
  cloudpickle psutil tqdm seaborn ];
  nativeBuildInputs = [ git tensorflow-tensorboard ];
}
