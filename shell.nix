with import (
  (import <nixpkgs> {}).fetchFromGitHub {
    owner = "nixos";
    repo = "nixpkgs";
    rev = "9237a09d8edbae9951a67e9a3434a07ef94035b7";
    sha256 = "05bizymljzzd665bpsjbhxamcgzq7bkjjzjfapkl2nicy774ak4x";
   }) {}; 
with python3Packages;

buildPythonPackage rec {
  name = "rl_smoothness";
  src = ./.;
  buildInputs = [ numpy matplotlib scipy tensorflow gym noise mpi4py joblib pyglet 
  cloudpickle psutil tqdm seaborn ];
  nativeBuildInputs = [ git tensorflow-tensorboard ];
}
