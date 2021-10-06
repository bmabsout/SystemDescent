{

  description = "A reproducible environment for learning certifiable controllers";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/6120ac5cd201f6cb593d1b80e861be0342495be9";
    mach-nix.url = github:DavHau/mach-nix;
    # gym-pybullet-drones.url = "github:utiasDSL/gym-pybullet-drones";
    gym-pybullet-drones.url = "path:./gym-pybullet-drones";
    gym-pybullet-drones.flake = false;
  };

  outputs = { self, flake-utils, nixpkgs, mach-nix, gym-pybullet-drones }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = (import (nixpkgs) { config = {allowUnfree = true;}; system =
              "x86_64-linux";
                  });
          extensions = (with pkgs.vscode-extensions; [
            ms-python.python
            ms-toolsai.jupyter
            jnoortheen.nix-ide
          ]);

          mach-nix-utils = import mach-nix {
            inherit pkgs;
            python = "python39";
          };

          vscodium-with-extensions = pkgs.vscode-with-extensions.override {
            vscode = pkgs.vscodium;
            vscodeExtensions = extensions;
          };

          python-with-deps = mach-nix-utils.mkPython {
            _.box2d-py = { nativeBuildInputs.add = with pkgs; [ swig ]; }; 
            providers.pyglet="nixpkgs";
            providers.pygame="nixpkgs";
            providers.pybullet="nixpkgs";

            requirements= ''
              numpy
              tensorflow
              scipy
              gym
              tqdm
              mypy
              matplotlib
              box2d-py
              pygame
              pybullet
            '';
            packagesExtra=[
              (mach-nix-utils.buildPythonPackage {
                src = gym-pybullet-drones;
              })
            ];
          };
      in {
        devShell = pkgs.mkShell {
          buildInputs=[
            python-with-deps
            vscodium-with-extensions
          ];
        };
      }
    );
}
