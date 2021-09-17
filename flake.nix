{

  description = "A reproducible environment for learning certifiable controllers";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/ff1ea3a36c1dfafdf0490e0d144c5a0ed0891bb9";
  inputs.mach-nix = {
      url = github:DavHau/mach-nix;
  };
  outputs = { self, flake-utils, nixpkgs, mach-nix }:
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
            requirements= ''
              numpy
              tensorflow
              scipy
              gym
              tqdm
              mypy
              matplotlib
              box2d-py
            '';
          };
      in {
        devShell = pkgs.mkShell {
          buildInputs=[
            python-with-deps
            vscodium-with-extensions
            pkgs.swig
          ];
        };
      }
    );
}
