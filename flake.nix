{

  description = "A reproducible environment for learning certifiable controllers";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/02336c5c5f719cd6bd4cfc5a091a1ccee6f06b1d";
    mach-nix.url = github:DavHau/mach-nix;
    nixGL.url = github:guibou/nixGL;
    nixGL.flake = false;
    tf2rl.url= github:bmabsout/tf2rl;
    tf2rl.flake= false;
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let pkgs = (import (inputs.nixpkgs) { config = {allowUnfree = true;}; system =
              "x86_64-linux";
                  });
                  
          extensions = (with pkgs.vscode-extensions; [
            ms-python.python
            ms-python.vscode-pylance
            ms-toolsai.jupyter
            jnoortheen.nix-ide
          ]);

          mach-nix-utils = import inputs.mach-nix {
            inherit pkgs;
            python = "python39Full";
            #pypiDataRev = "e18f4c312ce4bcdd20a7b9e861b3c5eb7cac22c4";
            #pypiDataSha256= "sha256-DmrRc4Y0GbxlORsmIDhj8gtmW1iO8/44bShAyvz6bHk=";
          };

          vscodium-with-extensions = pkgs.vscode-with-extensions.override {
            vscode = pkgs.vscode.fhs;
            # vscodeExtensions = extensions;
          };
          
          python-with-deps = mach-nix-utils.mkPython {
            _.box2d-py = { nativeBuildInputs.add = with pkgs; [ swig ]; }; 
	    providers = {
                pyglet="nixpkgs";
                gym="nixpkgs";
                pygame="nixpkgs";
                pybullet="nixpkgs";
                tkinter="nixpkgs";
	    };

            requirements=''
              numpy
              tk
              matplotlib
              tensorflow
              scipy
              gym
              tqdm
              mypy
              box2d-py
              noise
              pygame
              pybullet
              joblib
              pyquaternion
              pylint
              cpprb
              tensorflow-probability
              tensorflow-addons
              #GitPython>=3.1.17
            '';
            packagesExtra=[
              (mach-nix-utils.buildPythonPackage {
                 src=inputs.tf2rl;
              })
            ];
          };

	nixGLIntelScript = pkgs.writeShellScriptBin "nixGLIntel" ''
          $(NIX_PATH=nixpkgs=${inputs.nixpkgs} nix-build ${inputs.nixGL} -A nixGLIntel --no-out-link)/bin/* "$@"
        '';
        nixGLNvidiaScript = pkgs.writeShellScriptBin "nixGLNvidia" ''
          $(NIX_PATH=nixpkgs=${inputs.nixpkgs} nix-build ${inputs.nixGL} -A auto.nixGLNvidia --no-out-link)/bin/* "$@"
        '';
      in {
        devShell = pkgs.mkShell {
          buildInputs=[
            vscodium-with-extensions
	    pkgs.glxinfo
            pkgs.vscode-fhs
            pkgs.python39Packages.pip
            pkgs.python39Packages.virtualenv
	    pkgs.python39Packages.tkinter
            python-with-deps
            nixGLIntelScript
            nixGLNvidiaScript
          ];
        };
      }
    );
}
