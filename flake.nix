{

  description = "A reproducible environment for learning certifiable controllers";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/6120ac5cd201f6cb593d1b80e861be0342495be9";
    mach-nix.url = github:DavHau/mach-nix;
    # gym-pybullet-drones.url = "github:utiasDSL/gym-pybullet-drones";
    gym-pybullet-drones.url = "path:./gym-pybullet-drones";
    gym-pybullet-drones.flake = false;
    
    tf2rl.url = "github:keiohta/tf2rl";
    tf2rl.flake = false;
    # sd.url = "path:./.";
    # sd.flake = false;
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let pkgs = (import (inputs.nixpkgs) { config = {allowUnfree = true;}; system =
              "x86_64-linux";
                  });
                  
          extensions = (with pkgs.vscode-extensions; [
            ms-python.python
            ms-toolsai.jupyter
            jnoortheen.nix-ide
            
          ]);

          mach-nix-utils = import inputs.mach-nix {
            inherit pkgs;
            python = "python39Full";
            # pypiDataRev = "83af05378a5ca28e8e39cbc5760686f42f880dc4";
            # pypiDataSha256= sha256:1pmdvx4dngkqiy42gmpqpldlywrarxlzhdgb7lwmflrn05fma65h;
          };

          vscodium-with-extensions = pkgs.vscode-with-extensions.override {
            vscode = pkgs.vscodium.fhs;
            vscodeExtensions = extensions;
          };

          python-with-deps = mach-nix-utils.mkPython {
            _.box2d-py = { nativeBuildInputs.add = with pkgs; [ swig ]; }; 
            providers.pyglet="nixpkgs";
            providers.pygame="nixpkgs";
            providers.pybullet="nixpkgs";
            providers.tkinter="nixpkgs";
            providers.matplotlib="nixpkgs";
            #providers.tensorflow="nixpkgs";

            requirements= builtins.readFile ./requirements.txt;
            packagesExtra=[
              # (mach-nix-utils.buildPythonPackage {
              #   src = gym-pybullet-drones;
              # })
              # ./.
              # (mach-nix-utils.buildPythonPackage {
              #   src = tf2rl;
              #   requirements=''
              #   '';
              # })
              # (mach-nix-utils.buildPythonPackage {
              #   src = sd;
              # })
            ];
            # packagesExtra = [
      	     #  /home/bmabsout/Documents/gymfc
            # ];
          };
      in {
        devShell = pkgs.mkShell {
          buildInputs=[
            vscodium-with-extensions
            pkgs.python39Packages.pip
            pkgs.python39Packages.virtualenv
            python-with-deps
          ];

        };
      }
    );
}
