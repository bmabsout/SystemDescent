{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixgl.url = "github:guibou/nixGL";
    nixgl.inputs.nixpkgs.follows = "nixpkgs";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = {self, nixpkgs, nixgl, ... }@inp:
    let
      l = nixpkgs.lib // builtins;
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      forAllSystems = f: l.genAttrs supportedSystems
        (system: f system (import nixpkgs {inherit system;
        overlays=[nixgl.overlay]; 
        # config.allowUnfree=true; config.cudaSupport = true;
        # config.cudaCapabilities = [ "8.6" ];
        }));
      
    in
    {
      # enter this python environment by executing `nix shell .`
      devShell = forAllSystems (system: pkgs:
        let
            pybox2d = pkgs.python3.pkgs.buildPythonPackage rec {
                pname = "Box2D";
                version = "2.3.10";
              
                src = pkgs.fetchFromGitHub {
                    owner = "pybox2d";
                    repo = "pybox2d";
                    rev = "master";
                    sha256 = "a4JjUrsSbAv9SjqZLwuqXhz2x2YhRzZZTytu4X5YWX8=";
                };
                nativeBuildInputs = [ pkgs.pkg-config pkgs.swig ];
                doCheck = false;
                format="setuptools";
              };
            python = pkgs.python3.withPackages (p: with p;[numpy pygame pybullet
              matplotlib gymnasium tensorflow tqdm keras pybox2d dill pyquaternion]);
            sd = pkgs.python3.pkgs.buildPythonPackage rec {
                pname = "sd";
                version = "0.1.0";
                catchConflicts = false;
                src = ./.;
                doCheck = false;
                #format = "setuputils";
              
                propagatedBuildInputs = [
                  python
                ];
              };
            
        in pkgs.mkShell {
            buildInputs = [
                pkgs.nixgl.auto.nixGLDefault
                sd
            ];
          }
        );
    };
}
