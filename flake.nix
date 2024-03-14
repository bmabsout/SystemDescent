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
      devShell = forAllSystems (system: pkgs:
        let
            types-tensorflow = pkgs.python3.pkgs.buildPythonPackage rec {
              pname = "types-tensorflow";
              version = "2.15.0.20240314";
            
              src = pkgs.fetchPypi {
                  inherit pname version;
                  sha256 = "sha256-yOGgoxZsfgR5ajkZXcluIA+P4N8Hp9/FBnUjbl3YvQI=";
              };
              propagatedBuildInputs = with pkgs.python3.pkgs; [numpy types-requests types-protobuf urllib3];
            };

            python = pkgs.python3.withPackages (p: with p;[numpy pygame pybullet
              matplotlib gymnasium tensorflow tqdm keras pybox2d dill pyquaternion types-tqdm types-tensorflow]);
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
