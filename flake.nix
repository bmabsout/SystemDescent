{

  description = "A reproducible environment for learning certifiable controllers";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  # inputs.nixpkgs.url = "nixpkgs/34f85de51bbc74595e63b22ee089adbb31f7c7a2";
  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = (import (nixpkgs) { config = {allowUnfree = true;}; system =
              "x86_64-linux";
                  });
          extensions = (with pkgs.vscode-extensions; [
            bbenoist.Nix
            ms-python.python
            ms-toolsai.jupyter
          ]);

          vscodium-with-extensions = pkgs.vscode-with-extensions.override {
            vscode = pkgs.vscodium;
            vscodeExtensions = extensions;
          };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python39.withPackages (ps: with ps; 
              [ numpy tqdm matplotlib scipy gym tensorflow_2 jupyter ])
            )
            vscodium-with-extensions
          ];
        };
      }
    );
}
