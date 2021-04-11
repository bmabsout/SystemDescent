{
  description = "A very basic flake";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/34f85de51bbc74595e63b22ee089adbb31f7c7a2";
  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system};
          extensions = (with pkgs.vscode-extensions; [
            bbenoist.Nix
            ms-python.python
          ]);
          vscodium-with-extensions = pkgs.vscode-with-extensions.override {
            vscode = pkgs.vscodium;
            vscodeExtensions = extensions;
          };
      in {
        devShell = pkgs.mkShell {
          buildInputs = [
            (pkgs.python37.withPackages (ps: with ps; 
              [ numpy tqdm matplotlib scipy gym tensorflow_2 ])
            )
            vscodium-with-extensions
          ];
        };
      }
    );
}
