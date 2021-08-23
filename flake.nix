{

  description = "A reproducible environment for learning certifiable controllers";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/ff1ea3a36c1dfafdf0490e0d144c5a0ed0891bb9";
  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = (import (nixpkgs) { config = {allowUnfree = true;}; system =
              "x86_64-linux";
                  });
          extensions = (with pkgs.vscode-extensions; [
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
              [ numpy tqdm matplotlib scipy gym tensorflow mypy])
            )
            vscodium-with-extensions
          ];
          # QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
        };
      }
    );
}
