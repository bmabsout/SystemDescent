{

  description = "A reproducible environment for learning certifiable controllers";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "nixpkgs/13cef561850fc6ee01de09f945c0e6047c26ef3c";
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
              [ numpy tqdm matplotlib scipy gym tensorflow_2 ])
            )
            vscodium-with-extensions
          ];
          # QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
        };
      }
    );
}
