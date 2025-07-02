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
        #config.allowUnfree=true; config.cudaSupport = true;
        #config.cudaCapabilities = [ "8.6" ];
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
                nativeBuildInputs = [ pkgs.pkgconfig pkgs.swig ];
                doCheck = false;
                format="setuptools";
              };

            sd = pkgs.python3.pkgs.buildPythonPackage rec {
                pname = "sd";
                version = "0.1.0";
              
                src = ./.;
                doCheck = false;
              
                propagatedBuildInputs = with pkgs.python3.pkgs; [
                  numpy pygame pybullet matplotlib gymnasium tensorflow keras 
                  pybox2d dill tqdm mypy pip
                ];
            };
            
        in pkgs.mkShell {
            buildInputs = [
                pkgs.nixgl.auto.nixGLDefault
                # System Z3 and other SMT solvers
                pkgs.z3
                # Essential libraries that pip packages need
                pkgs.gcc
                pkgs.stdenv.cc.cc.lib  # Provides libstdc++.so.6
                pkgs.glibc
                pkgs.pkg-config
                pkgs.cmake
                # Additional libraries for dReal
                pkgs.gmp
                pkgs.mpfr
                # Python environment with core packages
                (pkgs.python3.withPackages (p: with p;[
                    numpy pygame pybullet matplotlib gymnasium tensorflow keras 
                    tqdm sd pybox2d mypy dill pip setuptools wheel
                ]))
            ];

            shellHook = ''
                echo "🚀 SystemDescent development environment loaded!"
                echo ""
                
                # Setup local Python package directory with correct Python version detection
                PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
                export PIP_PREFIX="$PWD/.nix-pip-packages"
                export PYTHONPATH="$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages:$PYTHONPATH"
                mkdir -p "$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages"
                
                # Setup library paths for SMT solvers
                export LD_LIBRARY_PATH="$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/z3/lib:$LD_LIBRARY_PATH"
                export LD_LIBRARY_PATH="$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/cvc5.libs:$LD_LIBRARY_PATH"
                export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
                export LD_LIBRARY_PATH="${pkgs.glibc}/lib:$LD_LIBRARY_PATH"
                
                # Z3-specific environment variables
                export Z3_LIBRARY_PATH="$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/z3/lib"
                
                echo "📦 Installing SMT solver Python packages..."
                echo "   Python version: $PYTHON_VERSION"
                echo "   Install path: $PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages"
                
                # Install Z3 (usually the most reliable)
                echo "🔧 Installing Z3 solver..."
                if ! python -c "import z3" 2>/dev/null; then
                    pip install --prefix="$PIP_PREFIX" z3-solver
                fi
                
                # Install CVC5 
                echo "🔧 Installing CVC5..."
                if ! python -c "import cvc5" 2>/dev/null; then
                    pip install --prefix="$PIP_PREFIX" cvc5
                fi
                
                # Install dReal (optional, often fails)
                echo "🔧 Installing dReal (may fail, that's OK)..."
                if ! python -c "import dreal" 2>/dev/null; then
                    pip install --prefix="$PIP_PREFIX" dreal 2>/dev/null || echo "   ⚠️  dReal install failed (optional)"
                fi
                
                # Update library paths after installation
                export LD_LIBRARY_PATH="$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/z3/lib:$LD_LIBRARY_PATH"
                export LD_LIBRARY_PATH="$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/cvc5.libs:$LD_LIBRARY_PATH"
                
                echo ""
                echo "🔍 Library paths setup:"
                echo "   LD_LIBRARY_PATH includes:"
                echo "     - Z3: $PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/z3/lib"
                echo "     - CVC5: $PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/cvc5.libs"
                echo "     - System libstdc++: ${pkgs.stdenv.cc.cc.lib}/lib"
                echo "   Z3_LIBRARY_PATH: $Z3_LIBRARY_PATH"
                
                echo ""
                echo "🧪 Testing solver availability..."
                
                # Test Z3 with detailed error reporting
                echo "Testing Z3..."
                if python -c "import z3; print('✅ Z3 Python bindings working, version:', z3.get_version_string())" 2>/dev/null; then
                    Z3_STATUS="✅ Available"
                else
                    echo "Z3 import failed, trying with explicit library path..."
                    if python -c "
import builtins
builtins.Z3_LIB_DIRS = ['$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/z3/lib']
import z3
print('✅ Z3 working with explicit path, version:', z3.get_version_string())
" 2>/dev/null; then
                        Z3_STATUS="✅ Available (with explicit path)"
                        echo "Setting up Z3_LIB_DIRS permanently..."
                        export Z3_LIB_DIRS="$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/z3/lib"
                    else
                        Z3_STATUS="❌ Failed"
                        python -c "import z3" 2>&1 | head -3
                    fi
                fi
                
                # Test CVC5 with detailed error reporting
                echo "Testing CVC5..."
                if python -c "import cvc5; print('✅ CVC5 Python bindings working')" 2>/dev/null; then
                    CVC5_STATUS="✅ Available"
                else
                    CVC5_STATUS="❌ Failed"
                    python -c "import cvc5" 2>&1 | head -3
                fi
                
                # Test dReal with detailed error reporting
                echo "Testing dReal..."
                if python -c "import dreal; print('✅ dReal Python bindings working')" 2>/dev/null; then
                    DREAL_STATUS="✅ Available"
                else
                    DREAL_STATUS="⚠️  Not available"
                    python -c "import dreal" 2>&1 | head -3 || echo "   (dReal often fails to install, this is normal)"
                fi
                
                echo ""
                echo "📋 SMT Solver Status:"
                echo "   - Z3:     $Z3_STATUS"
                echo "   - CVC5:   $CVC5_STATUS"  
                echo "   - dReal:  $DREAL_STATUS"
                echo ""
                
                # Recommend the best available solver
                if python -c "import z3" 2>/dev/null || python -c "
import builtins
builtins.Z3_LIB_DIRS = ['$PIP_PREFIX/lib/python$PYTHON_VERSION/site-packages/z3/lib']
import z3
" 2>/dev/null; then
                    RECOMMENDED="z3"
                elif python -c "import cvc5" 2>/dev/null; then
                    RECOMMENDED="cvc5"  
                else
                    RECOMMENDED="z3"
                    echo "⚠️  Using system Z3 as fallback"
                fi
                
                echo "🎯 Recommended solver: $RECOMMENDED"
                echo ""
                echo "🔧 Quick test commands:"
                echo "   python -c \"import z3; print('Z3 version:', z3.get_version_string())\""
                echo "   python -c \"import cvc5; print('CVC5 available')\""
                echo ""
                echo "🚀 Start training with SMT verification:"
                echo "   python your_enhanced_script.py --solver $RECOMMENDED --verify_every 10"
                echo ""
                echo "📝 Ready for Lyapunov verification training!"
            '';
          }
        );
    };
}