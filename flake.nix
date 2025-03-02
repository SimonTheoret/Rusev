{
  description = "Hell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
      in
      {
        devShells = {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                rustc
                rustfmt
                rust-analyzer
                cargo-expand
                cargo
                python313Full
                pyright
                ruff
                ruff-lsp
                isort
              ]
              ++ (with pkgs.python313Packages; [
                uv
              ]);
          };
        };
      }
    );
}
