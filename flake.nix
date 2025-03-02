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
                python312Full
                pyright
                ruff
                ruff-lsp
                isort
              ]
              ++ (with pkgs.python312Packages; [
                uv
                datasets
                fire
                jsonlines
                seqeval
              ]);
          };
        };
      }
    );
}
