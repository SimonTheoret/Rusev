{
  description = "A rust development shell";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rust = pkgs.rust-bin.stable.latest.default.override {
              extensions = ["rust-src" "rust-analyzer" "clippy"];
              targets = [];
            };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [rust] ++ (with pkgs;[
            pkg-config
            nixfmt
            nil
            shellcheck
            shfmt
            nodePackages_latest.bash-language-server
            cargo-expand
          ]);
        };
      }
    );
}
