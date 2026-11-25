{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [ pkgs.gcc (pkgs.python3.withPackages (ps: [ ps.numpy ps.matplotlib ps.scipy ])) ];
  OMP_NUM_THREADS = "1";
  OPENBLAS_NUM_THREADS = "1";
  MKL_NUM_THREADS = "1";
}
