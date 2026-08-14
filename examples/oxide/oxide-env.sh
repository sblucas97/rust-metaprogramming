# Environment for building the cuda-oxide benchmarks on this machine, where
# the CUDA toolkit is the apt one (headers in /usr/include, no /usr/local/cuda)
# and LLVM 21 + clang live in a user-local unpack of the official release
# tarball (no system LLVM 21 packages available). Source this before
# `cargo oxide build` -- scripts/run_bench.sh does it for you.

# Apt CUDA layout: cuda.h lives in /usr/include, so the toolkit root is /usr.
export CUDA_HOME="${CUDA_HOME:-/usr}"

# Apt puts libdevice outside the roots cuda-oxide probes.
export CUDA_OXIDE_LIBDEVICE="${CUDA_OXIDE_LIBDEVICE:-/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc}"

# User-local LLVM 21 (official release tarball unpacked at ~/.local/llvm21):
# llc for the PTX pipeline, clang/libclang for bindgen in cuda-bindings.
LLVM21="${LLVM21:-$HOME/.local/llvm21}"
export CUDA_OXIDE_LLC="${CUDA_OXIDE_LLC:-$LLVM21/bin/llc}"
export CLANG_PATH="${CLANG_PATH:-$LLVM21/bin/clang}"
export LIBCLANG_PATH="${LIBCLANG_PATH:-$LLVM21/lib}"
# `cargo oxide doctor` (and some clang-sys probes) look for `clang` on PATH.
case ":$PATH:" in
    *":$LLVM21/bin:"*) ;;
    *) export PATH="$LLVM21/bin:$PATH" ;;
esac

# The codegen backend links -lffi, but only libffi.so.8 is installed (no
# libffi-dev). ~/.local/lib holds a libffi.so -> libffi.so.8 symlink; cc's
# LIBRARY_PATH makes the linker see it without root.
export LIBRARY_PATH="$HOME/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
