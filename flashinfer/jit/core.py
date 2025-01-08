import logging
import os
import re
from pathlib import Path
from typing import List, Union

import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock

from .env import CUTLASS_INCLUDE_DIRS as CUTLASS_INCLUDE_DIRS
from .env import FLASHINFER_CSRC_DIR as FLASHINFER_CSRC_DIR
from .env import FLASHINFER_GEN_SRC_DIR as FLASHINFER_GEN_SRC_DIR
from .env import FLASHINFER_INCLUDE_DIR as FLASHINFER_INCLUDE_DIR
from .env import FLASHINFER_JIT_DIR as FLASHINFER_JIT_DIR
from .env import FLASHINFER_WORKSPACE_DIR as FLASHINFER_WORKSPACE_DIR
from flashinfer.utils import check_hip_availability, check_cuda_availability

os.makedirs(FLASHINFER_WORKSPACE_DIR, exist_ok=True)
os.makedirs(FLASHINFER_CSRC_DIR, exist_ok=True)


class FlashInferJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        self.addHandler(logging.StreamHandler())
        log_path = FLASHINFER_WORKSPACE_DIR / "flashinfer_jit.log"
        if not os.path.exists(log_path):
            # create an empty file
            with open(log_path, "w") as f:
                pass
        self.addHandler(logging.FileHandler(log_path))
        # set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.handlers[1].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

    def info(self, msg):
        super().info("flashinfer.jit: " + msg)


logger = FlashInferJITLogger("flashinfer.jit")


def check_cuda_arch():
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")


# TODO
def check_rocm_arch():
    # TODO
    # allowed_archs = ["native", "gfx90a", "gfx940", "gfx941", "gfx942"]
    for rocm_arch_flags in torch_cpp_ext._get_rocm_arch_flags():
        # arch = str(re.search(r"\-\-offload\-arch=(\w+)", rocm_arch_flags).group(1))
        # if arch not in allowed_archs:
            # raise RuntimeError("AMD ROCm archs mismatch")
        pass


def clear_cache_dir():
    if os.path.exists(FLASHINFER_JIT_DIR):
        for file in os.listdir(FLASHINFER_JIT_DIR):
            os.remove(os.path.join(FLASHINFER_JIT_DIR, file))


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


# TODO
def remove_unwanted_pytorch_hip_flags():
    REMOVE_HIP_FLAGS = []
    for flag in REMOVE_HIP_FLAGS:
        try:
            torch_cpp_ext.COMMON_HIP_FLAGS.remove(flag)
        except ValueError:
            pass


# TODO
def remove_unwanted_pytorch_hipcc_flags():
    REMOVE_HIPCC_FLAGS = []
    for flag in REMOVE_HIPCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_HIPCC_FLAGS.remove(flag)
        except ValueError:
            pass


remove_unwanted_pytorch_nvcc_flags()
if check_hip_availability():
    remove_unwanted_pytorch_hip_flags()
    remove_unwanted_pytorch_hipcc_flags()


def load_cuda_ops(
    name: str,
    sources: List[Union[str, Path]],
    extra_cflags: List[str] = [],
    extra_cuda_cflags: List[str] = [],
    extra_ldflags=None,
    extra_include_paths=None,
    verbose=False,
):
    cflags = ["-O3", "-g"]
    cuda_cflags = ["-O3", "-g", "-std=c++17", "-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_FP8"]
    with_cuda = True
    if check_hip_availability():
        print("Setting extra flags for ROCm/HIP")
        with_cuda = None
        # cflags += ["-x", "hip"]
        # FIXME
        cflags += ["-I/opt/rocm/include"]
        cuda_cflags += ["--offload-arch=gfx942", "-ffast-math", "-I/opt/rocm/include", "-L/opt/rocm/lib", "-lamdhip64", "-D__HIP_PLATFORM_AMD__"]
    else:
        print("Setting extra flags for CUDA")
        cflags += ["-Wno-switch-bool"]
        cuda_cflags += ["--threads", "4", "-use_fast_math"]

    cflags += extra_cflags
    cuda_cflags += extra_cuda_cflags
    logger.info(f"Loading JIT ops: {name}")
    if check_cuda_availability():
        check_cuda_arch()
    elif check_hip_availability():
        check_rocm_arch()
    build_directory = FLASHINFER_JIT_DIR / name
    os.makedirs(build_directory, exist_ok=True)
    if extra_include_paths is None:
        extra_include_paths = [
            FLASHINFER_INCLUDE_DIR,
            FLASHINFER_CSRC_DIR,
        ]
        if check_hip_availability():
            extra_include_paths += []
        elif check_cuda_availability():
            extra_include_paths += CUTLASS_INCLUDE_DIRS
    lock = FileLock(FLASHINFER_JIT_DIR / f"{name}.lock", thread_local=False)
    with lock:
        module = torch_cpp_ext.load(
            name,
            list(map(lambda _: str(_), sources)),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=list(map(lambda _: str(_), extra_include_paths)),
            build_directory=build_directory,
            verbose=verbose,
            with_cuda=with_cuda,
        )
    logger.info(f"Finished loading JIT ops: {name}")
    return module
