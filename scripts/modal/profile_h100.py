"""One-shot H100 Nsight Systems profiling for PushT MPC evaluation.

Upload the checkpoint once, then run. The dataset is fetched and
decompressed server-side by ``bootstrap`` (no multi-GB local upload):

    modal volume put stable-worldmodel-cache \
        ~/.stable_worldmodel/checkpoints/models--quentinll--lewm-pusht \
        checkpoints/
    modal run scripts/modal/profile_h100.py

Stage inputs only (download dataset, skip the H100 profile):

    modal run scripts/modal/profile_h100.py::bootstrap

Download reports:

    modal volume get stable-worldmodel-cache profiles ./profiles
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = 'stable-worldmodel-h100-profile'
CACHE_VOLUME_NAME = 'stable-worldmodel-cache'
CACHE_ROOT = Path('/cache')
REMOTE_REPO_ROOT = Path('/workspace/stable-worldmodel')

DATASET_NAME = 'pusht_expert_train.h5'
DATASET_URL = (
    'https://huggingface.co/datasets/quentinll/lewm-pusht/'
    'resolve/main/pusht_expert_train.h5.zst'
)
MODEL_ID = 'quentinll/lewm-pusht'

CUDA_VERSION = '13.0.2'
NSIGHT_VERSION = '2026.3.1'
NSIGHT_DEB_URL = (
    'https://developer.download.nvidia.com/devtools/repos/'
    'ubuntu2204/amd64/'
    'NsightSystems-linux-cli-public-2026.3.1.157-3804839.deb'
)
UV_VERSION = '0.11.19'
# Resolve the newest mutually compatible Python packages when the image builds.
PYTHON_PACKAGES = (
    'decord',
    'einops',
    'gymnasium',
    'h5py',
    'hdf5plugin',
    'hydra-core',
    'hydra-submitit-launcher',
    'imageio',
    'imageio-ffmpeg',
    'lancedb',
    'loguru',
    'numpy',
    'opencv-python-headless',
    'pillow',
    'pyarrow',
    'pygame',
    'pylance',
    'pymunk',
    'requests',
    'rich',
    'shapely',
    'stable-pretraining',
    'tabulate',
    'torch',
    'torchvision',
    'tqdm',
    'transformers==4.50.0',
    'typer',
    'wandb',
    'zstandard',
)

repo_root = (
    Path(__file__).resolve().parents[2]
    if modal.is_local()
    else REMOTE_REPO_ROOT
)
source_ignore = modal.FilePatternMatcher(
    '.git/**',
    '.venv/**',
    '.cache/**',
    '.pytest_cache/**',
    '.ruff_cache/**',
    '**/__pycache__/**',
    'outputs/**',
    '*.nsys-rep',
    '*.sqlite',
)

image = (
    modal.Image.from_registry(
        f'nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04',
        add_python='3.10',
    )
    .apt_install(
        'ca-certificates',
        'curl',
        'ffmpeg',
        'libgl1',
        'libglib2.0-0',
    )
    .run_commands(
        f'curl -fsSL {NSIGHT_DEB_URL} -o /tmp/nsight-systems.deb',
        'apt-get update && '
        'apt-get install -y /tmp/nsight-systems.deb && '
        'rm /tmp/nsight-systems.deb',
        f"nsys --version | grep -F '{NSIGHT_VERSION}'",
        f'test "$CUDA_VERSION" = "{CUDA_VERSION}"',
        "nvcc --version | grep -F 'release 13.0'",
    )
    .uv_pip_install(
        *PYTHON_PACKAGES,
        uv_version=UV_VERSION,
    )
    .run_commands(
        'python -c "'
        'import re; '
        'from importlib.metadata import version; '
        f'specs={PYTHON_PACKAGES!r}; '
        "names=[re.split(r'[=<>]', s, 1)[0] for s in specs]; "
        'print({name: version(name) for name in names})'
        '"',
        'python -c "'
        'from importlib.metadata import distributions; '
        "names=sorted(d.metadata['Name'] for d in distributions()); "
        "bad=[name for name in names if name.endswith('-cu12')]; "
        'assert not bad, bad; '
        "print('CUDA 12 packages:', bad)"
        '"',
        'python -c "'
        'import torch; '
        "assert torch.version.cuda == '13.0', torch.version.cuda; "
        'print(torch.__version__, torch.version.cuda)'
        '"',
    )
    .env(
        {
            'HYDRA_FULL_ERROR': '1',
            'MUJOCO_GL': 'egl',
            'PYTHONPATH': str(REMOTE_REPO_ROOT),
            'SDL_AUDIODRIVER': 'dummy',
            'SDL_VIDEODRIVER': 'dummy',
            'STABLEWM_HOME': str(CACHE_ROOT),
        }
    )
    .workdir(REMOTE_REPO_ROOT)
    .add_local_dir(
        repo_root,
        remote_path=str(REMOTE_REPO_ROOT),
        ignore=source_ignore,
        copy=True,
    )
)

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(
    CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)


def _validate_dataset(dataset_path: Path) -> None:
    import h5py

    with h5py.File(dataset_path, 'r') as dataset:
        required = {
            'action',
            'ep_len',
            'ep_offset',
            'episode_idx',
            'pixels',
            'proprio',
            'state',
            'step_idx',
        }
        missing = required.difference(dataset.keys())
        if missing:
            raise RuntimeError(
                f'PushT dataset is missing required keys: {sorted(missing)}'
            )
        if len(dataset['ep_len']) == 0:
            raise RuntimeError('PushT dataset contains no episodes')
        row_count = len(dataset['episode_idx'])
        for key in required - {'ep_len', 'ep_offset'}:
            if len(dataset[key]) != row_count:
                raise RuntimeError(
                    f'PushT dataset column {key!r} has '
                    f'{len(dataset[key]):,} rows; expected {row_count:,}'
                )
        print(
            f'[bootstrap] validated {len(dataset["ep_len"]):,} episodes '
            f'and {row_count:,} rows'
        )


def _download_dataset(dataset_path: Path) -> None:
    """Fetch the zstd-compressed HDF5 from HF and decompress onto the volume.

    Runs server-side on the Modal CPU container, so the ~12 GiB compressed
    download never crosses the local uplink. Streams decompression in a
    single pass to a ``.part`` file, validates it, then atomically renames.
    Idempotent: a failed run leaves only the ``.part`` and reruns restart.
    """
    import requests
    import zstandard

    partial = dataset_path.with_suffix(f'{dataset_path.suffix}.part')
    print(f'[bootstrap] downloading + decompressing {DATASET_URL}')
    decompressor = zstandard.ZstdDecompressor()
    with requests.get(DATASET_URL, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with partial.open('wb') as output:
            read, written = decompressor.copy_stream(response.raw, output)
            output.flush()
            os.fsync(output.fileno())
    print(f'[bootstrap] decompressed {read:,} -> {written:,} bytes')

    _validate_dataset(partial)
    partial.replace(dataset_path)
    cache_volume.commit()


@app.function(
    image=image,
    volumes={CACHE_ROOT: cache_volume},
    cpu=4,
    memory=16 * 1024,
    max_containers=1,
    timeout=6 * 60 * 60,
    scaledown_window=2,
)
def bootstrap() -> dict[str, str]:
    """Populate the persistent dataset and checkpoint cache."""
    dataset_path = CACHE_ROOT / 'datasets' / DATASET_NAME
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        print(f'[bootstrap] dataset exists: {dataset_path}')
        _validate_dataset(dataset_path)
    else:
        _download_dataset(dataset_path)

    for legacy_path in (
        dataset_path.with_suffix(f'{dataset_path.suffix}.part'),
        CACHE_ROOT / 'downloads' / f'{DATASET_NAME}.zst',
        CACHE_ROOT / 'downloads' / f'{DATASET_NAME}.zst.part',
    ):
        if legacy_path.exists():
            print(f'[bootstrap] removing obsolete partial: {legacy_path}')
            legacy_path.unlink()

    checkpoint_dir = (
        CACHE_ROOT / 'checkpoints' / f'models--{MODEL_ID.replace("/", "--")}'
    )
    weights_path = checkpoint_dir / 'weights.pt'
    config_path = checkpoint_dir / 'config.json'
    if weights_path.exists() and config_path.exists():
        print(f'[bootstrap] checkpoint already exists: {checkpoint_dir}')
    else:
        if checkpoint_dir.exists():
            print('[bootstrap] removing incomplete checkpoint download')
            shutil.rmtree(checkpoint_dir)
        print(f'[bootstrap] downloading checkpoint: {MODEL_ID}')
        from stable_worldmodel.wm.utils import load_pretrained

        model = load_pretrained(MODEL_ID)
        del model

    cache_volume.commit()
    return {
        'dataset': str(dataset_path),
        'checkpoint': str(checkpoint_dir),
    }


def _check_nsight_environment(nsys: str) -> None:
    status = subprocess.run(
        [nsys, 'status', '--environment'],
        check=False,
        capture_output=True,
        text=True,
    )
    output = f'{status.stdout}\n{status.stderr}'.strip()
    print(output)

    process_tree_ok = re.search(
        r'CPU Profiling Environment \(process-tree\):\s*OK',
        output,
    )
    if status.returncode != 0 or process_tree_ok is None:
        raise RuntimeError(
            'Nsight process-tree CPU sampling is unavailable on this Modal '
            'host. The requested --sample=process-tree --backtrace=dwarf '
            'profile cannot run safely.\n'
            f'{output}'
        )


def _copy_profile_outputs(
    report_dir: Path,
    persistent_dir: Path,
) -> list[str]:
    persistent_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source in sorted(report_dir.iterdir()):
        if source.is_file():
            destination = persistent_dir / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination))
    cache_volume.commit()
    return copied


@app.function(
    image=image,
    volumes={CACHE_ROOT: cache_volume},
    gpu='H100!',
    cpu=8,
    memory=32 * 1024,
    max_containers=1,
    timeout=4 * 60 * 60,
    scaledown_window=2,
)
def profile() -> list[str]:
    """Run the requested evaluation under Nsight Systems once."""
    import torch

    device_name = torch.cuda.get_device_name(0)
    print(f'[profile] CUDA device: {device_name}')
    if 'H100' not in device_name.upper():
        raise RuntimeError(f'Expected an H100, got {device_name!r}')

    nsys = shutil.which('nsys')
    if nsys is None:
        raise RuntimeError('nsys is not installed in the Modal image')
    print(
        subprocess.run(
            [nsys, '--version'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    _check_nsight_environment(nsys)

    dataset_path = CACHE_ROOT / 'datasets' / DATASET_NAME
    if not dataset_path.exists():
        raise RuntimeError(
            f'Dataset not found at {dataset_path}; run bootstrap first'
        )

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    report_name = f'cem-before-{timestamp}'
    report_dir = Path('/tmp/swm-profiles') / report_name
    report_dir.mkdir(parents=True)
    report_base = report_dir / report_name
    report_path = report_base.with_suffix('.nsys-rep')
    sqlite_path = report_base.with_suffix('.sqlite')

    command = [
        nsys,
        'profile',
        '--trace=cuda,nvtx,osrt',
        '--pytorch=autograd-nvtx',
        '--sample=process-tree',
        '--backtrace=dwarf',
        '--cudabacktrace=kernel',
        '--python-backtrace=cuda',
        '-o',
        str(report_base),
        sys.executable,
        'scripts/plan/eval_wm.py',
        f'policy={MODEL_ID}',
        f'eval.dataset_name={DATASET_NAME}',
        'eval.num_eval=5',
        'solver.batch_size=5',
        'bf16=true',
        'compile=true',
    ]
    print(f'[profile] running: {" ".join(command)}')

    profile_error = None
    export_error = None
    try:
        subprocess.run(
            command,
            check=True,
            cwd=REMOTE_REPO_ROOT,
            env={**os.environ, 'HYDRA_FULL_ERROR': '1'},
        )
    except subprocess.CalledProcessError as error:
        profile_error = error
    finally:
        if report_path.exists() and not sqlite_path.exists():
            try:
                subprocess.run(
                    [
                        nsys,
                        'export',
                        '--type=sqlite',
                        '--force-overwrite=true',
                        f'--output={sqlite_path}',
                        str(report_path),
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as error:
                export_error = error

        persistent_dir = CACHE_ROOT / 'profiles' / report_name
        copied = _copy_profile_outputs(report_dir, persistent_dir)
        print(f'[profile] persisted artifacts: {copied}')

    if profile_error is not None:
        raise profile_error
    if export_error is not None:
        raise export_error
    if not report_path.exists():
        raise RuntimeError(f'Nsight did not produce {report_path}')
    return copied


@app.local_entrypoint()
def main() -> None:
    """Bootstrap persistent inputs, run one H100 profile, and exit."""
    print('[local] bootstrapping persistent inputs on CPU')
    inputs = bootstrap.remote()
    print(f'[local] inputs ready: {inputs}')

    print('[local] starting one-shot H100 profile')
    artifacts = profile.remote()
    print('[local] profile complete')
    for artifact in artifacts:
        print(f'  {artifact}')
    print(
        '\nDownload all reports with:\n'
        f'  modal volume get {CACHE_VOLUME_NAME} profiles ./profiles'
    )
