"""One-shot H100 Nsight Systems profiling for PushT MPC evaluation.

Run:

    modal run scripts/modal/profile_h100.py

Download reports:

    modal volume get stable-worldmodel-cache profiles ./profiles
"""

from __future__ import annotations

import hashlib
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
    'resolve/main/pusht_expert_train.h5.zst?download=true'
)
DATASET_SHA256 = (
    '7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318'
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
    'transformers',
    'typer',
    'wandb',
    'zstandard',
)

CHUNK_SIZE = 16 * 1024 * 1024
COMMIT_INTERVAL = 1024 * 1024 * 1024

repo_root = Path(__file__).resolve().parents[2]
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
        'from importlib.metadata import version; '
        f'names={PYTHON_PACKAGES!r}; '
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_file(file) -> None:
    file.flush()
    os.fsync(file.fileno())


def _download_dataset_archive(archive: Path) -> None:
    import requests

    if archive.exists():
        print(f'[bootstrap] verifying cached archive: {archive}')
        if _sha256(archive) == DATASET_SHA256:
            return
        print(
            '[bootstrap] cached archive checksum mismatch; downloading again'
        )
        archive.unlink()

    partial = archive.with_suffix(f'{archive.suffix}.part')
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}

    print(f'[bootstrap] downloading {DATASET_URL} from byte offset {offset:,}')
    with requests.get(
        DATASET_URL,
        allow_redirects=True,
        headers=headers,
        stream=True,
        timeout=(30, 300),
    ) as response:
        if offset and response.status_code == 416:
            if _sha256(partial) == DATASET_SHA256:
                partial.replace(archive)
                cache_volume.commit()
                return
            partial.unlink()
            cache_volume.commit()
            raise RuntimeError(
                'Dataset server rejected the resume offset and the partial '
                'archive checksum is invalid; rerun bootstrap to restart it'
            )

        response.raise_for_status()
        if offset and response.status_code != 206:
            print('[bootstrap] server ignored Range; restarting download')
            offset = 0

        mode = 'ab' if offset else 'wb'
        written = offset
        next_commit = written + COMMIT_INTERVAL
        file = partial.open(mode)
        try:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                file.write(chunk)
                written += len(chunk)
                if written >= next_commit:
                    print(f'[bootstrap] downloaded {written:,} bytes')
                    _sync_file(file)
                    file.close()
                    cache_volume.commit()
                    file = partial.open('ab')
                    next_commit = written + COMMIT_INTERVAL
        finally:
            if not file.closed:
                _sync_file(file)
                file.close()
            cache_volume.commit()

    partial.replace(archive)
    cache_volume.commit()

    print(f'[bootstrap] verifying SHA256 for {archive}')
    actual_sha256 = _sha256(archive)
    if actual_sha256 != DATASET_SHA256:
        archive.unlink()
        cache_volume.commit()
        raise RuntimeError(
            'PushT archive checksum mismatch: '
            f'expected {DATASET_SHA256}, got {actual_sha256}'
        )


def _decompress_dataset(archive: Path, dataset_path: Path) -> None:
    import zstandard

    partial = dataset_path.with_suffix(f'{dataset_path.suffix}.part')
    if partial.exists():
        print('[bootstrap] restarting interrupted dataset decompression')
        partial.unlink()

    print(f'[bootstrap] decompressing {archive} to {dataset_path}')
    decompressor = zstandard.ZstdDecompressor()
    written = 0
    next_commit = COMMIT_INTERVAL
    with (
        archive.open('rb') as compressed,
        decompressor.stream_reader(compressed) as reader,
        partial.open('wb') as output,
    ):
        while chunk := reader.read(CHUNK_SIZE):
            output.write(chunk)
            written += len(chunk)
            if written >= next_commit:
                print(f'[bootstrap] decompressed {written:,} bytes')
                next_commit = written + COMMIT_INTERVAL
        _sync_file(output)

    _validate_dataset(partial)
    partial.replace(dataset_path)
    archive.unlink()
    cache_volume.commit()


def _validate_dataset(dataset_path: Path) -> None:
    import h5py

    with h5py.File(dataset_path, 'r') as dataset:
        required = {'ep_len', 'ep_offset', 'pixels', 'action'}
        missing = required.difference(dataset.keys())
        if missing:
            raise RuntimeError(
                f'PushT dataset is missing required keys: {sorted(missing)}'
            )
        if len(dataset['ep_len']) == 0:
            raise RuntimeError('PushT dataset contains no episodes')


@app.function(
    image=image,
    volumes={CACHE_ROOT: cache_volume},
    cpu=4,
    memory=16 * 1024,
    timeout=6 * 60 * 60,
    scaledown_window=2,
)
def bootstrap() -> dict[str, str]:
    """Populate the persistent dataset and checkpoint cache."""
    dataset_path = CACHE_ROOT / 'datasets' / DATASET_NAME
    archive = CACHE_ROOT / 'downloads' / f'{DATASET_NAME}.zst'
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        print(f'[bootstrap] dataset already exists: {dataset_path}')
        _validate_dataset(dataset_path)
    else:
        _download_dataset_archive(archive)
        _decompress_dataset(archive, dataset_path)

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
