"""Initialize a reproducible Colab context for notebooks 01-07.

This file is executed by the later notebooks so they do not depend on Python
variables left in a different notebook kernel.  It is intentionally safe to
run repeatedly: frozen inputs are copied only when the local workspace does
not already contain them, while run outputs are preserved.
"""
from pathlib import Path
import importlib.metadata as package_metadata
import json
import shutil
import stat
import subprocess
import sys
import zipfile

from google.colab import drive

drive.mount('/content/drive')


def remount_drive() -> None:
    """Recover a stale Drive FUSE mount before retrying a file copy."""
    print('Drive copy failed; forcing a Google Drive remount and retrying once.')
    drive.mount('/content/drive', force_remount=True)


def copy_frozen_tree_with_retry(source: Path, destination: Path) -> None:
    """Copy the immutable input tree, recovering one transient FUSE failure."""
    try:
        shutil.copytree(source, destination)
    except (OSError, shutil.Error) as first_error:
        if destination.exists():
            shutil.rmtree(destination)
        remount_drive()
        try:
            shutil.copytree(source, destination)
        except (OSError, shutil.Error) as retry_error:
            raise RuntimeError(
                'Unable to copy the frozen V3 package from Drive after a remount. '
                'Check that the Drive archive is fully uploaded and rerun Bootstrap.'
            ) from retry_error


def frozen_tree_complete(data_root: Path) -> bool:
    """Detect a partial local copy before trusting the workspace input tree."""
    manifest_path = data_root / 'reports' / 'freeze_manifest_v3.json'
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        prefix = 'data/lseg_v3/'
        for entry in manifest.get('files', []):
            relative = str(entry['path']).replace('\\', '/')
            if relative.startswith(prefix):
                relative = relative[len(prefix):]
            if not (data_root / relative).is_file():
                return False
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return True

REPO = Path('/content/kltn')
PINNED_COMMIT = 'c52e00ebab2cb0025881e6a4b80b9ed672edbf06'
PINNED_REF = 'refs/tags/v3-colab-checkpoint'
if not REPO.exists():
    subprocess.run(
        ['git', 'clone', '--depth', '1', 'https://github.com/maiphuowng205/kltn.git', str(REPO)],
        check=True,
    )
current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
if current != PINNED_COMMIT:
    subprocess.run(
        ['git', 'fetch', '--depth', '1', 'origin', f'{PINNED_REF}:{PINNED_REF}'],
        cwd=REPO,
        check=True,
    )
    subprocess.run(['git', 'checkout', '--detach', PINNED_COMMIT], cwd=REPO, check=True)

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(REPO / 'requirements-colab.txt')],
    check=True,
)

V3_DRIVE_ROOT = Path('/content/drive/MyDrive/kltn/frozen/vn_v3_lseg_2026-08-03')
DRIVE_RUN_ROOT = Path('/content/drive/MyDrive/kltn/runs')
WORKSPACE = Path('/content/vn_v3_workspace')
DATA_ROOT = WORKSPACE / 'data' / 'lseg_v3'

# Accept either an extracted package or the verified ZIP uploaded to Drive.
if not (V3_DRIVE_ROOT / 'data' / 'lseg_v3').exists() and not (V3_DRIVE_ROOT / 'curated').exists():
    zip_name = 'vn_v3_lseg_2026-08-03_drive.zip'
    candidates = [V3_DRIVE_ROOT.parent / zip_name, V3_DRIVE_ROOT.parent.parent / zip_name]
    candidates += list(Path('/content/drive/MyDrive').rglob(zip_name))
    archives = list(dict.fromkeys(path for path in candidates if path.exists()))
    if not archives:
        raise FileNotFoundError(
            f'Upload {zip_name} to My Drive/kltn/frozen/ or extract it there before continuing.'
        )
    V3_DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0], 'r') as archive:
        archive.extractall(V3_DRIVE_ROOT)

if (WORKSPACE / 'data').exists() and not frozen_tree_complete(DATA_ROOT):
    print('Local frozen copy is incomplete; removing it before a verified recopy.')
    shutil.rmtree(WORKSPACE / 'data')

if not (WORKSPACE / 'data').exists():
    if (V3_DRIVE_ROOT / 'data').exists():
        copy_frozen_tree_with_retry(V3_DRIVE_ROOT / 'data', WORKSPACE / 'data')
    else:
        (WORKSPACE / 'data').mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(V3_DRIVE_ROOT, DATA_ROOT, dirs_exist_ok=True)
        except (OSError, shutil.Error):
            if (WORKSPACE / 'data').exists():
                shutil.rmtree(WORKSPACE / 'data')
            remount_drive()
            shutil.copytree(V3_DRIVE_ROOT, DATA_ROOT, dirs_exist_ok=True)

# Frozen data is an input, never a run-output location.
for path in (WORKSPACE / 'data').rglob('*'):
    mode = path.stat().st_mode
    path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
WORKSPACE.joinpath('runs').mkdir(parents=True, exist_ok=True)
WORKSPACE.joinpath('artifacts').mkdir(parents=True, exist_ok=True)
DRIVE_RUN_ROOT.mkdir(parents=True, exist_ok=True)

# Restore completed artifacts from Drive when a later notebook starts in a
# fresh Colab runtime.  Local /content is ephemeral; Drive is the persistent
# handoff between notebook sessions.  Existing local files are preserved or
# merged so a resumable run is never reset by bootstrap.
RESTORE_RUNS = [
    'v3_forecast_baselines', 'v3_risk_coverage', 'v3_portfolio_benchmarks',
    'v3_ptcst_seed_sweep', 'v3_deep_baselines', 'v3_ptcst_ablations',
    'v3_walk_forward', 'v3_statistical_tests', 'v3_robustness',
    'v3_tables', 'v3_final_handoff',
]
restored = []
for name in RESTORE_RUNS:
    source = DRIVE_RUN_ROOT / name
    destination = WORKSPACE / 'runs' / name
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        restored.append(name)
for name in [
    'v3_leakage_report.json', 'v3_determinism_forecast.json',
    'v3_determinism_portfolio.json', 'v3_determinism_report.json',
    'v3_method_validation.json',
]:
    source = DRIVE_RUN_ROOT / name
    destination = WORKSPACE / 'runs' / name
    if source.exists():
        shutil.copy2(source, destination)
        restored.append(f'runs/{name}')
source_cache = DRIVE_RUN_ROOT / 'artifacts' / 'v3_tensor_cache'
if source_cache.exists():
    shutil.copytree(source_cache, WORKSPACE / 'artifacts' / 'v3_tensor_cache', dirs_exist_ok=True)
    restored.append('artifacts/v3_tensor_cache')

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()

def version(name):
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None

runtime = {
    'python': sys.version,
    'git_commit': commit,
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'packages': {
        name: version(name)
        for name in ['numpy', 'pandas', 'pyarrow', 'scikit-learn', 'cvxpy', 'torch', 'xgboost']
    },
}
(WORKSPACE / 'runs' / 'setup_environment.json').write_text(
    json.dumps(runtime, indent=2), encoding='utf-8'
)
shutil.copy2(WORKSPACE / 'runs' / 'setup_environment.json', DRIVE_RUN_ROOT / 'setup_environment.json')
print({'repo': str(REPO), 'commit': commit, 'data_root': str(DATA_ROOT), 'runtime': runtime})
print({'restored_from_drive': restored})
