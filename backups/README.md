# Restore the project after deleting the local checkout

The `source-data.zip` archive preserves all five original files from `data/source/`,
including the two inputs used by the current builder and three legacy artifacts.
`source-data-manifest.json` records the original sizes and SHA-256 checksums.
The archive was checked against every original file before upload.

## Download the complete project

```powershell
git clone --branch fix/analysis-issues https://github.com/SirijPathak/book-recommendation.git
cd book-recommendation
```

Keep `--branch fix/analysis-issues` until these changes have been merged into the
repository's default branch.

## Restore the original data and Python environment

With Python 3.12 installed:

```powershell
python -m zipfile -e backups/source-data.zip data/source
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv\Scripts\python.exe app.py
```

The app already includes a generated `recommendation_bundle.pkl`; rebuilding is optional.
To rebuild from the restored originals:

```powershell
& .\.venv\Scripts\python.exe build_artifacts.py
```

To verify the restored source files:

```powershell
@'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('backups/source-data-manifest.json').read_text())
for name, expected in manifest.items():
    path = Path('data/source') / name
    assert path.stat().st_size == expected['bytes'], name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], name
print('All source files verified.')
'@ | python -
```

The old `.venv/`, `myenv/`, `__pycache__/`, and `.pytest_cache/` directories are
generated local environments and caches. They are deliberately excluded from Git;
installing the dependencies recreates the environment needed by this project.
No local `.env` files were present when this backup was created.

Only load pickle files from trusted sources.
