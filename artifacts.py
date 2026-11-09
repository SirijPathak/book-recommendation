"""Versioned, integrity-checked artifact snapshots, published with one atomic replace."""
import hashlib
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def title_hash(titles):
    return hashlib.sha256(json.dumps(list(titles), ensure_ascii=False).encode()).hexdigest()


def save_bundle(path, popular, pivot, catalog, scores):
    path = Path(path)
    payload = pickle.dumps({'popular': popular, 'pivot': pivot, 'catalog': catalog,
                            'scores': scores, 'score_titles': list(pivot.index)})
    manifest = {'schema_version': SCHEMA_VERSION,
                'build_version': datetime.now(timezone.utc).isoformat(),
                'title_order_hash': title_hash(pivot.index),
                'payload_sha256': hashlib.sha256(payload).hexdigest()}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = stream.name
            pickle.dump({'manifest': manifest, 'payload': payload}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_bundle(path):
    # Pickle is only suitable for trusted build outputs, even with integrity checks.
    with open(path, 'rb') as stream:
        bundle = pickle.load(stream)
    manifest, payload = bundle['manifest'], bundle['payload']
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported artifact schema.')
    if hashlib.sha256(payload).hexdigest() != manifest['payload_sha256']:
        raise ValueError('Artifact payload checksum mismatch.')
    data = pickle.loads(payload)
    expected = manifest['title_order_hash']
    if title_hash(data['pivot'].index) != expected or title_hash(data['score_titles']) != expected:
        raise ValueError('Similarity matrix title order mismatch.')
    return data, manifest
