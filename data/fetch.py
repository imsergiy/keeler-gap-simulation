"""Fetch the four fixed archive files and verify their recorded hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def products():
    return json.loads((ROOT/'products.json').read_text())['products']


def verify(path, entry):
    raw = Path(path).read_bytes()
    if len(raw) != entry['byte_length'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError(f"Archive checksum or length mismatch: {entry['name']}")


def fetch(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for product in products():
        for entry in product['files']:
            path = cache/entry['name']
            if path.exists():
                verify(path, entry)
                continue
            with urlopen(entry['url'], timeout=120) as response:
                raw = response.read(entry['byte_length']+1)
            if len(raw) != entry['byte_length'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError(f"Archive checksum or length mismatch: {entry['name']}")
            path.write_bytes(raw)
            print(entry['name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'cache')
    fetch(parser.parse_args().cache)
