"""Download official pretrained StarDist weights and verify frozen SHA-256 values."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official ZIP.')
    args = parser.parse_args()
    source = json.loads((ROOT / 'models/2D_versatile_fluo_source.json').read_text())
    destination = ROOT / 'models' / source['model']
    expected = source['files']
    if destination.is_dir() and all((destination / name).is_file() and sha256(destination / name) == digest
                                    for name, digest in expected.items()):
        print('Official pretrained model already present and verified.')
        return
    archive = args.archive or ROOT / 'models/downloads/python_2D_versatile_fluo.zip'
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix('.zip.part')
        for attempt in range(3):
            try:
                print(f'Downloading official pretrained model (attempt {attempt + 1}/3)...', flush=True)
                with urllib.request.urlopen(source['source'], timeout=120) as response, partial.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if sha256(partial) != source['archive_sha256']:
                    raise ValueError('Downloaded ZIP checksum differs from the frozen model source.')
                partial.replace(archive)
                break
            except Exception:
                if attempt == 2:
                    raise
    if sha256(archive) != source['archive_sha256']:
        raise ValueError('ZIP checksum differs; no model file written.')
    payloads = {}
    with zipfile.ZipFile(archive) as zipped:
        for name, digest in expected.items():
            matches = [entry for entry in zipped.infolist() if not entry.is_dir() and Path(entry.filename).name == name]
            if len(matches) != 1 or matches[0].file_size > 30 * 1024 * 1024:
                raise ValueError(f'Unexpected model archive entry: {name}')
            content = zipped.read(matches[0])
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f'Model file checksum differs: {name}')
            target = destination / name
            if target.exists() and sha256(target) != digest:
                raise ValueError(f'Existing model differs; no file overwritten: {target}')
            payloads[name] = content
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        target = destination / name
        if not target.exists():
            partial = target.with_suffix(target.suffix + '.part')
            partial.write_bytes(content)
            partial.replace(target)
    print('Downloaded and verified official model:', destination)


if __name__ == '__main__':
    main()
