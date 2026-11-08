"""Download the public archive and verify the frozen SHA-256."""
from pathlib import Path
import hashlib
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL="https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
EXPECTED="dcb6edc2690f137406638b2309581a71522c4dff19157d118453b448dcddcb68"
target=ROOT/"data/downloads/stage1_train.zip"
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():
    if hashlib.sha256(target.read_bytes()).hexdigest()!=EXPECTED:
        raise SystemExit("Existing ZIP checksum differs; no file overwritten.")
    print("Archive already present and verified.")
else:
    partial=target.with_suffix('.zip.part')
    with urllib.request.urlopen(URL,timeout=180) as response, partial.open('wb') as output:
        while chunk:=response.read(1024*1024):
            output.write(chunk)
    if hashlib.sha256(partial.read_bytes()).hexdigest()!=EXPECTED:
        raise SystemExit("Downloaded checksum differs; partial file retained for inspection.")
    partial.rename(target)
    print("Downloaded and verified:",target)
