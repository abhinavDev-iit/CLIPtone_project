"""Fetch the three inference checkpoints from the authors' documented sources."""
from pathlib import Path
import hashlib
import urllib.request
import argparse

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'checkpoint'


def download(url, name):
    path = DEST / name
    if path.exists() and path.stat().st_size > 10000:
        print(f'Already present: {path.name}')
        return
    temp = path.with_suffix('.partial')
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if temp.stat().st_size < 10000:
            raise ValueError('Download is too small to be a checkpoint.')
        temp.replace(path)
        print(f'Downloaded {name}')
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    DEST.mkdir(exist_ok=True)
    failures = []
    try:
        download('https://raw.githubusercontent.com/ImCharlesY/AdaInt/main/pretrained/AiLUT-FiveK-sRGB.pth',
                 'AiLUT-FiveK-sRGB.pth')
    except Exception as error:
        failures.append(f'AiLUT: {error}')
    try:
        import clip
        url = clip.clip._MODELS['RN50']
        download(url, 'RN50.pt')
        expected = url.split('/')[-2]
        actual = hashlib.file_digest((DEST / 'RN50.pt').open('rb'), 'sha256').hexdigest()
        if actual != expected:
            (DEST / 'RN50.pt').unlink()
            raise ValueError('CLIP SHA256 mismatch; removed incomplete or incorrect weights.')
    except Exception as error:
        failures.append(f'CLIP RN50: {error}')
    try:
        # Standard Google Drive download endpoint avoids a legacy docs-host redirect.
        download('https://drive.usercontent.google.com/download?'
                 'id=171NTXGgme8AmSJJyy1F4hEE3OnRBQuql&export=download&confirm=t',
                 'pretrained.pth')
    except Exception as error:
        failures.append(f'Text adaptor: {error}')
    if failures:
        print('\nDOWNLOAD INCOMPLETE:\n' + '\n'.join(failures))
        print('Allow the documented hosts or manually place the real checkpoints in checkpoint/.')
        raise SystemExit(1)
    print('Downloads complete. Run the CLI to verify checkpoint compatibility.')


if __name__ == '__main__':
    main()
