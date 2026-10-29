"""Single-image CLI: python demo.py --image photo.jpg --prompt 'Warm photo.' --output out.jpg"""
import argparse
from pathlib import Path
from PIL import Image
from model_utils import ToneAdjuster, DEVICE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        model = ToneAdjuster()
        with Image.open(args.image) as image:
            result = model.adjust(image, args.prompt)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.save(args.output)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(f'Saved {args.output} | pretrained CLIPtone | device={DEVICE}')


if __name__ == '__main__':
    main()
