"""Fetch only the fixed SmolVLM processor/config files used for offline loading."""
import argparse
from pathlib import Path

REPO = 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
REVISION = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('assets/smolvlm'))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, revision=REVISION, local_dir=args.output,
                      allow_patterns=['*.json', '*.txt', 'LICENSE', 'LICENSE.*'])
    print(f'Processor/config files: {args.output}; no model weights downloaded.')

if __name__ == '__main__':
    main()
