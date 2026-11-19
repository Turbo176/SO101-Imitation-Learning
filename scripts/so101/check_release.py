"""Check project documentation links and locally bundled presentation assets."""
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]

def main():
    failures, links = [], 0
    paths = [ROOT / 'README.md', *sorted((ROOT / 'docs/so101').glob('*.md'))]
    for path in paths:
        text = path.read_text(encoding='utf-8')
        references = re.findall(r'!?(?:\[[^\]]*\])\(([^\s)]+)(?:\s+[^)]*)?\)', text)
        references += re.findall(r'(?:src|href)="([^\"]+)"', text)
        for ref in references:
            if ref.startswith(('http:', 'https:', 'mailto:', '#')):
                continue
            dest = path.parent / unquote(urlsplit(ref).path)
            links += 1
            if not dest.is_file():
                failures.append(f'{path.relative_to(ROOT)}: missing {ref}')
        if re.search(r'[C-F]:[\\/]|/root/autodl|ssh\s+[^\s]+@', text):
            failures.append(f'{path.relative_to(ROOT)}: machine-specific path/connection detail')
    for path in (ROOT / 'media/so101').glob('*.svg'):
        tree = ET.parse(path)
        if tree.getroot().tag != '{http://www.w3.org/2000/svg}svg':
            failures.append(f'Invalid SVG root: {path.name}')
    for path in (ROOT / 'configs/so101').rglob('*.json'):
        json.loads(path.read_text(encoding='utf-8'))
    for path in [ROOT / 'README.md', *(ROOT / 'scripts/so101').glob('*.py')]:
        if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)', path.read_text(encoding='utf-8')):
            failures.append(f'Credential-shaped text: {path.relative_to(ROOT)}')
    print(json.dumps({'ok': not failures, 'documents': len(paths), 'local_links': links, 'failures': failures}, indent=2))
    return bool(failures)

if __name__ == '__main__':
    sys.exit(main())
