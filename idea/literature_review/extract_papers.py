from pathlib import Path
import json
import re
import pymupdf

ROOT = Path(__file__).resolve().parent
DESKTOP = Path('C:/Users/shi-jia-liang/Desktop')
NAMES = {
    'DriveWorld-VLA': 'DriveWorld-VLA.pdf',
    'SafeAuto': 'SafeAuto Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models.pdf',
    'DeepSight': 'DeepSight.pdf',
    'Reasoning-VLA': 'Reasoning-VLA.pdf',
    'CoIRL-AD': 'CoIRL-AD.pdf',
    'AutoVLA': 'AutoVLA.pdf',
    'CRAFT': 'CRAFT.pdf',
    'RIFT': 'RIFT.pdf',
    'CarPlanner': 'CarPlanner.pdf',
    'AdaWM': 'ADAWM.pdf',
    'Raw2Drive': 'Raw2Drive.pdf',
    'Self-Play': 'Robust Autonomy Emerges from Self-Play.pdf',
    'CorrectionPlanner': 'CorrectionPlanner Self-Correction Plannerwith Reinforcement Learning in Autonomous Driving.pdf',
}

def main():
    out = ROOT / 'evidence'
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, filename in NAMES.items():
        source = DESKTOP / filename
        doc = pymupdf.open(source)
        pages = []
        for i, page in enumerate(doc):
            blocks = page.get_text('blocks', sort=True)
            pages.append({
                'page': i + 1,
                'text': page.get_text(sort=True),
                'links': [link.get('uri') for link in page.get_links() if link.get('uri')],
                'blocks': [{'id': f'{name}-p{i+1}-b{j+1}', 'bbox': list(b[:4]), 'text': b[4]} for j, b in enumerate(blocks) if len(b) > 4 and isinstance(b[4], str)],
            })
        data = {'paper': name, 'source': str(source), 'pages': pages, 'metadata': doc.metadata}
        (out / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / f'{name}.txt').write_text('\n\n'.join(f'=== PDF PAGE {p["page"]} ===\n{p["text"]}' for p in pages), encoding='utf-8')
        urls = sorted(set(u for p in pages for u in p['links']))
        manifest[name] = {'source': str(source), 'pages': len(pages), 'characters': sum(len(p['text']) for p in pages), 'urls': urls}
        print(f'\n{name}: {len(pages)} pages')
        print(pages[0]['text'][:2400])
        print('LINKS:', '\n'.join(u for u in urls if any(s in u.lower() for s in ['github', 'arxiv', 'huggingface', 'project', 'drive.google'])))
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
