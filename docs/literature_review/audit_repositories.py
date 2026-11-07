from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64
import datetime
import json
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent / 'evidence' / 'repositories'
REPOS = {
    'DriveWorld-VLA': 'liulin815/DriveWorld-VLA',
    'SafeAuto': 'AI-secure/SafeAuto',
    'DeepSight': 'hotdogcheesewhite/DeepSight',
    'Reasoning-VLA': 'xipi702/Reasoning-VLA',
    'CoIRL-AD': 'SEU-zxj/CoIRL-AD',
    'AutoVLA': 'ucla-mobility/AutoVLA',
    'CRAFT': 'CurryChen77/CraftPolicy',
    'RIFT': 'CurryChen77/RIFT',
    'Raw2Drive': 'Thinklab-SJTU/Raw2Drive',
    'CorrectionPlanner': 'guoyihonggyh/CorrectionPlanner-Self-Correction-Planner-with-Reinforcement-Learning-in-Autonomous-Driving',
}

def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'E2EAD-literature-audit', 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

def audit(item):
    name, repo = item
    result = {'paper': name, 'repo': repo, 'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        meta = get_json('https://api.github.com/repos/' + repo)
        result['metadata'] = {k: meta.get(k) for k in ['html_url','description','default_branch','pushed_at','license','size','archived']}
        tree = get_json(f'https://api.github.com/repos/{repo}/git/trees/{meta["default_branch"]}?recursive=1')
        paths = [entry['path'] for entry in tree.get('tree', []) if entry['type'] == 'blob']
        result['files'] = paths
        result['tree_sha'] = tree.get('sha')
        readme = get_json(f'https://api.github.com/repos/{repo}/readme')
        content = base64.b64decode(readme['content']).decode('utf-8', errors='replace')
        result['readme'] = content
        (ROOT / f'{name}.README.md').write_text(content, encoding='utf-8')
        print(name, 'files=', len(paths), 'code=', sum(p.endswith(('.py','.sh','.cpp','.cu')) for p in paths), 'sha=', result['tree_sha'])
    except Exception as exc:
        result['error'] = str(exc)
        print(name, 'ERROR', str(exc))
    (ROOT / f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(audit, REPOS.items()))
    (ROOT / 'audit.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
