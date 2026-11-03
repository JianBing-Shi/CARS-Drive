from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request
import re

DEST = Path(__file__).resolve().parent / 'evidence' / 'web'
URLS = {
    'CorrectionPlanner_repo': 'https://github.com/guoyihonggyh/CorrectionPlanner-Self-Correction-Planner-with-Reinforcement-Learning-in-Autonomous-Driving',
    'DriveWorld-VLA_icml': 'https://proceedings.mlr.press/v306/jia26f.html',
    'CoIRL-AD_icml': 'https://proceedings.mlr.press/v306/zheng26z.html',
    'DeepSight_icml': 'https://proceedings.mlr.press/v306/zhang26gq.html',
    'Reasoning-VLA_icml': 'https://proceedings.mlr.press/v306/zhang26ib.html',
    'CorrectionPlanner_icml': 'https://proceedings.mlr.press/v306/guo26ae.html',
    'SafeAuto_icml': 'https://proceedings.mlr.press/v267/zhang25cm.html',
    'SelfPlay_icml': 'https://proceedings.mlr.press/v267/cusumano-towner25a.html',
    'SelfPlay_project': 'https://waymo.com/research/robust-autonomy-emerges-from-self-play/',
    'CarPlanner_search': 'https://github.com/search?q=CarPlanner&type=repositories',
    'AdaWM_search': 'https://github.com/search?q=AdaWM&type=repositories',
}

def fetch(item):
    name, url = item
    result = {'name': name, 'url': url}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=25) as response:
            content = response.read().decode('utf-8', errors='replace')
            result.update(status=response.status, resolved_url=response.url, size=len(content))
        (DEST / (name + '.html')).write_text(content, encoding='utf-8')
        title = re.search(r'<title[^>]*>(.*?)</title>', content, re.S)
        result['title'] = title.group(1).strip() if title else None
        result['github_links'] = sorted(set(re.findall(r'https://github\.com/[\w./-]+', content)))
    except Exception as exc:
        result['error'] = str(exc)
    return result

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(fetch, URLS.items()))
    (DEST / 'checks.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
