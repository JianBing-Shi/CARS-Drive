from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request
import pymupdf

ROOT = Path(__file__).resolve().parent
TARGETS = {
 'CarPlanner_candidate_README': 'https://raw.githubusercontent.com/skr3178/CarPlanner/main/README.md',
 'AutoVLA_rft_config': 'https://raw.githubusercontent.com/ucla-mobility/AutoVLA/main/config/training/qwen2.5-vl-3B-mix-rft.yaml',
 'CoIRL_config': 'https://raw.githubusercontent.com/SEU-zxj/CoIRL-AD/main/projects/configs/coirl/coirl.py',
 'DeepSight_config': 'https://raw.githubusercontent.com/hotdogcheesewhite/DeepSight/main/config/ad_bev_v3.yaml',
 'SelfPlay_project': 'https://gigaflow.ai/',
}

def fetch(item):
    name,url=item
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=20) as res:
            data=res.read().decode('utf-8',errors='replace')
        (ROOT/'evidence'/'web'/(name+'.txt')).write_text(data,encoding='utf-8')
        return {'name':name,'url':url,'status':'ok','size':len(data)}
    except Exception as e:
        return {'name':name,'url':url,'error':str(e)}

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        results=list(pool.map(fetch,TARGETS.items()))
    (ROOT/'evidence'/'web'/'detail_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    out=ROOT/'evidence'/'page_previews'
    out.mkdir(exist_ok=True)
    for name,pages in {'Reasoning-VLA':[5,15],'AdaWM':[19],'DeepSight':[5],'CoIRL-AD':[17]}.items():
        data=json.loads((ROOT/'evidence'/(name+'.json')).read_text(encoding='utf-8'))
        with pymupdf.open(data['source']) as doc:
            for p in pages:
                doc[p-1].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(out/f'{name}-p{p}.png')
