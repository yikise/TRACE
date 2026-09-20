#!/usr/bin/env python3
"""Verify release checksums, path hygiene, Python syntax and raw-to-analysis margins."""
import argparse, ast, collections, gzip, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
  return [json.loads(l) for l in f if l.strip()]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--output',type=Path);a=ap.parse_args();root=a.root.resolve()
 cache={}
 def scores(panel,model):
  key=panel,model
  if key not in cache:
   d=collections.defaultdict(dict)
   for r in rows(root/f'data/scores/{panel}/{model}.jsonl.gz'):d[r['condition_id']][r['intervention']]=r['margin']
   cache[key]=d
  return cache[key]
 def check(panel,u,values=None,cid=None):
  v=values or u;d=scores(panel,u['model_id'])[cid or u['condition_id']]
  assert d['full']==v['mf'] and d['remove_target']==v['mt'],(panel,u['model_id'],cid)
  assert sorted(x for k,x in d.items() if k.startswith('remove_off_'))==sorted(v['mj']),(panel,u['model_id'],cid)
 f=root/'analysis/inputs/frozen';d=json.loads((f/'counterfactual_inputs.json').read_text());n=0
 for panel in ['baseline','global_rms']:
  for u in d['panels'][panel]:check(panel,u);n+=1
 for u in json.loads((f/'repair_inputs.json').read_text())['units']:check('repair',u);n+=1
 for u in json.loads((f/'counterfactual_interface_inputs.json').read_text())['units']:check('prompt',u);n+=1
 for u in json.loads((root/'data/design/subst_units.json').read_text())['units']:
  check('baseline',u,u['base']);check('subst',u,u['subst'],u['subst_condition_id']);n+=2
 sensitive=[re.compile(x) for x in [r'/U[s]ers/[^/\s]+',r'/mnt/ceph[f]s/[^/\s]+',r'/r[o]ot/',r'[\w.-]+\.woa\.com',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9]{25,}',r'AKIA[A-Z0-9]{16}']]
 failures=[];files=0;total=0;largest=(0,'');python_files=0
 excluded={'generated','logs','__pycache__','.git','.venv','outputs','rendered','checkpoints'}
 for p in root.rglob('*'):
  rel=p.relative_to(root)
  if any(x in excluded for x in rel.parts):continue
  if p.is_symlink():failures.append(f'symlink: {rel}');continue
  if not p.is_file():continue
  files+=1;size=p.stat().st_size;total+=size;largest=max(largest,(size,str(rel)))
  if size>=100*1024*1024:failures.append(f'oversized: {rel}')
  if p.suffix=='.py':ast.parse(p.read_text());python_files+=1
  if p.suffix in {'.py','.json','.jsonl','.md','.txt','.yaml','.yml','.sh','.toml'} or p.name in {'.gitignore','.gitattributes'}:
   text=p.read_text()
  elif p.suffix=='.gz':text=gzip.decompress(p.read_bytes()).decode()
  else:continue
  for pat in sensitive:
   if pat.search(text):failures.append(f'sensitive pattern {pat.pattern}: {rel}')
 sums=root/'SHA256SUMS'
 verified=0
 if not sums.exists():raise FileNotFoundError('Missing release SHA256SUMS')
 if sums.exists():
  for line in sums.read_text().splitlines():
   digest,rel=line.split('  ',1);assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel;verified+=1
 assert not failures,failures
 report={'status':'pass','raw_to_analysis_units_checked':n,'python_files_parsed':python_files,'release_files_scanned':files,'bytes_scanned':total,'largest_file':{'path':largest[1],'bytes':largest[0]},'checksum_files_verified':verified,'path_or_secret_pattern_findings':failures,'note':'Heuristic secret scan, not a guarantee. Generated outputs/logs/environments are excluded from public files.'}
 print(json.dumps(report,indent=2))
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
