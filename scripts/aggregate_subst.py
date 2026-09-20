#!/usr/bin/env python3
"""Recompute substitution sufficient statistics and compare against the frozen paper input."""
import argparse, collections, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
Q1={'helicopter','crackling fire','brushing teeth','airplane','washing machine','wind','fireworks','keyboard typing','door wood creaks','sea waves'}
Q5={'cat','door wood knock','sneezing','coughing','thunderstorm','clapping','dog','pouring water','sheep'}
def rf(b): return b['mf']>0 and min(b['mf'],*b['mj'])<=b['mt']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path);a=p.parse_args()
 d=json.loads((a.root/'data/design/subst_units.json').read_text());us=d['units']
 keys=['itt_den','itt_num','both_den','both_num','b_itt_den','b_itt_num','b_both_den','b_both_num']
 cells=collections.defaultdict(lambda:dict.fromkeys(keys,0));roles=collections.defaultdict(lambda:dict.fromkeys(['itt_den','itt_num','b_itt_den','b_itt_num'],0));skipped=0
 for u in us:
  cat=u['target_category'];q='Q1' if cat in Q1 else ('Q5' if cat in Q5 else None)
  if q is None: skipped+=1;continue
  c=cells[(u['model_id'],u['source_tuple_id'],q)];b,s=u['base'],u['subst'];okb,oks=b['mf']>0,s['mf']>0
  if okb: c['b_itt_den']+=1;c['b_itt_num']+=rf(b)
  if oks: c['itt_den']+=1;c['itt_num']+=rf(s)
  if okb and oks: c['both_den']+=1;c['both_num']+=rf(s);c['b_both_den']+=1;c['b_both_num']+=rf(b)
  r=roles[u['subst_role_id']]
  if okb:r['b_itt_den']+=1;r['b_itt_num']+=rf(b)
  if oks:r['itt_den']+=1;r['itt_num']+=rf(s)
 out={'source':'data/design/subst_units.json','n_units':len(us),'n_conditions':d['n_conditions'],'n_clips':d['n_clips'],'n_models':len({u['model_id'] for u in us}),'n_roles':len(roles),'n_tuples':len({u['source_tuple_id'] for u in us}),'skipped_other_quintiles':skipped,'cells':[dict(model=m,tuple=t,quintile=q,**v) for (m,t,q),v in sorted(cells.items())],'roles':[dict(role=k,**v) for k,v in sorted(roles.items())]}
 ref=json.loads((a.root/'analysis/inputs/frozen/subst_aggregates.json').read_text())
 for k in out:
  if k!='source':assert out[k]==ref[k],f'aggregate mismatch: {k}'
 print(json.dumps({'status':'pass','units':len(us),'cells':len(cells),'roles':len(roles),'all_statistics_match_frozen':True}))
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
