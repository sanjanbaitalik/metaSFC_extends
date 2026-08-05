#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

errors=[]; warnings=[]
def check(cond,msg,warning=False):
    if not cond: (warnings if warning else errors).append(msg)

fc_p=Path('inputs/dataset_FC/FC_all.npy'); sc_p=Path('inputs/dataset_SC/SC_all.npy'); y_p=Path('inputs/dataset_SC/label_all.npy')
for p in [fc_p,sc_p,y_p]: check(p.exists(),f'Missing {p}')
if not errors:
    fc=np.load(fc_p); sc=np.load(sc_p); y=np.load(y_p)
    check(fc.ndim==3 and fc.shape[1:]==(116,116),f'FC shape {fc.shape}')
    check(sc.shape==fc.shape,f'SC shape {sc.shape} != FC {fc.shape}')
    check(y.reshape(-1).shape[0]==fc.shape[0],f'Labels {y.shape} vs subjects {fc.shape[0]}')
    check(np.isfinite(fc).all() and np.isfinite(sc).all() and np.isfinite(y).all(),'Nonfinite values in arrays')
    check(fc.shape[0]>=100,f'Only {fc.shape[0]} subjects; final target is >=300',warning=True)
    check(abs(float(np.mean(y)))>0.2 or float(np.std(y))>2.0,'Labels appear globally standardized. Re-run scripts/24_pack_hcp_arrays.py to keep raw labels.',warning=True)

prior_dirs=['outputs/priors/working_memory/aal116','outputs/priors/working_memory_shuffled/aal116','outputs/priors/random_prior/aal116']
for d in prior_dirs:
    d=Path(d)
    for f in ['roi_prior.csv','module_prior.csv','edge_prior.npy']:
        check((d/f).exists(),f'Missing {d/f}')
    if (d/'roi_prior.csv').exists(): check(len(pd.read_csv(d/'roi_prior.csv'))==116,f'{d}/roi_prior.csv must have 116 rows')
    if (d/'edge_prior.npy').exists(): check(np.load(d/'edge_prior.npy').shape==(116,116),f'{d}/edge_prior.npy wrong shape')

configs=sorted(Path('configs/aaai').glob('E[0-9]_*.yaml'))
check(len(configs)==10,f'Expected 10 E0-E9 configs, found {len(configs)}')
for p in configs:
    c=yaml.safe_load(p.read_text())
    check(len(c.get('seeds',[]))>=5,f'{p}: fewer than 5 seeds',warning=True)
    check(c.get('n_folds',0)>=5,f'{p}: fewer than 5 folds',warning=True)

family=Path('inputs/dataset_SC/family_groups.npy')
check(family.exists(),'Family groups absent. Generate from restricted HCP data for leakage-safe final evaluation.',warning=True)
print('\nPREFLIGHT')
for x in warnings: print('[WARN]',x)
for x in errors: print('[ERROR]',x)
Path('outputs/aaai').mkdir(parents=True,exist_ok=True)
Path('outputs/aaai/preflight.json').write_text(json.dumps({'errors':errors,'warnings':warnings},indent=2))
if errors: raise SystemExit(1)
print('PASS with',len(warnings),'warnings')
