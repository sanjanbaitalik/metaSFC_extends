#!/usr/bin/env python3
"""Average multiple atlas-level prior sets into a composite cognitive prior."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser(); ap.add_argument('--inputs',nargs='+',required=True,help='Prior directories'); ap.add_argument('--out',required=True); ap.add_argument('--name',default='cognitive_control_composite'); args=ap.parse_args()
dirs=[Path(x) for x in args.inputs]; out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
rois=[pd.read_csv(d/'roi_prior.csv').sort_values('roi_index') for d in dirs]
base=rois[0][['roi_index','roi_label']].copy(); scores=np.stack([r.prior_score.to_numpy(float) for r in rois]); mean=scores.mean(0); mean=(mean-mean.min())/(mean.max()-mean.min()+1e-12); base['raw_score']=scores.mean(0); base['prior_score']=mean; base.to_csv(out/'roi_prior.csv',index=False)
# Preserve module IDs/order from first prior if available, otherwise derive from mapping.
mods=[]
for d in dirs:
 p=d/'module_prior.csv'
 if p.exists(): mods.append(pd.read_csv(p))
if mods:
 key=[c for c in ['module_id','module'] if c in mods[0].columns]; m=mods[0][key].copy(); vals=np.stack([x.prior_score.to_numpy(float) for x in mods]); v=vals.mean(0); m['raw_score']=v; m['prior_score']=(v-v.min())/(v.max()-v.min()+1e-12); m.to_csv(out/'module_prior.csv',index=False)
edge=np.outer(mean,mean).astype('float32'); np.save(out/'edge_prior.npy',edge)
(out/'metadata.json').write_text(json.dumps({'name':args.name,'sources':[str(d) for d in dirs]},indent=2)); print(out)
