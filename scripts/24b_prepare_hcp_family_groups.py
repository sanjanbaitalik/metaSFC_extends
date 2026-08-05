#!/usr/bin/env python3
"""Create family/group IDs aligned with packed HCP arrays.

Requires an HCP restricted CSV containing Subject and Family_ID (or another
chosen grouping column). Access and use must follow HCP data-use terms.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ap=argparse.ArgumentParser(); ap.add_argument('--restricted_csv',required=True); ap.add_argument('--group_col',default='Family_ID'); ap.add_argument('--subjects',default='inputs/dataset_SC/hcp_subjects_used.csv'); ap.add_argument('--out',default='inputs/dataset_SC/family_groups.npy'); args=ap.parse_args()
subjects=pd.read_csv(args.subjects); subject_col='subject' if 'subject' in subjects else 'Subject'; subjects[subject_col]=subjects[subject_col].astype(str)
r=pd.read_csv(args.restricted_csv); r['Subject']=r['Subject'].astype(str)
if args.group_col not in r: raise ValueError(f'Missing {args.group_col}')
merged=subjects[[subject_col]].merge(r[['Subject',args.group_col]],left_on=subject_col,right_on='Subject',how='left')
if merged[args.group_col].isna().any(): raise ValueError(f'Missing family IDs for {merged[merged[args.group_col].isna()][subject_col].tolist()[:10]}')
groups=merged[args.group_col].astype(str).to_numpy(); Path(args.out).parent.mkdir(parents=True,exist_ok=True); np.save(args.out,groups)
print('Saved',args.out,groups.shape,'unique groups',len(np.unique(groups)))
