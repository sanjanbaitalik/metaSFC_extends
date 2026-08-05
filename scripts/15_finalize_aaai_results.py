#!/usr/bin/env python3
import subprocess, sys
cmds=[
 [sys.executable,'scripts/10_build_aaai_tables.py'],
 [sys.executable,'scripts/11_statistical_tests.py'],
 [sys.executable,'scripts/12_generate_aaai_figures.py'],
 [sys.executable,'scripts/13_generate_method_overview.py'],
 [sys.executable,'scripts/18_build_submission_tables.py'],
]
for c in cmds:
 print('RUN',' '.join(c),flush=True); subprocess.run(c,check=True)
print('Final tables, tests and figures are under outputs/aaai/.')
