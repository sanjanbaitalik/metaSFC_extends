from pathlib import Path
import yaml

def test_e0_e9_configs_exist_and_are_consistent():
    files=sorted(Path('configs/aaai').glob('E[0-9]_*.yaml'))
    assert len(files)==10
    configs=[yaml.safe_load(p.read_text()) for p in files]
    assert {c['experiment_id'] for c in configs}=={f'E{i}' for i in range(10)}
    for c in configs:
        assert c['n_folds']>=5
        assert len(c['seeds'])>=5
        assert c['standardize_labels_within_fold'] is True
    assert [c for c in configs if c['experiment_id']=='E0'][0]['prior_type']=='none'

def test_control_lambdas_match_true_counterparts():
    cfg={yaml.safe_load(p.read_text())['experiment_id']:yaml.safe_load(p.read_text()) for p in Path('configs/aaai').glob('E[0-9]_*.yaml')}
    assert cfg['E1']['lambda_node']==cfg['E2']['lambda_node']==cfg['E3']['lambda_node']
    assert cfg['E4']['lambda_module']==cfg['E5']['lambda_module']==cfg['E6']['lambda_module']
    assert cfg['E7']['lambda_edge']==cfg['E8']['lambda_edge']==cfg['E9']['lambda_edge']
