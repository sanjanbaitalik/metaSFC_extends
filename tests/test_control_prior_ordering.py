import pandas as pd

def test_module_map_ids_are_consecutive():
    df=pd.read_csv('inputs/atlases/AAL116_coarse_modules.csv')
    assert len(df)==116
    assert sorted(df.module_id.unique().tolist())==list(range(df.module_id.nunique()))
