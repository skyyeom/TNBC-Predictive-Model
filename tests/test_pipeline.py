import pandas as pd
from src.dataset_harmonization import filter_tnbc, add_longitudinal_mri_deltas
from src.multimodal_fusion import collect_feature_columns


def test_tnbc_filter_and_features():
    df = pd.DataFrame({
        "patient_id": ["a", "b"], "pcr": [1, 0],
        "er_negative": [1, 0], "pr_negative": [1, 1], "her2_negative": [1, 1],
        "mri_ld_baseline": [10, 20], "mri_ld_early_ac": [8, 19],
        "mri_ld_interreg": [5, 15], "mri_ld_presurg": [0, 10],
        "age": [40, 50], "deep_mri_00": [0.1, 0.2]
    })
    out = filter_tnbc(df)
    assert len(out) == 1
    out = add_longitudinal_mri_deltas(out)
    cols = collect_feature_columns(out)
    assert "mri_ld_delta_t1_t4" in cols
    assert "deep_mri_00" in cols
