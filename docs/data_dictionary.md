# Data Dictionary

| Variable | Type | Description |
|---|---|---|
| patient_id | string | Deidentified patient identifier |
| dataset | string | Source dataset: ispy1, acrin6698, ispy2, duke, cbis_ddsm |
| pcr | binary | 1 = pathological complete response, 0 = residual disease |
| age | numeric | Age at diagnosis |
| tumor_size_cm | numeric | Baseline tumor size in centimeters |
| node_positive | binary | Lymph node involvement at diagnosis |
| grade | ordinal | Histological grade |
| er_negative | binary | ER negativity indicator |
| pr_negative | binary | PR negativity indicator |
| her2_negative | binary | HER2 negativity indicator |
| mri_ld_baseline | numeric | MRI longest diameter at baseline |
| mri_ld_early_ac | numeric | MRI longest diameter at early AC timepoint |
| mri_ld_interreg | numeric | MRI longest diameter at inter-regimen timepoint |
| mri_ld_presurg | numeric | MRI longest diameter prior to surgery |
| mri_ld_delta_t1_t2 | numeric | Relative LD reduction from baseline to early treatment |
| mri_ld_delta_t1_t3 | numeric | Relative LD reduction from baseline to inter-regimen |
| mri_ld_delta_t1_t4 | numeric | Relative LD reduction from baseline to pre-surgery |
| rad_* | numeric | Radiomics features extracted from tumor masks |
| deep_mri_* | numeric | Deep MRI encoder embeddings |
| tp53_mut | binary | Optional TP53 mutation status |
| pik3ca_mut | binary | Optional PIK3CA mutation status |
| hrd_score | numeric | Optional homologous recombination deficiency score |
| tmb | numeric | Optional tumor mutation burden |
