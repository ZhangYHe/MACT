Question counts (valid gold / total members):

- Overall 726/728
- Index 620/621
- Sort 42/42
- Group 279/280
- Filter 217/217
- GRO 108/108
- CAT 8/8
- TEM 28/28
- AGG 420/421
- ARI 214/214
- SPA 18/18
- QUA 171/172
- OTH 359/359


| Method                | model   | Metric               | Overall | Index |  Sort | Group | Filter |   GRO |   CAT |   TEM |   AGG |   ARI |   SPA |   QUA |   OTH | Explicit | Implicit | Bridging | Intersection | Comparison | Path                                                                                                    | git hash                                 |
| ----------------------- | --------- | ---------------------- | --------: | ------: | ------: | ------: | -------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ---: | ---: | ---: | ---: | ---: |--------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Direct LLM (baseline) | gpt-5.4 | Denotation EM | 52.34 | 53.55 | 42.86 | 62.01 | 46.54 | 55.56 | 25.00 | 53.57 | 46.67 | 45.79 | 33.33 | 71.35 | 54.87 | 52.23 | 54.07 | 41.22 | 60.00 | 58.18 | /home/zhangyunhe/nas/code/table/TableZoomer/baselines/output/gpt_5_4_direct_llm_crt_answerable_07211100 |                                          |
| Direct LLM (baseline) | gpt-5.4 | Normalized string EM | 51.65 | 52.90 | 40.48 | 61.65 | 46.08 | 55.56 | 25.00 | 53.57 | 45.71 | 44.86 | 33.33 | 71.35 | 54.32 | 51.54 | 53.33 | 40.08 | 59.05 | 57.82 |
| TAPAS(baseline)       |         |                      |         |       |       |       |        |       |       |       |       |       |       |       |       |                                                                                                         |                                          |
| TAPEX(baseline)       |         |                      |         |       |       |       |        |       |       |       |       |       |       |       |       |                                                                                                         |                                          |
| MACT                  | gpt-5.4 | Denotation EM | 58.95 | 61.45 | 47.62 | 66.67 | 52.07 | 60.19 | 25.00 | 64.29 | 50.71 | 54.21 | 38.89 | 83.04 | 59.61 | 59.25 | 57.78 | 53.82 | 66.67 | 59.27 | /home/zhangyunhe/nas/code/table/MACT/output/runs/crt_gpt-5.4_20260721_1428                              | 8fd845cfe694267b809d3ac9493ef5ec417f5f75 |
| MACT                  | gpt-5.4 | Normalized string EM | 24.10 | 24.03 | 30.95 | 22.58 | 26.73 | 32.41 | 12.50 | 39.29 | 30.48 | 28.50 | 27.78 | 7.02 | 18.11 | 27.57 | 9.63 | 28.24 | 21.90 | 21.45 |