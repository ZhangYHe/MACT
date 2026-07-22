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
- Explicit 584/584
- Implicit 135/135
- Bridging 262/262
- Intersection 105/105
- Comparison 275/275


| Method                | model                       | Metric               | Overall | Index |  Sort | Group | Filter |   GRO |   CAT |   TEM |   AGG |   ARI |   SPA |   QUA |   OTH | Explicit | Implicit | Bridging | Intersection | Comparison | Path                                                                                      | git hash                                 |
| ----------------------- | ----------------------------- | ---------------------- | --------: | ------: | ------: | ------: | -------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ---------: | ---------: | ---------: | -------------: | -----------: | ------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Direct LLM (baseline) | gpt-5.4                     | Denotation EM        |   52.34 | 53.55 | 42.86 | 62.01 |  46.54 | 55.56 | 25.00 | 53.57 | 46.67 | 45.79 | 33.33 | 71.35 | 54.87 |    52.23 |    54.07 |    41.22 |        60.00 |      58.18 | nas/code/table/MACT/baselines/DirectLLM/output/gpt_5_4_direct_llm_crt_answerable_07211100 |                                          |
| Direct LLM (baseline) | gpt-5.4                     | Normalized string EM |   51.65 | 52.90 | 40.48 | 61.65 |  46.08 | 55.56 | 25.00 | 53.57 | 45.71 | 44.86 | 33.33 | 71.35 | 54.32 |    51.54 |    53.33 |    40.08 |        59.05 |      57.82 |                                                                                           |                                          |
| TAPEX(baseline)       | tapex-large-finetuned-wtq   | Denotation EM        |   34.16 | 35.00 | 23.81 | 44.09 |  32.72 | 30.56 |  0.00 | 21.43 | 28.81 | 25.23 | 22.22 | 53.22 | 35.65 |    33.05 |    40.00 |    27.86 |        32.38 |      40.73 | nas/code/table/MACT/baselines/TAPEX/output/crt_tapex-large-wtq_20260722_130659            |                                          |
| TAPEX(baseline)       | tapex-large-finetuned-wtq   | Normalized string EM |   33.88 | 34.68 | 23.81 | 43.73 |  32.72 | 30.56 |  0.00 | 21.43 | 28.33 | 24.77 | 22.22 | 53.22 | 35.38 |    32.71 |    40.00 |    27.10 |        32.38 |      40.73 |                                                                                           |                                          |
| TAPAS(baseline)       | tapas-large-finetuned-wtq   | Denotation EM        |    5.23 |  5.81 |  4.76 |  5.38 |   8.29 |  7.41 |  0.00 |  3.57 |  7.38 |  4.21 |  5.56 |  0.00 |  3.34 |     5.99 |     2.22 |     7.63 |         4.76 |       3.27 | nas/code/table/MACT/baselines/TAPAS/output/crt_tapas-large-wtq_20260722_130512            |                                          |
| TAPAS(baseline)       | tapas-large-finetuned-wtq   | Normalized string EM |    5.23 |  5.81 |  4.76 |  5.38 |   8.29 |  7.41 |  0.00 |  3.57 |  7.38 |  4.21 |  5.56 |  0.00 |  3.34 |     5.99 |     2.22 |     7.63 |         4.76 |       3.27 |                                                                                           |                                          |
| ProTrix               | pkupie/ProTrix              | Denotation EM        |   35.67 | 36.77 | 26.19 | 45.16 |  31.34 | 32.41 | 12.50 | 21.43 | 30.48 | 30.37 | 27.78 | 53.22 | 39.83 |    34.93 |    39.26 |    27.86 |        37.14 |      42.55 | nas/code/table/MACT/baselines/ProTrix/output/crt_protrix-7b_20260722_171927               |                                          |
| ProTrix               | pkupie/ProTrix              | Normalized string EM |   34.85 | 35.97 | 26.19 | 43.73 |  30.88 | 31.48 | 12.50 | 21.43 | 30.00 | 29.44 | 27.78 | 52.05 | 38.44 |    34.08 |    38.52 |    26.34 |        37.14 |      41.82 |                                                                                           |                                          |
| OmniTab               | omnitab-large-finetuned-wtq | Denotation EM        |   29.06 | 30.81 | 19.05 | 36.56 |  27.19 | 25.00 | 12.50 | 17.86 | 22.86 | 18.69 | 16.67 | 46.78 | 32.03 |    27.23 |    37.78 |    25.95 |        30.48 |      29.09 | nas/code/table/MACT/baselines/OmniTab/output/crt_omnitab-large-wtq_20260722_173345        |                                          |
| OmniTab               | omnitab-large-finetuned-wtq | Normalized string EM |   28.79 | 30.48 | 19.05 | 36.20 |  27.19 | 25.00 | 12.50 | 17.86 | 22.38 | 18.22 | 16.67 | 46.78 | 31.75 |    26.88 |    37.78 |    25.19 |        30.48 |      29.09 |                                                                                           |                                          |
| TableLlama            |                             |                      |         |       |       |       |        |       |       |       |       |       |       |       |       |          |          |          |              |            |                                                                                           |                                          |
| work2                 | gpt-5.4                     | Denotation EM        |   58.95 | 61.45 | 47.62 | 66.67 |  52.07 | 60.19 | 25.00 | 64.29 | 50.71 | 54.21 | 38.89 | 83.04 | 59.61 |    59.25 |    57.78 |    53.82 |        66.67 |      59.27 | nas/code/table/MACT/output/runs/crt_gpt-5.4_20260721_1428                                 | 8fd845cfe694267b809d3ac9493ef5ec417f5f75 |
| work2                 | gpt-5.4                     | Normalized string EM |   24.10 | 24.03 | 30.95 | 22.58 |  26.73 | 32.41 | 12.50 | 39.29 | 30.48 | 28.50 | 27.78 |  7.02 | 18.11 |    27.57 |     9.63 |    28.24 |        21.90 |      21.45 |                                                                                           |                                          |
