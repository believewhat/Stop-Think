# Qwen3-30B-A3B: three-run variability

Source: Table Suppl. 5 of the updated manuscript supplied for this release.
These are **three-run aggregate statistics**, not separate rows for each run.
The main comparison in the root README is Table 2 and has a different
aggregation/selection basis; do not treat its values as the following means.

Accuracy is in percent; accuracy SD is in percentage points.
Tokens count mean **output tokens**, not only the THINK span.
Intervals are the manuscript's two-sided run-level Student-t 95% confidence
intervals with n = 3 and df = 2 (mean +/- 4.30265 * SD / sqrt(3)).
They quantify run-to-run variation, not a question-level binomial interval.
Displayed values are rounded as in the PDF.

| Dataset | Method | Accuracy mean | SD | 95% CI | Output tokens mean | SD | 95% CI |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| MedQA | Baseline | 84.34 | 0.16 | [83.94, 84.75] | 2,516.34 | 5.23 | [2,503.35, 2,529.33] |
| MedQA | DEER | 84.37 | 0.08 | [84.17, 84.56] | 1,151.63 | 4.31 | [1,140.92, 1,162.34] |
| MedQA | ESTAR-LITE | 83.56 | 0.82 | [81.52, 85.59] | 437.33 | 7.75 | [418.08, 456.58] |
| MedQA | ESTAR | 83.74 | 0.79 | [81.79, 85.69] | 428.58 | 7.60 | [409.72, 447.45] |
| JAMA | Baseline | 60.42 | 0.46 | [59.27, 61.57] | 2,581.66 | 9.62 | [2,557.76, 2,605.55] |
| JAMA | DEER | 59.74 | 0.56 | [58.36, 61.12] | 1,245.26 | 9.87 | [1,220.74, 1,269.78] |
| JAMA | ESTAR-LITE | 59.94 | 0.45 | [58.82, 61.06] | 553.40 | 12.22 | [523.05, 583.74] |
| JAMA | ESTAR | 60.10 | 0.39 | [59.13, 61.08] | 542.33 | 11.98 | [512.58, 572.08] |
| MATH-500 | Baseline | 96.53 | 0.42 | [95.50, 97.57] | 4,679.04 | 71.70 | [4,500.93, 4,857.16] |
| MATH-500 | DEER | 95.33 | 0.12 | [95.05, 95.62] | 3,212.59 | 146.03 | [2,849.83, 3,575.35] |
| MATH-500 | ESTAR-LITE | 96.27 | 0.42 | [95.23, 97.30] | 2,779.79 | 58.05 | [2,635.57, 2,924.00] |
| MATH-500 | ESTAR | 96.40 | 0.40 | [95.41, 97.39] | 2,724.19 | 56.89 | [2,582.87, 2,865.51] |
| AIME2024 | Baseline | 82.22 | 3.85 | [72.66, 91.78] | 15,063.59 | 969.62 | [12,654.92, 17,472.26] |
| AIME2024 | DEER | 77.78 | 9.62 | [53.87, 101.68] | 10,772.29 | 128.74 | [10,452.48, 11,092.10] |
| AIME2024 | ESTAR-LITE | 81.11 | 5.09 | [68.46, 93.76] | 8,320.38 | 1,009.38 | [5,812.93, 10,827.82] |
| AIME2024 | ESTAR | 82.22 | 5.09 | [69.57, 94.87] | 8,153.97 | 989.20 | [5,696.66, 10,611.28] |

AIME2024 has 30 questions per run and consequently wide intervals. The DEER
upper endpoint of 101.68% is faithfully retained from the unbounded t interval;
it is not an attainable accuracy and is not silently clipped to 100%.

The source PDF does not establish a uniform single-run selection rule for
Table 2. For conclusions about robustness across methods, prefer these
three-run summaries rather than comparing selected favorable/unfavorable runs.
Raw outputs were not re-scored during this documentation update.
