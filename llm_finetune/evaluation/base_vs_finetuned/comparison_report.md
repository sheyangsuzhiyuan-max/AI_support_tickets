# Base Model vs Fine-tuned Model Comparison Report

Generated: 2025-12-31 23:43:21

---

## Executive Summary

- **Ticket Type Classification:** 0.00% → 0.00% (N/A)
- **Priority Classification:** 0.00% → 0.00% (N/A)
- **Queue Routing:** 0.00% → 0.00% (N/A)

- **Response Quality (ROUGE-1):** 0.0000 → 0.0000 (N/A)
- **Response Quality (ROUGE-L):** 0.0000 → 0.0000 (N/A)


## Detailed Comparison

### Classification Accuracy

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| Type Accuracy | 0.00% | 0.00% | N/A |
| Priority Accuracy | 0.00% | 0.00% | N/A |
| Queue Accuracy | 0.00% | 0.00% | N/A |

### Response Generation Quality

| Metric | Base Model | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| ROUGE-1 F1 | 0.0000 | 0.0000 | N/A |
| ROUGE-2 F1 | 0.0000 | 0.0000 | N/A |
| ROUGE-L F1 | 0.0000 | 0.0000 | N/A |


## Analysis

❌ **Fine-tuning did not show significant improvement**

- Consider adjusting hyperparameters (learning rate, LoRA rank)
- Check training data quality and distribution


## Recommendations

- **Type Classification:** Consider collecting more training examples for underrepresented types
- **Priority Classification:** Review priority labeling criteria - may need clearer definitions
- **Queue Routing:** 8-way classification is challenging - consider hierarchical classification
- **Response Quality:** Increase training epochs or use larger LoRA rank for better generation