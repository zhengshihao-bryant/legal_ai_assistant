# Ablation: rerank（v1）

- Top-K 档位：[5, 10, 20]

## 总表

| 配置 | chunk R@5 | chunk R@10 | chunk R@20 | chunk MRR | doc R@5 | doc R@10 | doc R@20 | doc MRR | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.66 | 0.71 | 0.76 | 0.5265 | 0.73 | 0.75 | 0.76 | 0.5638 | 0.6543 |
| rerank | 0.65 | 0.69 | 0.76 | 0.5252 | 0.72 | 0.76 | 0.76 | 0.5704 | 0.6608 |

## 延迟分解（ms/query）

- Hybrid（embedding + dense + bm25 + rrf）：442.3
- Rerank（CrossEncoder 12→20 对）：2024.1

## 按文档类型 Recall@10（doc 级）

| 类型 | Hybrid | Rerank |
| --- | --- | --- |
| 法规 | 0.5 | 0.5 |
| 制度 | 0.9737 | 0.9737 |
| 合同 | 1.0 | 1.0 |
| 案例 | 0.55 | 0.6 |

*由 Ablation Harness 生成，完整明细见同名 JSON。*