# Ablation: rerank（v1）

- Top-K 档位：[5, 10, 20]

## 总表

| 配置 | chunk R@5 | chunk R@10 | chunk R@20 | chunk MRR | doc R@5 | doc R@10 | doc R@20 | doc MRR | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | 0.77 | 0.83 | 0.86 | 0.6097 | 0.85 | 0.86 | 0.86 | 0.6521 | 0.7562 |
| rerank | 0.74 | 0.82 | 0.86 | 0.5979 | 0.84 | 0.86 | 0.86 | 0.6463 | 0.752 |

## 延迟分解（ms/query）

- Hybrid（embedding+dense+bm25+rrf）：44.3
- Rerank（CrossEncoder）：1782.3

## 按文档类型 Recall@10（doc 级）

| 类型 | Hybrid | Rerank |
| --- | --- | --- |
| 法规 | 0.9333 | 0.9333 |
| 制度 | 0.9737 | 0.9737 |
| 合同 | 1.0 | 1.0 |
| 案例 | 0.45 | 0.45 |


*由 Ablation Harness 生成，完整明细见同名 JSON。*