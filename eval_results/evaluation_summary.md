# SongFormer Evaluation Results

## Metric Descriptions

本次评估涉及以下指标，均为音乐结构分析（Music Structure Analysis, MSA）领域的常用评估方法：

| Metric | Full Name | Description | Range |
|--------|-----------|-------------|-------|
| **ACC** | Accuracy | 帧级别准确率，逐帧比较预测标签与真实标签的一致性比例 | [0, 1] |
| **HR.5F** | Hit Rate @ 0.5s tolerance | 边界命中率（容差 0.5 秒）。预测边界与真实边界的距离在 0.5 秒以内即视为命中，取 F-measure | [0, 1] |
| **HR3F** | Hit Rate @ 3s tolerance | 边界命中率（容差 3 秒）。容差更宽松，更关注大致位置是否正确 | [0, 1] |
| **HR1F** | Hit Rate @ 1s tolerance | 边界命中率（容差 1 秒）。介于 0.5s 和 3s 之间的中等精度要求 | [0, 1] |
| **PWF** | Pairwise F-measure | 成对 F 值。对所有帧对判断是否属于同一段落，比较预测与真实的一致性。该指标不依赖标签语义，仅关注分段聚类的质量 | [0, 1] |
| **Sf** | Normalized Conditional Entropy (S_f) | 基于归一化条件熵的分段度量（Over-/Under-segmentation 的调和平均）。衡量预测分段与真实分段之间的信息论对齐程度 | [0, 1] |
| **IoU** | Intersection over Union | 交并比。逐类别计算预测区域与真实区域的重叠程度，然后取所有类别的平均值（Per-class IoU 表中展示了各类别的明细） | [0, 1] |

> 以上指标取值均在 **[0, 1]** 范围内，**越接近 1 表示性能越好**。

---

## SongFormBench-CN

| Metric | Value |
|--------|-------|
| Samples | 100 |
| ACC | 0.8575 |
| HR.5F | 0.6955 |
| HR3F | 0.8642 |
| HR1F | 0.7967 |
| PWF | 0.8506 |
| Sf | 0.8725 |
| IoU | 0.7496 |

### Per-class IoU

| Label | IoU |
|-------|-----|
| bridge | 0.2693 |
| chorus | 0.8285 |
| inst | 0.7233 |
| intro | 0.9056 |
| outro | 0.6242 |
| pre-chorus | 0.3303 |
| silence | 0.5937 |
| verse | 0.7950 |

---

## SongFormBench-HarmonixSet

| Metric | Value |
|--------|-------|
| Samples | 200 |
| ACC | 0.7750 |
| HR.5F | 0.7132 |
| HR3F | 0.7994 |
| HR1F | 0.7627 |
| PWF | 0.7794 |
| Sf | 0.8123 |
| IoU | 0.6163 |

### Per-class IoU

| Label | IoU |
|-------|-----|
| bridge | 0.4740 |
| chorus | 0.6946 |
| inst | 0.4700 |
| intro | 0.7657 |
| outro | 0.3719 |
| pre-chorus | 0.2703 |
| silence | 0.5983 |
| verse | 0.7413 |
