# Hiểu về Log-loss trong Machine Learning

## Công thức tổng quát

Log-loss (còn gọi là cross-entropy loss) được định nghĩa như sau:

$$
\text{Log-loss} = -\frac{1}{N}\sum_{i=1}^{N} [y_i\log(\hat{p_i}) + (1-y_i)\log(1-\hat{p_i})]
$$

Trong đó:
- $N$ là số lượng mẫu
- $y_i$ là nhãn thực tế (0 hoặc 1)
- $\hat{p_i}$ là xác suất dự đoán cho lớp 1

## Các trường hợp đặc biệt

1. **Dự đoán hoàn hảo cho lớp 1:**
   Khi $\hat{p_i} = 1$ và $y_i = 1$:
   $$
   -\log(\hat{p_i}) = -\log(1) = 0
   $$

2. **Dự đoán hoàn toàn sai:**
   Khi $\hat{p_i} = 0$ và $y_i = 1$:
   $$
   -\log(\hat{p_i}) \rightarrow \infty
   $$
