# Hiểu về Log-loss trong Machine Learning

## Công thức tổng quát

Log-loss (còn gọi là cross-entropy loss) được định nghĩa như sau:

\[
\text{Log-loss} = -\frac{1}{N}\sum_{i=1}^{N} [y_i\log(\hat{p_i}) + (1-y_i)\log(1-\hat{p_i})]
\]

Trong đó:
- \(N\) là số lượng mẫu
- \(y_i\) là nhãn thực tế (0 hoặc 1)
- \(\hat{p_i}\) là xác suất dự đoán cho lớp 1

## Các trường hợp đặc biệt

1. **Dự đoán hoàn hảo cho lớp 1:**
   Khi \(\hat{p_i} = 1\) và \(y_i = 1\):
   \[
   -\log(\hat{p_i}) = -\log(1) = 0
   \]

2. **Dự đoán hoàn toàn sai:**
   Khi \(\hat{p_i} = 0\) và \(y_i = 1\):
   \[
   -\log(\hat{p_i}) \rightarrow \infty
   \]

## Ví dụ đơn giản

Giả sử có một mẫu dự đoán với xác suất \(\hat{p} = 0.8\):

1. Nếu nhãn thực tế \(y = 1\):
   \[
   \text{Loss} = -\log(0.8) \approx 0.223
   \]

2. Nếu nhãn thực tế \(y = 0\):
   \[
   \text{Loss} = -\log(1-0.8) = -\log(0.2) \approx 1.609
   \]

## Tại sao sử dụng Log-loss?

Log-loss có những ưu điểm sau:
1. Phạt nặng các dự đoán sai mà model rất tự tin
2. Khuyến khích các dự đoán xác suất thực tế
3. Là một hàm lồi (convex function), thuận lợi cho tối ưu hóa

## Code Python đơn giản

```python
import numpy as np

def log_loss(y_true, y_pred):
    epsilon = 1e-15  # Để tránh log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```
```

Khi bạn mở file này bằng một trình xem Markdown có hỗ trợ LaTeX (như VS Code với extension phù hợp, Typora, hoặc GitHub), các công thức sẽ được render đẹp như trong ví dụ trước.

Lưu ý:
1. Một số trình xem Markdown có thể yêu cầu cấu hình thêm để hiển thị LaTeX
2. Trên GitHub, bạn có thể cần sử dụng `$$` thay vì `\[` và `\]`
3. Một số editor phổ biến hỗ trợ tốt Markdown + LaTeX:
   - Typora
   - VS Code + Markdown All in One + Markdown Preview Enhanced
   - HackMD
   - Notion
