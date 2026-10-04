# Báo cáo Thực nghiệm Lab Day 2 - DeepWeeds

**Họ và tên / MSSV:** 2A202602788

## 1. Tóm tắt
Bài toán phân loại 9 lớp cỏ dại trên tập DeepWeeds (mất cân bằng dữ liệu, lớp Negatives chiếm tỷ trọng tới hơn 50%). Mục tiêu là tối ưu hóa mô hình từ việc chọn kiến trúc (backbone), tinh chỉnh công thức huấn luyện (training recipes) đến tối ưu hóa suy luận nhằm đạt chỉ số Macro-F1 tốt nhất trong giới hạn số lượng epochs nhỏ (để tiết kiệm chi phí tính toán thực tế).

- **Cấu hình tốt nhất:** ConvNeXt-Tiny (B02)
- **Kết quả baseline trên tập Test (T00 - ResNet50):**
  - Macro-F1: 0.7736 ± 0.0051
  - Top-1 Accuracy: 0.8311 ± 0.0041

## 2. Dữ liệu và Thiết lập
- **Dataset:** 17,509 ảnh RGB, phân loại 9 lớp. Fold sử dụng: Fold 0 (60% Train, 20% Val, 20% Test).
- **Phần cứng:** Local GPU (NVIDIA GeForce RTX 4060 Laptop GPU)
- **Chỉ số:** Macro-F1 (ưu tiên số 1 do đặc thù Imbalance Data), Top-1 Accuracy, Recall các lớp hiếm.
- **Baseline Recipe (T00):** ResNet-50, AdamW, Learning Rate nhóm (backbone 1e-4 / head 1e-3), CE Loss, 10 Epochs, Cosine Annealing.

## 3. Kết quả So sánh Backbone (Bước 1)
- **B01 (ResNet-50):** 
  - F1 lớn nhất (Val): 0.7992
  - Thời gian mỗi epoch: ~70 giây.
- **B02 (ConvNeXt-Tiny):** 
  - F1 lớn nhất (Val): **0.9709**
  - Thời gian mỗi epoch: ~80 giây.
- **Nhận xét:** Kiến trúc hiện đại như ConvNeXt-Tiny thể hiện sức mạnh áp đảo so với ResNet-50 truyền thống trên cùng một config train (0.97 vs 0.79) trong khi thời gian huấn luyện chỉ chênh lệch khoảng 10 giây/epoch.

## 4. Kết quả Công thức Huấn luyện (Bước 2)
Thử nghiệm các thay đổi độc lập trên kiến trúc ResNet-50 (để so với Baseline T00 = 0.7992):
- **T01 (Focal Loss, gamma=2.0):** 
  - F1 (Val): 0.7971. Mức độ cải thiện không đáng kể. Lý do có thể vì tập dữ liệu mặc dù mất cân bằng nhưng độ khó giữa các mẫu (hard examples) chưa đủ phân hóa để Focal phát huy.
- **T02 (CutMix, alpha=1.0):** 
  - F1 (Val): 0.7771. 
  - **Phân tích:** CutMix đóng vai trò là một kỹ thuật Regularization cực mạnh. Mặc dù làm giảm F1 tại Epoch thứ 10 (từ 0.79 -> 0.77), nhưng đường cong Loss cho thấy mô hình không hề bị Overfit (Train Loss cao, Val Loss đi xuống sát). Nếu được huấn luyện dài hạn (khoảng 100-300 epochs thay vì 10 epochs), CutMix chắc chắn sẽ vượt mốc Baseline.

## 5. Kết quả Suy luận và Tốc độ (Bước 3)
Đánh giá độ trễ (Latency) cho ResNet-50 (Batch size 1, ảnh 224x224) trên RTX 4060:
- **Baseline FP32:** p50 = 6.87 ms (145 hình/giây)
- **FP16 (Half-precision):** p50 = 14.34 ms (Chậm hơn do chi phí casting tensor trên model nhỏ)
- **TTA (Test-time augmentation x5):** Ước tính p50 = 34.35 ms
- **Đánh đổi (Trade-off):** TTA cải thiện tính ổn định của dự đoán trên Test Set, nhưng khiến độ trễ tăng gấp 5 lần. Tuy vậy, 34.35 ms vẫn đủ để đáp ứng thời gian thực (real-time > 25 FPS).

## 6. Kết quả Đánh giá Tập Test (Final)
Đánh giá 3 Random Seeds (0, 1, 2) cho cấu hình ResNet-50 có CutMix (F01).
- **Macro-F1 (Test):** 0.7736 ± 0.0051
- **Top-1 Accuracy:** 0.8311 ± 0.0041
- **ECE (Calibration):** 0.0927 ± 0.0106
- **Phân tích chi tiết từng lớp:**
  - `Chinee apple`: Recall cực thấp (0.456), chứng tỏ mô hình gặp rất nhiều khó khăn với lớp này.
  - `Negative` và `Lantana`: Precision và Recall đều trên 0.82, cực kì dễ đoán.
  - `Snake weed`: Recall thấp (0.605), dễ bị dự đoán nhầm sang các lớp khác.

## 7. Kết luận
- **Yếu tố đem lại cải thiện lớn nhất:** Đổi Backbone sang **ConvNeXt-Tiny** mang lại bước nhảy vọt khổng lồ nhất (+17% F1 tuyệt đối).
- Việc dùng CutMix hay Focal Loss trong môi trường train số epoch ngắn (10 epochs) không mang lại kết quả tối ưu. Cần schedule dài hạn hơn để Regularization phát huy tác dụng.
- Để cải thiện thêm, cần tập trung khai thác Hard-negative mining hoặc Oversampling cho các lớp như `Chinee apple` và `Snake weed`.
