# Lab Day 2 - DeepWeeds Classification
**Sinh viên:** Nguyễn Hồ Nam
**MSSV:** 2A202602788

## 1. Môi trường huấn luyện
- Nền tảng: Local GPU (NVIDIA GeForce RTX 4060 Laptop GPU)
- Mã nguồn nằm trong thư mục `code/`

## 2. Cách chạy lại (Reproducibility)
Để chạy lại toàn bộ quy trình từ huấn luyện đến tạo file csv dự đoán, hãy chạy file `run_all.py`:
```bash
cd code/
python run_all.py
```

Để đo độ trễ suy luận, chạy:
```bash
cd code/
python run_latency.py
```

## 3. Cấu trúc bài nộp
- `report.md`: Báo cáo chi tiết quá trình thực nghiệm, phân tích kết quả và kết luận.
- `results.xlsx`: Tổng hợp bảng điểm các lần run (Backbones, Hyperparameters, v.v.)
- `curves/`: Các biểu đồ Loss và F1 trong suốt quá trình train.
- `predictions/`: Chứa kết quả test-set của các models tốt nhất.
- `code/`: Chứa mã nguồn PyTorch và scripts.
