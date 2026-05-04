# Dự án Phân Loại & Tư Vấn Điện Thoại (Mobile Price Classification)

Dự án Machine Learning nhằm **dự đoán phân khúc giá** điện thoại và cung cấp **tư vấn gợi ý** cho người dùng. 
Hệ thống sử dụng các mô hình học máy (Random Forest, SVM) và giao diện trực quan bằng Streamlit.

## 📂 Cấu trúc thư mục

- `app.py` — Ứng dụng Giao diện Web (Streamlit) cho phép người dùng nhập thông số và nhận tư vấn.
- `train_model.py` — Script tiền xử lý dữ liệu và huấn luyện mô hình Random Forest (Pipeline + GridSearchCV) để tạo file `.pkl`.
- `run_experiments.py` — Script nghiên cứu: Tuning SVM, đánh giá Cross-Validation (cv=10) và tự động sinh các biểu đồ (ROC, Confusion Matrix, Feature Importance).
- `train.csv` — Dataset đầu vào (Bạn cần tự tải từ Kaggle - Mobile Price Classification và để ở thư mục này).
- `outputs/` — Thư mục tự động sinh ra chứa các file ảnh biểu đồ để làm báo cáo.
- `src/` — Chứa file `evaluate.py` gồm các hàm chuyên hỗ trợ vẽ biểu đồ nâng cao.

## ⚙️ Yêu cầu & Cài đặt môi trường

Mở terminal và khởi tạo môi trường (khuyên dùng):
```bash
python -m venv .venv
# Trên Windows: .\.venv\Scripts\Activate.ps1
# Trên Mac/Linux: source .venv/bin/activate
```

Cài đặt thư viện:
```bash
pip install -r requirements.txt
```
*(Yêu cầu phải có: pandas, scikit-learn, matplotlib, seaborn, streamlit).*

## 🚀 Hướng dẫn Sử Dụng (Quy trình chuẩn)

Dự án này được chia làm 3 bước rõ ràng để bạn thực hiện từ lúc train đến lúc có giao diện:

### Bước 1: Khảo sát & Vẽ biểu đồ làm Báo cáo
Chạy script dưới đây để máy tự động Tuning SVM, chạy CV=10 và sinh ra một loạt các biểu đồ đẹp mắt vào thư mục `outputs/`:
```bash
python run_experiments.py
```

### Bước 2: Huấn luyện Mô hình chính thức cho Ứng dụng
Chạy script sau để hệ thống dùng GridSearchCV tìm ra mô hình Random Forest tốt nhất, chuẩn hóa dữ liệu và xuất ra file `rf_model.pkl`:
```bash
python train_model.py
```

### Bước 3: Khởi chạy Giao diện (Streamlit Web App)
Sau khi đã có file `rf_model.pkl`, bạn khởi động giao diện bằng lệnh:
```bash
streamlit run app.py
```
Trình duyệt sẽ mở lên (thường là `http://localhost:8501`). Tại đây, bạn dùng các thanh trượt điều chỉnh cấu hình (RAM, Màn hình, Pin...) và nhấn nút để xem hệ thống dự đoán phân khúc + gợi ý mua máy!

---

## 💡 Khắc phục lỗi thường gặp (FAQ)

**1. Lỗi "Không tìm thấy mô hình. Vui lòng chạy file train_model.py trước!"**
- **Nguyên nhân:** File `rf_model.pkl` chưa được tạo ra.
- **Khắc phục:** Đảm bảo bạn đã đưa file `train.csv` vào thư mục, sau đó chạy lại lệnh `python train_model.py` ở **Bước 2** để huấn luyện và sinh ra mô hình. `rf_model.pkl` là một file dữ liệu (tri thức của AI), không phải file mã nguồn để chạy trực tiếp.

**2. Lỗi "streamlit is not recognized..." trên Windows PowerShell**
- **Khắc phục:** Dùng lệnh an toàn sau đây để gọi Streamlit thông qua Python:
  ```bash
  python -m streamlit run app.py
  ```

**3. Streamlit hỏi nhập Email (Welcome to Streamlit)**
- Ở lần chạy đầu tiên, Streamlit sẽ hiện chữ `Email:` và tạm dừng ứng dụng. Bạn chỉ cần click chuột vào terminal và **nhấn phím Enter** (bỏ trống) là ứng dụng sẽ chạy ngay.
- Hoặc dùng lệnh sau để bỏ qua câu hỏi vĩnh viễn:
  ```bash
  python -m streamlit run app.py --browser.gatherUsageStats false
  ```
