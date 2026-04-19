# Python — Dự đoán phân khúc giá điện thoại

Dự án Machine Learning nhằm **dự đoán phân khúc giá** điện thoại và **hỗ trợ tư vấn** người dùng (gợi ý dựa trên nhãn / xác suất dự đoán). Mã nguồn gồm pipeline huấn luyện (scikit-learn), notebook khám phá dữ liệu, và API FastAPI để triển khai dự đoán.

## Cấu trúc thư mục

- `data/raw/` — dữ liệu gốc (không commit file lớn; xem `.gitignore`)
- `data/processed/` — dữ liệu đã xử lý
- `notebooks/` — Jupyter: EDA và thử nghiệm huấn luyện
- `src/` — `model.py` (SVM, Random Forest), `evaluate.py` (F1-Score)
- `api/` — backend FastAPI (`main.py`)

## Yêu cầu

- Python 3.10+ (khuyến nghị)

## Cài đặt môi trường

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Cài dependency:

```bash
pip install -r requirements.txt
```

## Chạy Jupyter (EDA / training)

```bash
jupyter notebook notebooks/01_EDA_and_Training.ipynb
```

Hoặc:

```bash
jupyter lab
```

## Chạy API (FastAPI)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Tài liệu tự động: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Kiểm tra sức khỏe: `GET /health`

Sau khi có mô hình đã train, cần load artifact trong `api/main.py` (ví dụ `joblib`) và nối endpoint `POST /predict`.

## Quy trình gợi ý

1. Đặt file dữ liệu vào `data/raw/`, xử lý và lưu vào `data/processed/`.
2. Khám phá và huấn luyện trong `notebooks/01_EDA_and_Training.ipynb`.
3. Tách logic huấn luyện tái sử dụng qua `src/model.py` và đánh giá qua `src/evaluate.py`.
4. Xuất model + preprocessor, tích hợp vào `api/main.py`.

## Giấy phép

Thêm license phù hợp khi công khai repo.
