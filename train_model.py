import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ==========================================
# 1. ĐỌC DỮ LIỆU & EDA
# ==========================================
try:
    df = pd.read_csv('train.csv')
    print("✅ Đã load dataset thành công.")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file train.csv. Vui lòng tải file từ Kaggle và đặt cùng thư mục.")
    exit()

# Trực quan hóa EDA cơ bản
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['price_range'], y=df['ram'])
plt.title("Phân bố RAM theo Phân khúc giá")
plt.savefig("eda_ram_price.png") 
print("✅ Đã lưu biểu đồ EDA (eda_ram_price.png)")

# ==========================================
# 2. CHIA TẬP DỮ LIỆU (Train 70% / Val 15% / Test 15%)
# ==========================================
X = df.drop('price_range', axis=1)
y = df['price_range']

# Bước 1: Tách 70% Train, 30% cho (Val + Test). Dùng stratify để giữ tỉ lệ nhãn.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Bước 2: Tách 30% còn lại thành 15% Val và 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Kích thước tập Train: {X_train.shape}")
print(f"Kích thước tập Val:   {X_val.shape}")
print(f"Kích thước tập Test:  {X_test.shape}")

# ==========================================
# 3. THIẾT LẬP PIPELINE & XỬ LÝ DỮ LIỆU
# ==========================================
# Pipeline giải quyết triệt để vấn đề Data Leakage và Missing Values
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Điền giá trị thiếu (NaN) bằng trung vị
    ('scaler', MinMaxScaler()),                     # Chuẩn hóa (Checklist đề xuất MinMaxScaler cho RF)
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)) # Khắc phục lệch nhãn nhẹ (nếu có)
])

# ==========================================
# 4. TUNING HYPERPARAMETER (GridSearchCV)
# ==========================================
# Thiết lập lưới tham số theo Checklist
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10],
    'rf__max_features': ['sqrt', 'log2']
}

print("\n⏳ Đang huấn luyện và tìm tham số tốt nhất (GridSearchCV)...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5, # Cross validation 5-fold
    scoring='f1_weighted',
    n_jobs=-1, # Dùng tất cả nhân CPU để chạy nhanh
    verbose=1
)

# Fit trên tập Train
grid_search.fit(X_train, y_train)
print(f"✅ Tham số tốt nhất: {grid_search.best_params_}")

# Lấy mô hình tốt nhất từ GridSearch
best_model = grid_search.best_estimator_

# ==========================================
# 5. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
# Đánh giá sơ bộ trên tập Validation
y_val_pred = best_model.predict(X_val)
print("\n📊 BÁO CÁO TRÊN TẬP VALIDATION:")
print(classification_report(y_val, y_val_pred))

# Đánh giá cuối cùng trên tập Test (Chưa từng nhìn thấy)
y_test_pred = best_model.predict(X_test)
print("\n📊 BÁO CÁO TRÊN TẬP TEST (FINAL):")
print(classification_report(y_test, y_test_pred))

# ==========================================
# 6. LƯU MÔ HÌNH ĐỂ APP SỬ DỤNG
# ==========================================
# Lưu BEST MODEL (Bao gồm cả Imputer, Scaler, và Random Forest) vào 1 file duy nhất
joblib.dump(best_model, 'rf_model.pkl')
print("\n✅ Đã lưu toàn bộ Pipeline (chuẩn hóa + mô hình) vào 'rf_model.pkl'")