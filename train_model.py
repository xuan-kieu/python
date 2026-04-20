import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Đọc dữ liệu
df = pd.read_csv('train.csv')

# 2. Trực quan hóa EDA (Lưu ảnh để cho vào Slide)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['price_range'], y=df['ram'])
plt.title("Phân bố RAM theo Phân khúc giá")
plt.savefig("eda_ram_price.png") # Lưu ảnh lại

# 3. Tiền xử lý & Train/Test Split
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Huấn luyện Random Forest (Đã tuning cơ bản)
rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
rf.fit(X_train, y_train)

# 5. Phân tích lỗi (Error Analysis)
y_pred = rf.predict(X_test)
print("BÁO CÁO PHÂN LOẠI:\n", classification_report(y_test, y_pred))

# 6. Lưu mô hình cho Streamlit App
joblib.dump(rf, 'rf_model.pkl')
print("✅ Đã lưu mô hình thành công vào 'rf_model.pkl'")