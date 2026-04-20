import streamlit as st
import joblib
import numpy as np

# Load mô hình
model = joblib.load('rf_model.pkl')

st.set_page_config(page_title="Tư vấn Điện thoại", page_icon="📱")
st.title("📱 Hệ thống Dự đoán & Tư vấn Điện thoại")

# Giao diện nhập liệu
st.sidebar.header("Thông số kỹ thuật")
ram = st.sidebar.slider("RAM (MB)", 256, 8192, 2048)
battery = st.sidebar.slider("Pin (mAh)", 500, 6000, 3000)
int_memory = st.sidebar.slider("Bộ nhớ trong (GB)", 2, 256, 32)
n_cores = st.sidebar.slider("Số nhân CPU", 1, 8, 4)

# Form dự đoán
if st.button("Dự đoán Phân khúc & Nhận Tư vấn"):
    # Tạo mảng 20 features chứa giá trị 0 (hoặc trung bình) cho các cột không nhập
    # Thay đổi vị trí index cho đúng với thứ tự cột trong file train.csv của bạn
    features = np.zeros((1, 20)) 
    features[0, 13] = ram           # Giả sử RAM ở cột 13
    features[0, 0] = battery        # Giả sử Pin ở cột 0
    features[0, 6] = int_memory     # Giả sử Bộ nhớ ở cột 6
    features[0, 10] = n_cores       # Giả sử CPU ở cột 10
    
    # Dự đoán
    prediction = model.predict(features)[0]
    
    # Kết quả & Tư vấn
    st.success(f"Phân khúc giá dự đoán: **Loại {prediction}** (0: Thấp, 1: TB, 2: Cao, 3: Rất cao)")
    
    if prediction == 0:
        st.info("💡 Tư vấn: Máy giá rẻ, phù hợp nghe gọi, lướt web và dùng làm máy phụ.")
    elif prediction == 1:
        st.info("💡 Tư vấn: Máy tầm trung, đáp ứng tốt nhu cầu học tập, làm việc văn phòng và giải trí nhẹ nhàng.")
    elif prediction == 2:
        st.info("💡 Tư vấn: Máy cận cao cấp, camera đẹp, chơi game mượt mà với cấu hình ổn định.")
    else:
        st.info("💡 Tư vấn: Máy cao cấp (Flagship), thiết kế sang trọng, trải nghiệm đỉnh cao cho mọi tác vụ nặng nhất.")