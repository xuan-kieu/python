import streamlit as st
import joblib
import numpy as np

# Load mô hình
try:
    model = joblib.load('rf_model.pkl') # Ghi chú: Nếu dùng Pipeline thì đổi tên file tương ứng
except FileNotFoundError:
    st.error("Không tìm thấy mô hình. Vui lòng chạy file train_model.py trước!")
    st.stop()

st.set_page_config(page_title="Tư vấn Điện thoại", page_icon="📱", layout="wide")
st.title("📱 Hệ thống Dự đoán & Tư vấn Phân khúc Điện thoại")

# Giao diện nhập liệu
st.sidebar.header("Thông số kỹ thuật")

# Thêm đủ các trường như Checklist yêu cầu
ram = st.sidebar.slider("RAM (MB)", 256, 8192, 2048, step=256)
battery = st.sidebar.slider("Pin (mAh)", 500, 6000, 3000, step=100)
int_memory = st.sidebar.slider("Bộ nhớ trong (GB)", 2, 256, 32, step=2)
n_cores = st.sidebar.slider("Số nhân CPU", 1, 8, 4)

st.sidebar.markdown("---")
st.sidebar.subheader("Màn hình & Camera")
fc = st.sidebar.slider("Camera trước (MP)", 0, 20, 5)
pc = st.sidebar.slider("Camera sau (MP)", 0, 30, 10)
px_height = st.sidebar.slider("Chiều dọc màn hình (Pixel)", 0, 1960, 800, step=10)
px_width = st.sidebar.slider("Chiều ngang màn hình (Pixel)", 500, 2000, 1200, step=10)

# Form dự đoán
if st.button("Dự đoán Phân khúc & Nhận Tư vấn", type="primary"):
    
    # Mảng giá trị trung bình/mặc định (để tránh mô hình dự đoán sai vì các thông số quan trọng = 0)
    # Thứ tự 20 features của tập Mobile Price Classification:
    # 0:battery_power, 1:blue, 2:clock_speed, 3:dual_sim, 4:fc, 5:four_g, 6:int_memory, 7:m_dep, 
    # 8:mobile_wt, 9:n_cores, 10:pc, 11:px_height, 12:px_width, 13:ram, 14:sc_h, 15:sc_w, 
    # 16:talk_time, 17:three_g, 18:touch_screen, 19:wifi
    
    features = np.array([[
        battery,       # 0: battery_power
        1,             # 1: blue (Có bluetooth)
        1.5,           # 2: clock_speed
        1,             # 3: dual_sim
        fc,            # 4: fc (Front camera)
        1,             # 5: four_g
        int_memory,    # 6: int_memory
        0.5,           # 7: m_dep
        140,           # 8: mobile_wt
        n_cores,       # 9: n_cores
        pc,            # 10: pc (Primary camera)
        px_height,     # 11: px_height
        px_width,      # 12: px_width
        ram,           # 13: ram
        12,            # 14: sc_h
        6,             # 15: sc_w
        11,            # 16: talk_time
        1,             # 17: three_g
        1,             # 18: touch_screen
        1              # 19: wifi
    ]])
    
    # Dự đoán
    prediction = model.predict(features)[0]
    
    # Lấy xác suất % (Confidence) - Checklist yêu cầu
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[prediction] * 100
    else:
        confidence = 100 # Default nếu model không hỗ trợ
        
    # Kết quả & Tư vấn
    st.success(f"Phân khúc giá dự đoán: **Loại {prediction}** (Độ tự tin: {confidence:.1f}%)")
    st.markdown("*(0: Thấp | 1: Trung bình | 2: Cao | 3: Rất cao)*")
    
    if prediction == 0:
        st.info("💡 **Tư vấn:** Máy giá rẻ (Khoảng 2 - 4 triệu VNĐ). \n\n*Phù hợp:* Nghe gọi, lướt web và dùng làm máy phụ. \n*Gợi ý:* Nokia C20, Redmi A2, Samsung Galaxy A04.")
    elif prediction == 1:
        st.info("💡 **Tư vấn:** Máy tầm trung (Khoảng 5 - 8 triệu VNĐ). \n\n*Phù hợp:* Học tập, làm việc văn phòng và giải trí nhẹ nhàng. \n*Gợi ý:* Xiaomi Redmi Note 12, Samsung Galaxy A34, Oppo A78.")
    elif prediction == 2:
        st.info("💡 **Tư vấn:** Máy cận cao cấp (Khoảng 9 - 14 triệu VNĐ). \n\n*Phù hợp:* Chơi game mượt mà, chụp ảnh đẹp với cấu hình ổn định. \n*Gợi ý:* Xiaomi 13T, Samsung Galaxy A54, iPhone 11.")
    else:
        st.info("💡 **Tư vấn:** Máy cao cấp - Flagship (Trên 15 triệu VNĐ). \n\n*Phù hợp:* Trải nghiệm đỉnh cao, camera xuất sắc, thiết kế sang trọng. \n*Gợi ý:* iPhone 15 Pro Max, Samsung Galaxy S24 Ultra, Xiaomi 14.")