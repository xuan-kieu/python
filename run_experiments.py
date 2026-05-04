import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# Thêm đường dẫn để import các hàm từ src/evaluate.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.evaluate import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_learning_curve, 
    plot_validation_curve
)

def main():
    print("==============================================")
    print(" BẮT ĐẦU EXPERIMENTS: SVM TUNING & PLOTTING")
    print("==============================================\n")
    
    # 1. Đọc dữ liệu
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy 'train.csv'. Hãy đảm bảo file nằm cùng thư mục.")
        return

    X = df.drop('price_range', axis=1)
    y = df['price_range']
    
    # Chia tập dữ liệu 70% Train, 30% Test (để đơn giản hóa việc vẽ ROC/Confusion Matrix)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 2. Xây dựng Pipeline cho SVM (SVM bắt buộc dùng StandardScaler)
    svm_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()), 
        ('svm', SVC(probability=True, class_weight='balanced', random_state=42)) 
        # LƯU Ý: probability=True là BẮT BUỘC để vẽ ROC Curve bằng predict_proba
    ])

    # 3. Tuning SVM (GridSearchCV)
    print("⏳ [1/4] Đang Tuning SVM (GridSearchCV)... Quá trình này có thể mất 1-2 phút.")
    svm_param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1],
        'svm__kernel': ['rbf', 'linear']
    }

    svm_grid = GridSearchCV(
        estimator=svm_pipeline, 
        param_grid=svm_param_grid, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)

    best_svm = svm_grid.best_estimator_
    print(f"✅ Tham số TỐT NHẤT cho SVM: {svm_grid.best_params_}")

    # 4. Cross Validation (cv=10) theo đúng Checklist
    print("\n⏳ [2/4] Đang chạy Cross-Validation (cv=10)...")
    cv_scores = cross_val_score(best_svm, X_train, y_train, cv=10, scoring='f1_weighted', n_jobs=-1)
    print(f"✅ F1-score (cv=10): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 5. Sinh Biểu đồ Đánh giá Nâng cao
    print("\n⏳ [3/4] Đang vẽ và lưu các biểu đồ (Confusion Matrix, ROC, Learning/Validation Curve)...")
    os.makedirs('outputs', exist_ok=True)
    
    y_pred = best_svm.predict(X_test)

    # Dùng các hàm từ src/evaluate.py
    plot_confusion_matrix(y_test, y_pred, save_path='outputs/svm_confusion_matrix.png')
    plot_roc_curve(best_svm, X_test, y_test, save_path='outputs/svm_roc_curve.png')
    plot_learning_curve(best_svm, X_train, y_train, save_path='outputs/svm_learning_curve.png')

    # Validation curve cần X_train đã scale (vì hàm plot_validation_curve định nghĩa SVC thô bên trong)
    scaler_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    X_train_scaled = scaler_pipe.fit_transform(X_train)
    plot_validation_curve(
        X=X_train_scaled, 
        y=y_train, 
        param_name='C', 
        param_range=[0.01, 0.1, 1, 10, 100], 
        save_path='outputs/svm_validation_curve.png'
    )

    # 6. Feature Importance cho SVM (Chỉ hỗ trợ vẽ nếu kernel='linear')
    print("\n⏳ [4/4] Tính toán Feature Importance...")
    if svm_grid.best_params_['svm__kernel'] == 'linear':
        svm_model = best_svm.named_steps['svm']
        # Lấy giá trị tuyệt đối trung bình của hệ số coef_ qua các lớp
        importance = np.mean(np.abs(svm_model.coef_), axis=0)
        features = X.columns
        indices = np.argsort(importance)[::-1][:5] # Lấy Top 5 Feature
        
        plt.figure(figsize=(8, 5))
        plt.title("Top 5 Feature Importance (SVM - Linear Kernel)")
        plt.bar(range(5), importance[indices], color='teal')
        plt.xticks(range(5), features[indices], rotation=45)
        plt.tight_layout()
        plt.savefig('outputs/svm_feature_importance.png')
        plt.close()
        print("✅ Đã lưu Top 5 Feature Importance.")
    else:
        print("⚠️ Bỏ qua Feature Importance (SVM hiện tại dùng kernel RBF/Poly, không hỗ trợ coef_).")

    print("\n🚀 HOÀN TẤT! Hãy vào thư mục 'outputs/' để lấy các file ảnh PNG ghép vào Báo cáo/Slide của bạn.")

if __name__ == "__main__":
    main()
