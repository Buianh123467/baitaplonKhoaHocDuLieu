import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Gọi hàm tiền xử lý từ file preprocessing.py cùng thư mục src
from preprocessing import load_data, clean_and_preprocess

def train_model():
    # Lấy đường dẫn tuyệt đối của thư mục chứa file hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Định vị file dữ liệu gốc
    data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'diabetes.csv'))
    
    print("========= DANG KHOI TẠO TIEN XU LY DU LIEU =========")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = clean_and_preprocess(df)
    
    print("\n========= BAT DAU HUAN LUYEN MO HINH (RANDOM FOREST) =========")
    # Khởi tạo và huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Dự đoán thử nghiệm
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"-> Do chinh xac dat duoc: {accuracy * 100:.2f}%")
    print("\nBao cao chi tiet:")
    print(classification_report(y_test, y_pred))
    
    # Định vị thư mục models để lưu trữ kết quả đầu ra
    models_dir = os.path.abspath(os.path.join(current_dir, '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    
    # Lưu mô hình và bộ chuẩn hóa vào thư mục models
    model_save_path = os.path.join(models_dir, 'diabetes_model.pkl')
    scaler_save_path = os.path.join(models_dir, 'scaler.pkl')
    
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    print(f"\n[THANH CONG] Da ghi de va cap nhat model tai: {models_dir}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\n[LOI] Co loi xay ra khi train model: {e}")