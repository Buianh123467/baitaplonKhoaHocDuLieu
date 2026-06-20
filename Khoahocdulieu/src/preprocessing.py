import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Đọc dữ liệu từ file CSV"""
    return pd.read_csv(filepath)

def clean_and_preprocess(df):
    """Tiền xử lý dữ liệu: Xử lý giá trị 0 vô lý và chuẩn hóa"""
    # Thay thế các giá trị 0 vô lý ở các cột sinh tồn bằng NaN
    cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_fix] = df[cols_fix].replace(0, np.nan)
    
    # Điền giá trị thiếu bằng giá trị trung bình (mean) của cột đó
    df.fillna(df.mean(), inplace=True)
    
    # Tách đặc trưng (X) và nhãn mục tiêu (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Chia tập dữ liệu (80% để huấn luyện, 20% để kiểm tra)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chuẩn hóa số liệu đưa về cùng một quy chuẩn
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    try:
        # Lấy đường dẫn tuyệt đối của thư mục chứa file preprocessing.py hiện tại
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Định vị chính xác file dữ liệu nằm trong thư mục data (bằng cách đi từ thư mục gốc dự án)
        data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'diabetes.csv'))
        
        print(f"Dang doc du lieu tu: {data_path}")
        df = load_data(data_path)
        
        # Tiến hành tiền xử lý dữ liệu
        X_train, X_test, y_train, y_test, scaler = clean_and_preprocess(df)
        
        print("\n==================================================")
        print(" CHUC MUNG! File preprocessing.py hoat dong hoan hao!")
        print("==================================================")
        print(f"-> Kich thuoc tap Train (X_train): {X_train.shape}")
        print(f"-> Kich thuoc tap Test (X_test):   {X_test.shape}")
        
    except Exception as e:
        print(f"\n[LOI] Co loi bat ngo xay ra: {e}")