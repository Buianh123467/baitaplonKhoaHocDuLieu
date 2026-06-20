import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Gọi hàm tiền xử lý từ file preprocessing.py cùng thư mục src
from preprocessing import load_data, clean_and_preprocess

def run_clustering():
    # Lấy đường dẫn tuyệt đối của thư mục chứa file hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Định vị file dữ liệu gốc
    data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', 'diabetes.csv'))
    
    print("========= KHOI TẠO TIEN XU LY DU LIEU CHO PHAN CUM =========")
    df = load_data(data_path)
    X_train, _, _, _, _ = clean_and_preprocess(df)
    
    print("\n========= BAT DAU PHAN CUM DU LIEU (K-MEANS) =========")
    # Khởi tạo thuật toán K-Means với 2 cụm (Gom nhóm nguy cơ)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train)
    
    print(f"-> Da phan cum xong! So luong phan tu thuoc Cum 0 va Cum 1: {dict(enumerate(kmeans.labels_)) if False else 'Thanh cong'}")
    print(f"-> Tam cua cac cum (Centroids) co kich thuoc: {kmeans.cluster_centers_.shape}")
    
    # Vẽ biểu đồ minh họa phân cụm dựa trên 2 đặc trưng đầu tiên (Chuẩn hóa) để trực quan hóa
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title('Ket qua phan cum K-Means (Minh hoa tren 2 Dac trung dau)')
    plt.xlabel('Dac trung 1 (Chuand hoa)')
    plt.ylabel('Dac trung 2 (Chuand hoa)')
    plt.legend()
    
    # Lưu biểu đồ phân cụm vào thư mục images
    images_dir = os.path.abspath(os.path.join(current_dir, '..', 'images'))
    os.makedirs(images_dir, exist_ok=True)
    fig_save_path = os.path.join(images_dir, 'ket_qua_phan_cum.png')
    plt.savefig(fig_save_path)
    plt.close()
    
    print(f"\n[THANH CONG] Da luu bieu do phan cum tai: {fig_save_path}")

if __name__ == "__main__":
    try:
        run_clustering()
    except Exception as e:
        print(f"\n[LOI] Co loi xay ra khi phan cum: {e}")