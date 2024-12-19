import os
import cv2
import numpy as np
from time import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Đường dẫn dữ liệu
dataset_path = "dataset/"

# Kích thước ảnh sau resize
resize_dim = (100, 100)

# Tham số PCA
n_components = 7  # Có thể thử nghiệm với các giá trị khác

# Khởi tạo Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_dataset_with_face_detection(dataset_path, resize_dim=(100, 100)):
    X, Y, target_names = [], [], []
    label_count = 0
    missing_faces_count = 0

    print("\nLoading dataset with face detection...")
    for directory in os.listdir(dataset_path):
        dir_path = os.path.join(dataset_path, directory)
        if not os.path.isdir(dir_path):
            continue

        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError(f"Cannot read image: {file_path}")

                # Chuyển đổi sang ảnh xám và làm tăng độ sáng (Histogram Equalization)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.equalizeHist(gray_img)

                # Phát hiện khuôn mặt
                faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

                if len(faces) == 0:
                    missing_faces_count += 1
                    continue

                # Nếu phát hiện nhiều khuôn mặt, chọn khuôn mặt đầu tiên
                if len(faces) > 1:
                    print(f"Multiple faces detected in {file_path}. Only the first face will be used.")

                # Lấy khuôn mặt đầu tiên và resize
                (x, y, w, h) = faces[0]
                face = gray_img[y:y+h, x:x+w]
                face = cv2.resize(face, resize_dim)

                # Lưu dữ liệu (mỗi người một label duy nhất)
                X.append(face.flatten())
                Y.append(label_count)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Lưu tên lớp (người)
        target_names.append(directory)
        label_count += 1

    print(f"Skipped {missing_faces_count} images due to missing faces.")
    return np.array(X), np.array(Y), target_names

# Đọc dữ liệu
X, Y, target_names = load_dataset_with_face_detection(dataset_path, resize_dim=resize_dim)
n_classes = len(target_names)
print(f"Loaded {len(X)} samples from {n_classes} classes.")

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Chuẩn hóa dữ liệu trước PCA
print("\nNormalizing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
print("\nApplying PCA...")
t0 = time()
pca = PCA(n_components=n_components, whiten=True).fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA completed in {time() - t0:.2f}s.")

# Huấn luyện mô hình SVM
print("\nTraining SVM...")
svm = SVC(kernel='linear', class_weight='balanced')
svm.fit(X_train_pca, y_train)

# Dự đoán và đánh giá mô hình
y_pred = svm.predict(X_test_pca)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Hiển thị các Eigenfaces sau khi dữ liệu đã được chuẩn bị
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape(h, w), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# Giảm số lượng thành phần chính nếu cần
n_components = 7  # Bạn có thể thử nghiệm với giá trị này để giảm sự phức tạp

# Hiển thị các Eigenfaces sau khi dữ liệu đã được chuẩn bị
eigenface_titles = [f"Eigenface {i+1}" for i in range(n_components)]
plot_gallery(pca.components_[:n_components], eigenface_titles, resize_dim[0], resize_dim[1])
plt.show()

# Huấn luyện SVM với GridSearchCV để tìm các tham số tối ưu
print("\nTraining SVM with GridSearchCV...")
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, verbose=5)
clf.fit(X_train_pca, y_train)
print(f"Best parameters: {clf.best_params_}")
print(f"Training completed in {time() - t0:.2f}s.")

# Dự đoán với mô hình SVM tối ưu
y_pred = clf.predict(X_test_pca)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# CAMERA
import numpy as np
import cv2

# Khởi tạo webcam và CascadeClassifier để phát hiện khuôn mặt
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap.set(3, 640)  # Chiều rộng
cap.set(4, 480)  # Chiều cao

# Kích thước ảnh sau resize (phải khớp với kích thước đã huấn luyện PCA)
resize_dim = (100, 100)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(40, 40))

    for (x, y, wf, hf) in faces:
        # Cắt và resize khuôn mặt
        face_crop = frame[y:y+hf, x:x+wf]
        face_resized = cv2.resize(face_crop, resize_dim)

        # Chuyển sang ảnh xám và làm phẳng ảnh
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_flattened = face_gray.flatten().reshape(1, -1)

        # Chuẩn hóa ảnh khuôn mặt (dùng scaler đã học khi huấn luyện)
        face_scaled = scaler.transform(face_flattened)

        # Áp dụng PCA vào ảnh khuôn mặt
        face_pca = pca.transform(face_scaled)

        # Dự đoán lớp của khuôn mặt
        pred = clf.predict(face_pca)
        
        # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị tên người
        frame = cv2.rectangle(frame, (x, y), (x + wf, y + hf), (255, 51, 255), 2)
        cv2.putText(frame, target_names[pred[0]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hiển thị khung hình có kết quả
    cv2.imshow('frame', frame)

    # Dừng webcam khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
