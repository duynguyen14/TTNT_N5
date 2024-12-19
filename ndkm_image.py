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
# dataset_path = "dataset/"
dataset_path = r"D:\Ki5\AI\TTNT_NHOM13\TTNT_NHOM13\dataset"
# Kích thước ảnh sau resize
resize_dim = (100, 100)

# Tham số PCA
n_components = 4  # Có thể thử nghiệm với các giá trị khác

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
n_components = 4  # Bạn có thể thử nghiệm với giá trị này để giảm sự phức tạp

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






def predict_image_with_face_detection(file_path, cascade_path="haarcascade_frontalface_default.xml"):
    try:
        # Tải bộ phát hiện khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

        # Đọc ảnh gốc
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error: Unable to read image {file_path}")
            return

        # Chuyển sang ảnh xám
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Cân bằng histogram để tăng độ tương phản cho ảnh
        gray_img = cv2.equalizeHist(gray_img)

        # Phát hiện khuôn mặt với các tham số đã điều chỉnh
        faces = face_cascade.detectMultiScale(
            gray_img, 
            scaleFactor=1.05,  # Thử giảm giá trị scaleFactor
            minNeighbors=3,    # Giảm số lượng hàng xóm cần thiết để phát hiện
            minSize=(40, 40),  # Thử giảm kích thước khuôn mặt tối thiểu
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            print("No faces detected!")
            return

        # Lọc các khuôn mặt nếu chúng đè lên nhau, giữ khuôn mặt nhỏ hơn
        filtered_faces = []
        for i, (x1, y1, w1, h1) in enumerate(faces):
            keep = True
            area1 = w1 * h1
            for j, (x2, y2, w2, h2) in enumerate(faces):
                if i == j:
                    continue

                # Kiểm tra nếu khuôn mặt (i) bị đè bởi khuôn mặt (j)
                if (x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                    area2 = w2 * h2
                    if area1 >= area2:
                        keep = False
                        break
            if keep:
                filtered_faces.append((x1, y1, w1, h1))

        # Duyệt qua các khuôn mặt đã được lọc
        for (x, y, w_face, h_face) in filtered_faces:
            # Crop vùng khuôn mặt
            face_img = gray_img[y:y + h_face, x:x + w_face]

            # Resize về kích thước chuẩn (100x100 hoặc kích thước đã sử dụng trong huấn luyện PCA)
            face_img_resized = cv2.resize(face_img, (100, 100))  # Đảm bảo kích thước phù hợp với PCA

            # Flatten ảnh và chuẩn hóa (nếu đã sử dụng chuẩn hóa trong quá trình huấn luyện)
            face_img_flattened = face_img_resized.flatten().reshape(1, -1)  # Chuyển thành vector 1 chiều

            # Chuẩn hóa ảnh nếu cần thiết (dùng scaler đã học từ quá trình huấn luyện)
            face_img_scaled = scaler.transform(face_img_flattened)

            # Áp dụng PCA lên ảnh đã chuẩn hóa
            face_img_pca = pca.transform(face_img_scaled)

            # Dự đoán lớp
            pred = clf.predict(face_img_pca)
            label = target_names[pred[0]]

            # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị nhãn
            cv2.rectangle(img, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị ảnh kết quả
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
# Gọi hàm dự đoán
test_file_path = r"D:\Ki5\AI\TTNT_NHOM13\TTNT_NHOM13\test\trinhvanduy1.jpg"
print(f"Processing file: {test_file_path}")
predict_image_with_face_detection(test_file_path)

