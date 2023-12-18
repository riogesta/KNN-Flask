import pandas as pd
import os     

from models.config import check_k_value

def check_available_data():
    file_path = './static/data/dataset.csv'
    if os.path.getsize(file_path) == 0:
        return False
    else:
        return True
    
def read_dataset():
    if check_available_data :
        return pd.read_csv('./static/data/dataset.csv')

def knn_method():
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    
    data = read_dataset()
    kolom = data.columns
    x = data.drop(['class'], axis=1)
    y = data['class']
    
    # classification KNN
    knn = KNeighborsClassifier(n_neighbors=check_k_value())
    
    # membagi data test , train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # memasukkan data training pada fungsi klasifikasi untuk KNN
    knn.fit(x_train, y_train)
    # menentukkan prediksi
    y_pred = knn.predict(x_test)
    
    # menjalankan generate_confussion_matrix
    generate_consuffion_matrix(y_pred=y_pred, y_test=y_test)
    
    distances, indices = knn.kneighbors(x_test)
    
    return {
        'kolom': kolom.to_list(),
        'data_uji': x_test,
        'data_latih': x_train,
        'class': data,
        'y_pred': y_pred,
        'X_train': x_train,
        'y_train': y_train,
        'knn': knn,
        'classification': y.to_list(),
        'confussion_matrix': confusion_matrix(y_test, y_pred),
        'akurasi': classification_report(y_test, y_pred, output_dict=True),
        'jarak': distances,
        'indeks': indices
    }
    
def find_k_nearest_neighbors(input, k):
    import numpy as np
    
    data = read_dataset()
    
    # Ambil kolom data tanpa kolom target (class)
    features = data.drop(['class'], axis=1)

    # Hitung jarak menggunakan Euclidean distance
    distances = np.linalg.norm(features - input, axis=1)

    # Urutkan indeks berdasarkan jarak
    sorted_indices = np.argsort(distances)

    # Ambil K tetangga terdekat
    nearest_neighbors = data.iloc[sorted_indices[:k]]

    return nearest_neighbors

def generate_consuffion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_test, y_pred)
    
    TP = cm[1,1]
    FP = cm[0,1]
    TN = cm[0,0]
    FN = cm[1,0]
    
    plt.figure(figure=(8,6))
    sns.heatmap([[TN, FP], [FN, TP]], annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("./static/assets/images/confussion_matrix.png")
    
def prediksi(input):
    
    data = knn_method()
    knn = data['knn']

    knn_pred = knn.predict(input)
    
    return knn_pred

