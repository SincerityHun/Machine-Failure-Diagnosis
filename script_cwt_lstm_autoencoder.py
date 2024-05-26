import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from scipy.fft import fft, ifft
import pywt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pywt

# Data Processing
print("Data Processing")
RPM = 3000
base_path = "5000hz_raw_data/" + str(RPM) + "rpm/"
folders = [
    str(RPM) + "rpm " + "normal data",
    str(RPM) + "rpm " + "carriage damage",
    str(RPM) + "rpm " + "high-speed damage",
    str(RPM) + "rpm " + "lack of lubrication",
    str(RPM) + "rpm " + "oxidation and corrosion",
]
columns = ["motor1_x", "motor1_y", "motor1_z", "sound", "time"]


# 데이터를 읽고 결합하는 함수
def read_and_concatenate(folder):
    all_files = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder, file_name)
            df = pd.read_csv(file_path, usecols=columns)
            all_files.append(df)
            break # TEST 한개의 파일만 사용
    combined_df = pd.concat(all_files)
    combined_df.sort_values("time", inplace=True)  # 시간 열 기준 정렬
    return combined_df


# CWT를 적용하는 함수
def apply_cwt(data, scales, wavelet_name="morl"):
    coefficients, frequencies = pywt.cwt(data, scales, wavelet_name)
    return coefficients


concatenated_df = dict()
folder_index = [
    "normal data",
    "carriage damage",
    "high-speed damage",
    "lack of lubrication",
    "oxidation and corrosion",
]
# 각 폴더에서 데이터를 처리
for index, folder_name in enumerate(folders):
    folder_path = os.path.join(base_path, folder_name)
    concatenated_df[folder_index[index]] = read_and_concatenate(folder_path)

    # time 열 제거
    concatenated_df[folder_index[index]].drop(columns="time", inplace=True)
    # Label 열 추가
    concatenated_df[folder_index[index]]["label"] = index

# 데이터 결합
combined_data = pd.concat(
    [
        concatenated_df[folder_index[0]],
        concatenated_df[folder_index[1]],
        concatenated_df[folder_index[2]],
        concatenated_df[folder_index[3]],
        concatenated_df[folder_index[4]],
    ],
    ignore_index=True,
)
features = combined_data[["motor1_x", "motor1_y", "motor1_z", "sound"]]
labels = combined_data["label"]

# 데이터 정규화
scalser = StandardScaler()
X_scaled = scalser.fit_transform(features)

BATCH_SIZE = 64
# 데이터를 훈련 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(
    features.values, labels.values, test_size=0.2, random_state=42
)

# 훈련 데이터를 훈련 및 검증 세트로 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# CWT 변환
def apply_cwt_to_dataset(data, scales, wavelet_name="morl"):
    cwt_features = []
    for feature in data.T:  # Apply CWT on each feature (column) of the dataset
        cwt_matrix, _ = pywt.cwt(feature, scales, wavelet_name)
        # Normalize the CWT matrix
        cwt_matrix = (cwt_matrix - np.mean(cwt_matrix)) / np.std(cwt_matrix)
        cwt_features.append(cwt_matrix)
    # Stack to form [samples, features, time, CWT_coefficients]
    return np.stack(cwt_features, axis=1)

print("CWT 변환")
scales = np.arange(1, 128)  # Example range of scales
X_train_cwt = apply_cwt_to_dataset(X_train, scales)
X_val_cwt = apply_cwt_to_dataset(X_val, scales)
X_test_cwt = apply_cwt_to_dataset(X_test, scales)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_cwt, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_cwt, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_cwt, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# PyTorch의 Dataset 및 DataLoader 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# LSTM 오토인코더 모델 정의
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        enc_output, _ = self.encoder(x)
        latent = self.latent(enc_output[:, -1, :])
        latent = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_output, _ = self.decoder(latent)
        output = self.output_layer(dec_output)
        return output


# 분류기
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Hyperparameters
input_dim = X_train.shape[2]  # ["motor1_x", "motor1_y", "motor1_z", "sound"]
hidden_dim = 128
num_layers = 2
learning_rate = 0.001
num_epochs = 100
latent_dim = 64
hidden_dim_classifier = 64
num_classes = len(np.unique(labels.values))  # Number of unique labels

# 모델 초기화
model = LSTMAutoencoder(input_dim, hidden_dim, num_layers, latent_dim)
classifier = Classifier(latent_dim, hidden_dim_classifier, num_classes)

# Loss function 및 optimizer 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion_classifier = nn.CrossEntropyLoss()

# Training
# 체크포인트 파일 경로 설정
checkpoint_path = "./checkpoint/cwt_lstm_autoencoder/model_checkpoint"

# 모델 학습
best_val_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    classifier.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_x.size(0)

        # Use the encoder to get the latent representation
        enc_output, _ = model.encoder(batch_x)
        latent = model.latent(enc_output[:, -1, :])
        # Use the classifier to predict the class
        output_classifier = classifier(latent)
        loss_classifier = criterion_classifier(output_classifier, batch_y)
        loss_classifier.backward()
        optimizer.step()

        # 정확도 계산
        _, predicted = torch.max(output_classifier, 1)
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()

    # Train Loss & Accuracy 계산
    train_loss /= len(train_loader.dataset)
    train_accuracy = correct_train / total_train

    # 검증 세트로 모델 평가
    model.eval()
    classifier.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            loss = criterion(output, batch_x)
            val_loss += loss.item() * batch_x.size(0)

            # Use the encoder to get the latent representation
            enc_output, _ = model.encoder(batch_x)
            latent = model.latent(enc_output[:, -1, :])
            # Use the classifier to predict the class
            output_classifier = classifier(latent)

            # 정확도 계산
            _, predicted = torch.max(output_classifier, 1)
            total_val += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()

    # Validation Loss & Accuracy 계산
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct_val / total_val

    # 모델의 체크포인트 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 체크포인트 저장
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path + str(epoch) + ".pth",
        )
        print("Checkpoint saved.")

    if (epoch + 1) % 1 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

# Test
# 체크포인트 불러오기
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
classifier.load_state_dict(checkpoint["classifier_state_dict"])

model.eval()
classifier.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
with torch.no_grad():
    for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
        output = model(batch_x)
        loss = criterion(output, batch_x)
        test_loss += loss.item() * batch_x.size(0)

        # Use the encoder to get the latent representation
        enc_output, _ = model.encoder(batch_x)
        latent = model.latent(enc_output[:, -1, :])
        # Use the classifier to predict the class
        output_classifier = classifier(latent)

        # Calculate accuracy
        _, predicted = torch.max(output_classifier, 1)
        total_test += batch_y.size(0)
        correct_test += (predicted == batch_y).sum().item()

# Test Loss & Accuracy 계산
test_loss /= len(test_loader.dataset)
test_accuracy = correct_test / total_test

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
