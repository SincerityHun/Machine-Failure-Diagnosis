import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import scipy.signal
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    RepeatVector,
    TimeDistributed,
    Input,
    LeakyReLU,
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from keras.layers import Layer
import tensorflow.keras.backend as K


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


# 데이터 파일 로드 및 정규화
def normalize_data(df, features):
    scaler = StandardScaler()
    return scaler.fit_transform(df[features])


# 데이터 파일 로드
normal_data = pd.read_csv(
    "./csv/5000hz/300rpm/300rpm normal data/stream2024_4_22_23_22.csv"
)
carriage_data = pd.read_csv(
    "./csv/5000hz/300rpm/300rpm carriage damage/stream2024_4_23_2_56.csv"
)
highspeed_data = pd.read_csv(
    "./csv/5000hz/300rpm/300rpm high-speed damage/stream2024_4_23_0_20.csv"
)
lack_data = pd.read_csv(
    "./csv/5000hz/300rpm/300rpm lack of lubrication/stream2024_4_23_2_6.csv"
)
corrosion_data = pd.read_csv(
    "./csv/5000hz/300rpm/300rpm oxidation and corrosion/stream2024_4_23_1_15.csv"
)

features = ["motor1_x", "motor1_y", "motor1_z", "sound"]
normal_data_scaled = normalize_data(normal_data, features)
carriage_data_scaled = normalize_data(carriage_data, features)
highspeed_data_scaled = normalize_data(highspeed_data, features)
lack_data_scaled = normalize_data(lack_data, features)
corrosion_data_scaled = normalize_data(corrosion_data, features)


def create_graph_data(data, k_neighbors=2):
    knn_graph = kneighbors_graph(
        data, k_neighbors, mode="connectivity", include_self=True
    )
    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
    x = torch.tensor(data, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


normal_graph = create_graph_data(normal_data_scaled)
carriage_graph = create_graph_data(carriage_data_scaled)
highspeed_graph = create_graph_data(highspeed_data_scaled)
lack_graph = create_graph_data(lack_data_scaled)
corrosion_graph = create_graph_data(corrosion_data_scaled)


# GCN 모델 정의
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# GCN 모델 학습 및 특징 추출
def train_gcn(graph_data, epochs=10):
    model = GCN(
        in_channels=graph_data.x.shape[1],
        hidden_channels=16,
        out_channels=graph_data.x.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        loss = F.mse_loss(out, graph_data.x)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
    return model, out.detach().numpy()


print("Training GCN model for normal data...")
normal_model, normal_gcn_features = train_gcn(normal_graph)
print("Training GCN model for carriage damage data...")
carriage_model, carriage_gcn_features = train_gcn(carriage_graph)
print("Training GCN model for high-speed damage data...")
highspeed_model, highspeed_gcn_features = train_gcn(highspeed_graph)
print("Training GCN model for lack of lubrication data...")
lack_model, lack_gcn_features = train_gcn(lack_graph)
print("Training GCN model for corrosion data...")
corrosion_model, corrosion_gcn_features = train_gcn(corrosion_graph)


# 슬라이딩 윈도우 함수 정의
def sliding_window(data, window_size, step_size):
    n_samples = data.shape[0]
    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)


window_size = 128
step_size = 64

# 슬라이딩 윈도우 적용
normal_windows = sliding_window(normal_gcn_features, window_size, step_size)
carriage_windows = sliding_window(carriage_gcn_features, window_size, step_size)
highspeed_windows = sliding_window(highspeed_gcn_features, window_size, step_size)
lack_windows = sliding_window(lack_gcn_features, window_size, step_size)
corrosion_windows = sliding_window(corrosion_gcn_features, window_size, step_size)

# STFT 수행
fs = 5000


def compute_stft(data):
    return np.array(
        [
            np.abs(scipy.signal.stft(d, fs=fs, nperseg=min(d.shape[-1], 256))[2])
            for d in data
        ]
    )  # 절대값을 사용하여 실수 데이터로 변환


print("Computing STFT...")
normal_stft = compute_stft(normal_windows)
carriage_stft = compute_stft(carriage_windows)
highspeed_stft = compute_stft(highspeed_windows)
lack_stft = compute_stft(lack_windows)
corrosion_stft = compute_stft(corrosion_windows)


# 데이터 합치기
def concatenate_data(*stft_data):
    min_time_steps = min(stft.shape[2] for stft in stft_data)
    return np.concatenate([stft[:, :, :min_time_steps] for stft in stft_data], axis=0)


X_normal = concatenate_data(normal_stft)
X_carriage = concatenate_data(carriage_stft)
X_highspeed = concatenate_data(highspeed_stft)
X_lack = concatenate_data(lack_stft)
X_corrosion = concatenate_data(corrosion_stft)

# 레이블 생성
y_normal = np.zeros(X_normal.shape[0])
y_carriage = np.ones(X_carriage.shape[0])
y_highspeed = np.full((X_highspeed.shape[0],), 2)
y_lack = np.full((X_lack.shape[0],), 3)
y_corrosion = np.full((X_corrosion.shape[0],), 4)

# 데이터셋 합치기
X = np.concatenate((X_normal, X_carriage, X_highspeed, X_lack, X_corrosion), axis=0)
y = np.concatenate((y_normal, y_carriage, y_highspeed, y_lack, y_corrosion), axis=0)

# 레이블을 원-핫 인코딩으로 변환
y = to_categorical(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 데이터 형태 확인
print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# 데이터 형태 변환
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], -1
)  # (샘플 수, 시간, 특징 수)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
print("Train data shape after reshape:", X_train.shape)
print("Test data shape after reshape:", X_test.shape)


# LSTM Autoencoder 모델 정의 (Attention 추가)
def create_lstm_ae_with_attention(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, return_sequences=True)(inputs)
    encoded = LSTM(32, return_sequences=True)(encoded)
    attention = Attention()(encoded)
    repeated = RepeatVector(input_shape[0])(attention)
    decoded = LSTM(32, return_sequences=True)(repeated)
    decoded = LSTM(64, return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(input_shape[1]))(decoded)
    model = Model(inputs, outputs)
    return model


input_shape = (X_train.shape[1], X_train.shape[2])  # (시간, 특징 수)
model = create_lstm_ae_with_attention(input_shape)

model.compile(optimizer="adam", loss="mse")

# 얼리스탑 콜백 설정
early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

# 모델 훈련
print("Training LSTM Autoencoder model with Attention...")
history = model.fit(
    X_train,
    X_train,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, X_test),
    callbacks=[early_stopping],
)

# 모델 서머리
model.summary()

# 훈련 및 검증 데이터에 대한 손실 확인
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# 훈련 및 검증 데이터에 대한 손실 그래프
plt.plot(train_loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# LSTM Autoencoder에서 인코딩된 특징 추출
encoder = Model(inputs=model.input, outputs=model.layers[3].output)

X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# T-SNE를 사용한 시각화
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(np.concatenate((X_train_encoded, X_test_encoded)))

# 시각화
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=np.argmax(np.concatenate((y_train, y_test)), axis=1),
    cmap="viridis",
)
plt.colorbar(scatter)
plt.title("T-SNE Visualization of Sensor Embedding Vectors")
plt.xlabel("T-SNE Component 1")
plt.ylabel("T-SNE Component 2")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=[
        "Normal",
        "Carriage Damage",
        "High-Speed Damage",
        "Lack of Lubrication",
        "Oxidation and Corrosion",
    ],
)
plt.show()

# 다중 클래스 분류 모델 정의
classifier = Sequential()
classifier.add(Dense(64, input_shape=(X_train_encoded.shape[1],)))
classifier.add(LeakyReLU(alpha=0.01))  # Leaky ReLU 활성화 함수 사용
classifier.add(Dropout(0.5))
classifier.add(Dense(32))
classifier.add(LeakyReLU(alpha=0.01))  # Leaky ReLU 활성화 함수 사용
classifier.add(Dropout(0.2))
classifier.add(Dense(y_train.shape[1], activation="softmax"))

classifier.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# 다중 클래스 분류 모델 훈련
print("Training classifier model...")
history = classifier.fit(
    X_train_encoded,
    y_train,
    epochs=1000,
    batch_size=256,
    validation_data=(X_test_encoded, y_test),
    callbacks=[early_stopping],
)

# 분류 모델 서머리
classifier.summary()

# 훈련 및 검증 데이터에 대한 손실 및 정확도 확인
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

# 훈련 및 검증 데이터에 대한 손실 및 정확도 그래프
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 테스트 데이터에 대한 예측 및 평가
y_test_pred = classifier.predict(X_test_encoded)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

# 분류 결과의 T-SNE 시각화
X_test_tsne = tsne.fit_transform(X_test_encoded)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_test_pred_classes, cmap="viridis"
)
plt.colorbar(scatter)
plt.title("T-SNE Visualization of Classification Results")
plt.xlabel("T-SNE Component 1")
plt.ylabel("T-SNE Component 2")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=[
        "Normal",
        "Carriage Damage",
        "High-Speed Damage",
        "Lack of Lubrication",
        "Oxidation and Corrosion",
    ],
)
plt.show()

print("Accuracy on test set:", accuracy_score(y_test_true_classes, y_test_pred_classes))
print(
    classification_report(
        y_test_true_classes,
        y_test_pred_classes,
        target_names=[
            "Normal",
            "Carriage Damage",
            "High-Speed Damage",
            "Lack of Lubrication",
            "Oxidation and Corrosion",
        ],
    )
)
