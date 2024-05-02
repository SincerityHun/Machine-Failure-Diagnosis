import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from scipy.fft import fft, ifft
import pywt

if torch.cuda.is_available():
    # 사용 가능한 GPU의 개수
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Device 세팅
    device = torch.device("cuda")
    print("GPU is available and set as the current device.")

    # 현재 device로 설정된 GPU 확인
    current_device = torch.cuda.current_device()
    print(f"Current GPU Device: index[{current_device}]")
    
    # 현재 device로 설정된 GPU의 이름 출력
    print(f"Device name: {torch.cuda.get_device_name(current_device)}")

    torch.cuda.empty_cache()
    
    
else:
    print("GPU is not available, using CPU instead.")
    device = torch.device("cpu")

sensor1 = pd.read_csv("raw_data/g1_sensor1.csv",names=["time","normal","type1","type2","type3"])
sensor2 = pd.read_csv("raw_data/g1_sensor2.csv",names=["time","normal","type1","type2","type3"])
sensor3 = pd.read_csv("raw_data/g1_sensor3.csv",names=["time","normal","type1","type2","type3"])
sensor4 = pd.read_csv("raw_data/g1_sensor4.csv",names=["time","normal","type1","type2","type3"])

from scipy import interpolate
x_new = np.arange(0,140,0.001) # 0,0.001,0.002,,,,,,139.999
y_new1 = []
y_new2 = []
y_new3 = []
y_new4 = []

# 모든 센서의 각 타입 데이터 별로 선형 보간을 수행한 결과를 추출
for item in ["normal","type1","type2","type3"]:
    f_linear1 = interpolate.interp1d(sensor1["time"],sensor1[item],kind="linear") 
    y_new1.append(f_linear1(x_new)) 

    f_linear2 = interpolate.interp1d(sensor2["time"],sensor2[item],kind="linear")
    y_new2.append(f_linear2(x_new))
    f_linear3 = interpolate.interp1d(sensor3["time"],sensor3[item],kind="linear")
    y_new3.append(f_linear3(x_new))
    f_linear4 = interpolate.interp1d(sensor4["time"],sensor4[item],kind="linear")
    y_new4.append(f_linear4(x_new))

# 시간축을 기준으로, 모든 센서에서 추출된 데이터를 이어붙인다.
normal_ = pd.concat([sensor1["normal"],sensor2["normal"],sensor3["normal"],sensor4["normal"]],axis=1) # 각 센서에서 추출된 normal 데이터 
type1_ = pd.concat([sensor1["type1"],sensor2["type1"],sensor3["type1"],sensor4["type1"]],axis=1) # Type 1 이상치 데이터
type2_ = pd.concat([sensor1["type2"],sensor2["type2"],sensor3["type2"],sensor4["type2"]],axis=1) # Type 2 이상치 데이터
type3_ = pd.concat([sensor1["type3"],sensor2["type3"],sensor3["type3"],sensor4["type3"]],axis=1) # Type 3 이상치 데이터

# 어디 센서에서 나온 결과인지, 열의 이름 달기
normal_.columns = ["s1","s2","s3","s4"]
type1_.columns = ["s1","s2","s3","s4"]
type2_.columns = ["s1","s2","s3","s4"]
type3_.columns = ["s1","s2","s3","s4"]

# apply_cwt 함수 정의: 각 센서 데이터에 대해 CWT를 적용합니다.
def apply_cwt(data, scales, wavelet='cmor'):
    cwt_coeffs = []
    for column in data:
        sensor_data = data[column].values  # Pandas Series를 numpy 배열로 변환
        if sensor_data.ndim != 1:
            raise ValueError(f"Data for sensor {column} is not 1-dimensional.")
        # 연속 웨이블릿 변환 적용
        cwt_matrix, frequencies = pywt.cwt(sensor_data, scales, wavelet)
        # 해당 시계열의 cwt_matrix의 절댓값이 가장 큰 값의 frequencies을 넣어준다.
        # Find the index of the max coefficient at each time point across all scales
        cwt_coeffs.append(np.mean(np.abs(cwt_matrix),axis=0))  # 결과를 1차원으로 평탄
        # cwt_coeffs.append(np.abs(cwt_matrix))  # 결과를 1차원으로 평탄
    # numpy 배열로 변환
    return np.column_stack(cwt_coeffs)

wavelet = 'morl'
scales = np.arange(1,128) # 스케일이 커질 수록, 낮은 주파수 성분을 잡아낼 수 있음-> 이거 그냥 64가 제일 잘 나와서 이걸로 함

# 각 데이터셋에 대해 CWT 변환 적용
normal_cwt = apply_cwt(normal_, scales,wavelet)
type1_cwt = apply_cwt(type1_, scales,wavelet)
type2_cwt = apply_cwt(type2_, scales,wavelet)
type3_cwt = apply_cwt(type3_, scales,wavelet)

M =15 # 이동평균 필터 사이즈
def apply_moving_average(data):
    # 이동 평균 적용 및 데이터 재구성
    temp = [np.convolve(data[col], np.ones(M), 'valid') / M for col in data.columns]
    return np.column_stack(temp)

# 이동 평균 필터 적용 -> 노이즈 제거용
normal_ma = apply_moving_average(normal_)
type1_ma = apply_moving_average(type1_)
type2_ma = apply_moving_average(type2_)
type3_ma = apply_moving_average(type3_)
# CWT 결과와 이동평균 결과 결합
print(np.shape(normal_ma),np.shape(normal_cwt))

# CWT 결과 중 필요한 부분만 슬라이싱하여 이동 평균 결과와 결합
start_index = 14  # 이동 평균을 적용했을 때 데이터가 얼마나 줄어드는지에 따라 조정
normal_features = np.concatenate((normal_ma, normal_cwt[start_index:, :]), axis=1)
type1_features = np.concatenate((type1_ma, type1_cwt[start_index:, :]), axis=1)
type2_features = np.concatenate((type2_ma, type2_cwt[start_index:, :]), axis=1)
type3_features = np.concatenate((type3_ma, type3_cwt[start_index:, :]), axis=1)
print(np.shape(normal_features))

scaler = MinMaxScaler()
scaler.fit(normal_cwt) # normal_데이터셋의 데이터 분포가 어떻게 정규화되어 있는지 학습

# normal_ 데이터셋 분포에 맞게 다른 모든 데이터 셋의 분포를 전환
normal= scaler.fit_transform(normal_features)
type1 = scaler.transform(type1_features)
type2= scaler.transform(type2_features)
type3= scaler.transform(type3_features)

# 끝에 NAN 쓰레기값, 초반에 불안정함때문에 중간 100,000개만 데이터로 사용
# 끝에 NAN 쓰레기값, 초반에 불안정함때문에 중간 100,000개만 데이터로 사용
normal = normal[30000:130000][:]
type1 = type1[30000:130000][:]
type2 = type2[30000:130000][:]
type3 = type3[30000:130000][:]

# 데이터 분배, train = 60,000개, valid = 20,000개, test = 20,000개 
# 데이터 분배, train = 60,000개, valid = 20,000개, test = 20,000개 
normal_train = normal[:][:60000]; normal_valid = normal[:][60000:80000]; normal_test =normal[:][80000:]
type1_train = type1[:][:60000]; type1_valid = type1[:][60000:80000]; type1_test =type1[:][80000:]
type2_train = type2[:][:60000]; type2_valid = type2[:][60000:80000]; type2_test =type2[:][80000:]
type3_train = type3[:][:60000]; type3_valid = type3[:][60000:80000]; type3_test =type3[:][80000:]

# 데이터 합치기
train = np.concatenate((normal_train,type1_train,type2_train,type3_train))
valid = np.concatenate((normal_valid,type1_valid,type2_valid,type3_valid))
test = np.concatenate((normal_test,type1_test,type2_test,type3_test))

# 모델이 예측한 결과값을 담을 데이터 구조 생성
train_label = np.concatenate((np.full((60000,1),0), np.full((60000,1),1),
np.full((60000,1),2), np.full((60000,1),3)))
valid_label = np.concatenate((np.full((20000,1),0), np.full((20000,1),1),
np.full((20000,1),2), np.full((20000,1),3)))
test_label = np.concatenate((np.full((20000,1),0), np.full((20000,1),1),
np.full((20000,1),2), np.full((20000,1),3)))

# train data, valid data, test data 전부 index 셔플
idx = np.arange(train.shape[0]); np.random.shuffle(idx)
train = train[:][idx]; train_label = train_label[:][idx]

idx_v = np.arange(valid.shape[0]); np.random.shuffle(idx_v)
valid = valid[:][idx_v]; valid_label = valid_label[:][idx_v]

idx_t = np.arange(test.shape[0]); np.random.shuffle(idx_t)
test = test[:][idx_t]; test_label = test_label[:][idx_t]

# 토치 텐서로 변환, 그냥 좀 절삭
x_train = torch.from_numpy(train).float()
y_train = torch.from_numpy(train_label).float().T[0]
x_valid = torch.from_numpy(valid).float()
y_valid = torch.from_numpy(valid_label).float().T[0]
x_test = torch.from_numpy(test).float()
y_test = torch.from_numpy(test_label).float().T[0]

# 데이터셋 생성 및 배치사이즈로 미리 나누며 iterator 생성
# 데이터셋 생성 및 배치사이즈로 미리 나누며 iterator 생성
BATCH_SIZE =  1024
train = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train, batch_size =BATCH_SIZE, shuffle=True)
valid = TensorDataset(x_valid, y_valid)
valid_dataloader = DataLoader(valid, batch_size =len(x_valid), shuffle=False)
test = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test, batch_size =len(x_valid), shuffle=False)

class KAMP_CNN(nn.Module):
    def __init__(self):
        super(KAMP_CNN, self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=100, kernel_size=2, stride=1, padding='same'),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=1, stride=1),
        nn.Dropout(p=0.2))

        self.conv2 = nn.Sequential(
        nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2, stride=1, padding='same'),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=1, stride=1),
        nn.Dropout(p=0.2))

        self.conv3 = nn.Sequential(
        nn.Conv1d(in_channels=100, out_channels=100, kernel_size=2, stride=1, padding='same'),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=1, stride=1),
        nn.Dropout(p=0.2))

        self.conv4 = nn.Sequential(
        nn.Conv1d(in_channels=100, out_channels=4, kernel_size=2, stride=1, padding='same'),
        nn.BatchNorm1d(4),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=1, stride=1))

        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(4, 4)
        
    def forward(self, input):
        input = input.unsqueeze(1)
        out =self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out =self.conv4(out)
        out =self.final_pool(out)
        out =self.linear(out.squeeze(-1))
        return out
    
    # GPU: device
def train_model(model, criterion, optimizer, num_epoch, train_dataloader, PATH,accumulation_steps=1):
    # Model을 GPU로 이동
    model.to(device)

    loss_values = []
    loss_values_v = [] 
    accuracy_past =0
    for epoch in range(1, num_epoch +1):
        #---------------------- 모델 학습 ---------------------#
        model.train()
       
        running_loss =0.0
        optimizer.zero_grad()  # 최적화 초기화는 배치 처리 바깥에서 수행

        for batch_idx, samples in enumerate(train_dataloader):
            # 데이터 GPU로 옮기기
            x_train, y_train = samples[0].to(device), samples[1].to(device) 

            # 변수 초기화
            y_hat = model.forward(x_train)
            loss = criterion(y_hat,y_train.long())
            loss.backward()

            # 누적 스템마다 파라미터 업데이트
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad() 
            
            running_loss += loss.item()
        

        # 누적된 마지막 그래디언트를 업데이트
        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_values.append(running_loss / len(train_dataloader))
    #---------------------- 모델 검증 ---------------------#
        model.eval()
        accuracy =0.0
        total =0.0
        for batch_idx, data in enumerate(valid_dataloader):
            x_valid, y_valid = data[0].to(device), data[1].to(device)

            v_hat = model.forward(x_valid)
            v_loss = criterion(v_hat,y_valid.long())
            _, predicted = torch.max(v_hat.data, 1)
            total += y_valid.size(0)
            accuracy += (predicted == y_valid).sum().item()
        loss_values_v.append(loss.item())
        accuracy = (accuracy / total)
    #----------------Check for early stopping---------------#
        if epoch % 1 ==0:
            print('[Epoch {}/{}] [Train_Loss: {:.6f} /Valid_Loss: {:.6f}]'.format(epoch, num_epochs, loss.item(),v_loss.item()))
            print('[Epoch {}/{}] [Accuracy : {:.6f}]'.format(epoch, num_epochs, accuracy))
        
        # checkpoint + early stopping
        if accuracy_past < accuracy:
            accuracy_past = accuracy
            torch.save(model.state_dict(), PATH + f'model_epoch_{epoch}_acc_{accuracy:.4f}.pt')
            print(f"Checkpoint saved at epoch {epoch} with validation accuracy {accuracy:.4f}.")

    # return loss..
    return loss_values, loss_values_v


CNN_model = KAMP_CNN()
num_epochs =1000
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNN_model.parameters())
accumulation_steps = 4
PATH ='save/CNN/'
CNN_loss_values, CNN_loss_values_v = train_model(CNN_model, criterion, optimizer,
num_epochs, train_dataloader, PATH,accumulation_steps)

# TEST
def test_model(model, PATH):
    model = torch.load(PATH +'model.pt')
    #---------------------- 모델 시험 ---------------------#
    model.eval()
    total =0.0
    accuracy =0.0
    for batch_idx, data in enumerate(test_dataloader):
        x_test, y_test = data[0].to(device),data[1].to(device)

        t_hat = model(x_test)
        _, predicted = torch.max(t_hat.data, 1)
        total += y_test.size(0)
        accuracy += (predicted == y_test).sum().item()
    accuracy = (accuracy / total)
    #------------------------------------------------------#
    print(accuracy)


test_model(CNN_model,'save/CNN/')