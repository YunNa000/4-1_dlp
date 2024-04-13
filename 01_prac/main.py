from tensorflow import keras
import random
import os
from matplotlib import pyplot as plt
import numpy as np

# 예를 들어 "123.45(67)"라는 문자열을 받았을 때, "123.45"라는 문자열만 남기고 이를 실수형 값 123.45로 변환
def process(txt):
    if "(" in txt:
        txt = txt.split("(")[0]
    txt = txt.strip()
    return float(txt)


def draw_graph(prediction, label, history):
    X = prediction / np.max(prediction, axis=0)
    Y = label / np.max(label, axis=0)

    minval = min(np.min(X), np.min(Y))
    maxval = max(np.max(X), np.max(Y))

    fig = plt.figure(figsize=(8, 8))
    plt.title("Regression Result")
    plt.xlabel("Ground Truth")
    plt.ylabel("AI Predict")
    plt.scatter(X, Y)
    plt.plot([minval, maxval], [minval, maxval], "red")
    fig.savefig("result.png")

    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("train_history.png")


# 몇 에포크 만큼 학습을 시킬 것인지 결정합니다.
EPOCHS = 5 

file = open("data/" + os.listdir('data')[0])
data = []
file.readline() #파일의 첫 번째 줄을 읽고 건너뜀
for line in file:
    splt = line.split(",")
    chest = process(splt[2])
    arm = process(splt[3])
    height = process(splt[4])
    waist = process(splt[5])
    sat = process(splt[6])
    head = process(splt[7])
    feet = process(splt[8])
    weight = process(splt[9])

    data.append((chest, arm, sat, head, feet, waist, height, weight))

random.shuffle(data) #데이터 셔플링
data = np.asarray(data) #Numpy 배열 변환
normalize_factors = np.max(data, axis=0)
#각 측정값을 해당 측정값의 최댓값으로 나누어 정규화. 이 과정은 모든 데이터 포인트를 0과 1 사이의 값으로 변환
normalized_data = data / normalize_factors 

x, y = normalized_data.shape

train_X = normalized_data[:int(x * 0.8), :-2]
train_Y = normalized_data[:int(x * 0.8), -2:]
test_X = normalized_data[int(x * 0.8):, :-2]
test_Y = normalized_data[int(x * 0.8):, -2:]

file.close()

print("\n\nData Read Done!")
print("Training X Size : " + str(train_X.shape))
print("Training Y Size : " + str(train_Y.shape))
print("Test X Size : " + str(test_X.shape))
print("Test Y Size : " + str(test_Y.shape) + '\n\n')

# 인공신경망을 제작합니다.
model = keras.Sequential([
    keras.layers.Dense(6),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(2, activation='sigmoid')
])

# 인공신경망을 컴파일합니다.
model.compile(optimizer="adam", loss="mse", metrics=['mae'])

# 인공신경망을 학습시킵니다.
print("\n\n************ TRAINING START ************ ")
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(train_X, train_Y, epochs=EPOCHS,
                    validation_data=(test_X, test_Y),
                    callbacks=[early_stop])

# 학습 결과를 그래프로 출력합니다.
draw_graph(model(test_X), test_Y, history)
