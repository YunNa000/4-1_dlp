import os
import numpy as np
from PIL import Image
import random
from matplotlib import pyplot as plt

class DataReader():
    def __init__(self): # 생성자, self는 인스턴스 자기 자신을 참조하는데 사용
    
        self.label = ["Rock", "Paper", "Scissors"]

        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []

        self.read_images()

    def read_images(self):
        data = []
        print("Reading Data...")
        classes = os.listdir("data")
        for i, cls in enumerate(classes): #enumerate 함수는 리스트의 원소와 인덱스를 동시에 반환
            print("Opening " + cls + "/")
            for el in os.listdir("data/" + cls): #주어진 폴더내의 모든 파일을 순회
                img = Image.open("data/" + cls + "/" + el) #PIL의 Image.open 를 사용하여 이미지 오픈
                data.append((np.asarray(img), i))
                #이미지 파일을 numpy 배열로 변환(np.asarray(img))하고, 
                #이 배열과 현재 클래스의 인덱스(레이블)를 튜플로 묶어 data 리스트에 추가
                img.close()

        random.shuffle(data)

        for i in range(len(data)):
            if i < 0.8*len(data):
                self.train_X.append(data[i][0])
                self.train_Y.append(data[i][1])
                #data[i][0] (이미지 데이터)을 self.train_X 리스트에, 
                #data[i][1] (레이블)을 self.train_Y 리스트에 추가
            else:
                self.test_X.append(data[i][0])
                self.test_Y.append(data[i][1])

        self.train_X = np.asarray(self.train_X) / 255.0
        self.train_Y = np.asarray(self.train_Y)
        self.test_X = np.asarray(self.test_X) / 255.0
        self.test_Y = np.asarray(self.test_Y)

        
        print("\n\nData Read Done!")
        print("Training X Size : " + str(self.train_X.shape))
        print("Training Y Size : " + str(self.train_Y.shape))
        print("Test X Size : " + str(self.test_X.shape))
        print("Test Y Size : " + str(self.test_Y.shape) + '\n\n')


def draw_graph(history):
    train_history = history.history["loss"]
    validation_history = history.history["val_loss"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Loss History")
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS Function")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("loss_history.png")

    train_history = history.history["accuracy"]
    validation_history = history.history["val_accuracy"]
    fig = plt.figure(figsize=(8, 8))
    plt.title("Accuracy History")
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.plot(train_history, "red")
    plt.plot(validation_history, 'blue')
    fig.savefig("accuracy_history.png")
