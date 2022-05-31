import numpy as np
import matlab.engine
from collections import deque
from keras.models import load_model
import os


class Agent:
    def __init__(self):
        self.BER_memory = deque()
        self.env = matlab.engine.start_matlab()
        self.Len_block = 1024
        self.nBlock = 10
        self.feedbackSNR = 50
        self.mod = [1, 2, 3]
        self.rate = [1 / 3, 1 / 2, 2 / 3]
        self.realBER = np.zeros(9)
        self.model = load_model('data/BER_Predict_model.h5')
        self.BER_weight = 1     # np.array([0.75, 0.75, 1, 1.5, 1.5, 1.5, 1, 0.75, 0.25]).reshape(-1, 1)

    def get_BER_index(self, predictBER):
        if self.BER_memory:
            berIndex = 0
            bestBERIndex = 0
            errorTemp = 100
            for BER_old in self.BER_memory:
                total_error = np.linalg.norm((abs(BER_old - predictBER)*self.BER_weight), ord=1)

                # label = 0
                # label += 1 if abs(BER_old[0] - predictBER[0]) < 0.001 else 0
                # label += 1 if abs(BER_old[1] - predictBER[1]) < 0.005 else 0
                # label += 1 if abs(BER_old[2] - predictBER[2]) < 0.01 else 0
                # label += 1 if abs(BER_old[3] - predictBER[3]) < 0.03 else 0
                # label += 1 if abs(BER_old[4] - predictBER[4]) < 0.05 else 0
                # label += 1 if abs(BER_old[5] - predictBER[5]) < 0.06 else 0
                # label += 1 if abs(BER_old[6] - predictBER[6]) < 0.10 else 0
                # label += 1 if abs(BER_old[7] - predictBER[7]) < 0.10 else 0
                # label += 1 if abs(BER_old[8] - predictBER[8]) < 0.05 else 0

                if total_error <= 0.5:
                # if label >= 6:  # 有2/3及以上满足误差条件可认为是同一信道
                    if total_error < errorTemp:
                        errorTemp = total_error
                        bestBERIndex = berIndex
                berIndex += 1
            if errorTemp == 100:  # 你有问题
                self.BER_memory.append(predictBER)

                return berIndex
            else:
                return bestBERIndex
        else:
            self.BER_memory.append(predictBER)

            return 0

    def sendData(self, channel):
        for modulation in range(3):
            for rate in range(3):
                self.realBER[modulation * 3 + rate] = self.env.BER_generate_new(self.Len_block, self.nBlock,
                                                                                self.feedbackSNR, self.mod[modulation],
                                                                                self.rate[rate],
                                                                                channel+1)
        return self.realBER

    def getFeedbackPilot(self, channel):
        feedbackPilot = self.env.fadePilot(self.feedbackSNR, channel + 1)
        return np.array(feedbackPilot)

    def predict_BER(self, Pilot):
        # 组合模型输入数据
        pilotData = np.zeros([9, Pilot.shape[1] + 2])
        pilotData[:, 0] = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        pilotData[:, 1] = np.array([1 / 3, 1 / 2, 2 / 3, 1 / 3, 1 / 2, 2 / 3, 1 / 3, 1 / 2, 2 / 3])
        pilotData[:, 2:] = Pilot
        # 预测
        predictBER = self.model.predict_on_batch(pilotData)
        predictBER = 10 ** (-1 / predictBER)
        return predictBER

    def printMemory(self):
        print(self.BER_memory)
        print(len(self.BER_memory))


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    agent = Agent()
    chan = 0
    BERIndexTemp = []
    # BERtemp = []
    for channelIndex in range(5120):
        pilot = agent.getFeedbackPilot(channelIndex)
        predict_BER = agent.predict_BER(pilot)
        # real_BER = agent.sendData(channelIndex)
        BER_index = agent.get_BER_index(predict_BER)
        # BERsum = np.sum(predict_BER)
        # BERtemp.append(BERsum)
        BERIndexTemp.append(BER_index)
        if channelIndex % 50 == 0:
            print(channelIndex)
    agent.printMemory()
    # with open('output/BERsum.txt', 'w') as ber_file:  # 写文件
    #     ber_file.write(str(BERtemp))
    with open('output/BERindex.txt', 'w') as data_file:  # 写文件
        data_file.write(str(BERIndexTemp))
