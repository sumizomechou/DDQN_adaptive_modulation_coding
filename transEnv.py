import h5py
import numpy as np
import random
import matlab.engine


class Env:
    def __init__(self, SNR, mod, rate, feedbackSNR):
        self.BER_table, self.BER_table50 = self.load_ber_table()
        self.eng = matlab.engine.start_matlab()
        # 块大小
        self.Len_block = 1024
        # 块数量
        self.nBlock = 10
        # 信噪比
        self.SNR = SNR
        self.mod = mod
        self.rate = rate
        self.channelIndex = 1
        self.feedbackSNR = feedbackSNR

    def load_ber_table(self):
        dict_data = h5py.File('data/BER_table.mat', 'r')
        BER_table = np.array(dict_data.get('BER_table'))
        BER_table = BER_table.transpose((0, 2, 1))

        dict_data = h5py.File('data/BER_table50.mat', 'r')
        BER_table50 = np.array(dict_data.get('BER_target'))
        BER_table50 = BER_table50.transpose()
        return BER_table, BER_table50

    def sendData(self, action):
        realBER = self.BER_table[action[0], (self.channelIndex-1), (action[1]*3+action[2])]
        return realBER

    def getFeedbackPilot(self):
        # predict BER
        feedbackPilot = self.eng.fadePilot(self.feedbackSNR, self.channelIndex)

        # real BER
        # feedbackPilot = self.BER_table50[(self.channelIndex-1), :]

        return np.array(feedbackPilot)

    def reset(self, episode):
        self.channelIndex = 32 * episode + 1
        # 随机选一个动作
        action = np.array([random.randint(0, len(self.SNR) - 1),
                           random.randint(0, len(self.mod) - 1),
                           random.randint(0, len(self.rate) - 1)])
        # 传输数据
        _ = self.sendData(action)
        # 状态转移
        self.stateTrans()
        pilot = self.getFeedbackPilot()
        return pilot

    def stateTrans(self):
        self.channelIndex += 1


if __name__ == "__main__":
    snr = [60, 62, 64, 66, 68, 70, 72, 74]
    modulation = [1, 2, 3]
    code_rate = [1 / 3, 1 / 2, 2 / 3]
    fbSNR = 52
    env = Env(snr, modulation, code_rate, fbSNR)
    Action = "58-2"
    BERreal = env.sendData(Action)
    Pilot = env.reset(1)
    print(BERreal)