import h5py
import numpy as np
import random
import matlab.engine


class Env:
    def __init__(self, SNR, mod, rate, feedbackSNR):
        # self.BER_table, self.BER_table50 = self.load_ber_table()
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
        # 信道&CP&导频
        self.channel, self.CP, self.pilot = self.load_simu_data()

    # def load_ber_table(self):
    #     dict_data = h5py.File('data/BER_table.mat', 'r')
    #     BER_table = np.array(dict_data.get('BER_table'))
    #     BER_table = BER_table.transpose((0, 2, 1))
    #
    #     dict_data = h5py.File('data/BER_table50.mat', 'r')
    #     BER_table50 = np.array(dict_data.get('BER_target'))
    #     BER_table50 = BER_table50.transpose()
    #     return BER_table, BER_table50
    
    def load_simu_data(self):
        chan_dict_data = h5py.File('data/channel.mat', 'r')
        channel = np.array(chan_dict_data.get('Qchannel')).transpose()
        coef_dict_data = h5py.File('data/instant_conf.mat', 'r')
        CP = np.array(coef_dict_data.get('CP')).transpose()
        pilot = np.array(coef_dict_data.get('Pilot')).transpose()
        return channel, CP, pilot

    def sendData(self, action):
        # realBER = self.BER_table[action[0], (self.channelIndex-1), (action[1]*3+action[2])]
        realBER = self.eng.BER_generate_new(self.Len_block, self.nBlock,
                                            self.SNR[action[0]], self.mod[action[1]], self.rate[action[2]],
                                            matlab.double(self.channel[self.channelIndex-1, :].tolist()),
                                            matlab.double(self.CP.tolist()),
                                            matlab.double(self.pilot.tolist()))
        return realBER

    def getFeedbackPilot(self):
        # predict BER
        feedbackPilot = self.eng.fadePilot(self.feedbackSNR,
                                           matlab.double(self.channel[self.channelIndex-1, :].tolist()),
                                           matlab.double(self.pilot.tolist()))

        # real BER
        # feedbackPilot = self.BER_table50[(self.channelIndex-1), :]
        return np.array(feedbackPilot)

    def reset(self, episode):
        self.channelIndex = 32 * episode + 1
        # 随机选一个动作（只为了保证仿真逻辑的正确，跑程序时可注释掉）
        action = np.array([random.randint(0, len(self.SNR) - 1),
                           random.randint(0, len(self.mod) - 1),
                           random.randint(0, len(self.rate) - 1)])
        # 传输数据（只为了保证仿真逻辑的正确，跑程序时可注释掉）
        _ = self.sendData(action)
        # 状态转移
        self.stateTrans()
        pilot = self.getFeedbackPilot()
        return pilot

    def stateTrans(self):
        self.channelIndex += 1


if __name__ == "__main__":
    #测试功能是否正常
    snr = [60, 62, 64, 66, 68, 70, 72, 74]
    modulation = [1, 2, 3]
    code_rate = [1 / 3, 1 / 2, 2 / 3]
    fbSNR = 52
    env = Env(snr, modulation, code_rate, fbSNR)
    Action = np.array([random.randint(0, len(snr) - 1),
                       random.randint(0, len(modulation) - 1),
                       random.randint(0, len(code_rate) - 1)])
    BERreal = env.sendData(Action)
    Pilot = env.reset(1)
    print(BERreal)
