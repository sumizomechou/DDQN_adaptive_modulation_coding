import os
import pickle
import random
import numpy as np
import scipy.io as scio
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras import optimizers
from collections import deque
from transEnv import Env


class DoubleDQN:
    def __init__(self, TargetDATA, TimeLIMIT, BER_limit, SNR, mod, rate, feedbackSNR):
        self.TargetDATA = TargetDATA
        self.TimeLimit = TimeLIMIT
        self.BER_limit = BER_limit
        self.SNR = SNR
        self.mod = mod
        self.rate = rate
        self.BER_memory = deque()
        self.env = Env(self.SNR, self.mod, self.rate, feedbackSNR)
        # DQN learning rate
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        # BER_Predict_model.h5使用导频估计BER(低SNR)，用于之后的基于BER的信道分类
        self.BER_predict_model = load_model('data/BER_Predict_model.h5')
        # experience replay.
        self.memory_buffer = deque(maxlen=5000)
        # discount rate for q value.
        self.gamma = 0.9
        # epsilon of ε-greedy.
        self.epsilon = 1.0
        # discount rate for epsilon.
        self.epsilon_decay = 0.99
        # min epsilon of ε-greedy.
        self.epsilon_min = 0.05
        self.BERWeight = np.array([2, 2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2]).reshape(-1, 1)

    def load(self):  # 没啥卵用
        if os.path.exists('data/ddqn.h5'):
            print('接续训练：')
            self.model = load_model('data/ddqn.h5')
            self.target_model = self.model
            self.epsilon = self.epsilon_min
            with open('data/BER_memory.bin', 'rb') as pickle_file:  # 读文件
                self.BER_memory = pickle.load(pickle_file)

    def save(self, TransmitData, EnergyData):  # 保存训练记录
        self.model.save('output/ddqn.h5')
        with open('output/BER_memory.bin', 'wb') as pickle_file:  # 写文件
            pickle.dump(self.BER_memory, pickle_file)
        QData = np.array(TransmitData, dtype=float)
        QEnergy = np.array(EnergyData)
        QEnergyEfficiency = QData / QEnergy
        scio.savemat('output/QTrainLog.mat',
                     {'QData': QData, 'QEnergy': QEnergy, 'QEnergyEfficiency': QEnergyEfficiency})

    def reset(self, Pilot, Time, Data):  # 初始化每个回合的状态
        done = False

        # predict BER
        predictBER = self.predict_BER(Pilot)
        # real BER
        # predictBER = self.env.getFeedbackPilot()

        BER_index = self.get_BER_index(predictBER)
        if Time == self.TimeLimit:
            done = True
        State = np.array([[BER_index, Time, Data]])
        return State, done

    def predict_BER(self, Pilot):  # 估计BER向量
        # 拼接模型输入数据：[调制阶数， 编码速率， 导频]
        pilotData = np.zeros([9, Pilot.shape[1] + 2])
        pilotData[:, 0] = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        pilotData[:, 1] = np.array([1/3, 1/2, 2/3, 1/3, 1/2, 2/3, 1/3, 1/2, 2/3])
        pilotData[:, 2:] = Pilot
        # 预测
        # 模型输出为预处理后的BER向量，预处理为BER_label = -1/(log10(BER));
        predictBER = self.BER_predict_model.predict_on_batch(pilotData)
        # 将模型输出处理回BER形式的数据
        predictBER = 10 ** (-1 / predictBER)
        return predictBER

    def get_BER_index(self, predictBER):  # 基于BER的信道分类
        if self.BER_memory:
            berIndex = 0
            bestBERIndex = 0
            errorTemp = 100  # 若存在满足条件的BER_old，errorTemp会被修改，反之不变
            # 从self.BER_memory中选出最接近predictBER的BER_old，返回其索引作为BER_index
            for BER_old in self.BER_memory:
                total_error = np.linalg.norm(np.abs(BER_old - predictBER).reshape(-1, 1)*self.BERWeight, ord=1)

                if total_error <= 0.5:  # 调整total_error可以改变总的分类数

                    if total_error < errorTemp:
                        errorTemp = total_error
                        bestBERIndex = berIndex
                berIndex += 1
            if errorTemp == 100:
                # 如果self.BER_memory中不存在满足条件的BER_old
                self.BER_memory.append(predictBER)
                return berIndex
            else:
                return bestBERIndex
        else:
            self.BER_memory.append(predictBER)
            return 0

    def emission_energy(self, action_snr, distance=3.75, f=12.5, B=5):  # 计算消耗能量、剩余能量、奖励
        SNR = self.SNR[action_snr]
        # 传输距离1km,带宽10KHz
        k = 1.5
        s = 0.5
        w = 0
        eta = 0.2  # 电声转换效率
        a_f = 0.11 * f * f / (1 + f * f)
        a_f = a_f + 44 * f * f / (4100 + f * f)
        a_f = a_f + 2.75e-4 * f * f
        a_f = a_f + 0.003
        A_lf = k * 10 * np.log10(distance * 1000) + distance * a_f  # 通信损耗

        Nt_f = 17 - 30 * np.log10(f)  # 海水流动的噪音
        Ns_f = 40 + 20 * (s - 0.5) + 26 * np.log10(f) - 60 * np.log10(f + 0.03)  # 海上船舶的噪音
        Nw_f = 50 + 7.5 * w ** 0.5 + 20 * np.log10(f) - 40 * np.log10(f + 0.4)  # 各种风浪的噪音
        Nth_f = -15 + 20 * np.log10(f)  # 热噪音
        N_f = 10 * np.log10(10 ** (Nt_f / 10) + 10 ** (Ns_f / 10) + 10 ** (Nw_f / 10) + 10 ** (Nth_f / 10))     # 谱级
        N_f = N_f + 10 * np.log10(B * 1000)     # 噪声级

        SNR = SNR - 24  # 这里的SNR值是有问题的，先凑合用
        SL = SNR + A_lf + N_f  # 声源级
        Ph = 10 ** ((SL - 170.8 - 10 * np.log10(eta)) / 10)  # 换能器发射功率
        return Ph.item()

    def feedback(self, real_BER, Action, Energy_Cons, Data_Trans, N):
        """
        Arguments:
            real_BER: Transmission result
            Action: Transmission action
            Energy_Cons: Energy consumed by action
            Data_Trans: Transferred data
            N: Number of episodes performed
        Returns:
            R: Reward
            Data_Trans: Transferred data
        """
        if real_BER > self.BER_limit:
            succeedData = 0
        else:
            succeedData = self.mod[Action[1]] * self.rate[Action[2]]
        Data_Trans += succeedData  # 更新已传输的数据
        LastR = 0
        if N == self.TimeLimit:
            if Data_Trans < self.TargetDATA:
                LastR = -1 * (self.TargetDATA - Data_Trans)
            else:
                LastR = 0.5
        R = 0.1*succeedData/Energy_Cons + LastR  # - 0.01*Energy_Cons + LastR
        return R, Data_Trans

    def build_model(self):
        """basic model.
        """

        states = Input(shape=(3,))
        x = Dense(48, activation='relu', name='hidden_0')(states)
        x = Dense(48, activation='relu', name='hidden_1')(x)
        qValue = Dense(len(self.SNR)*len(self.mod)*len(self.rate), activation='linear', name='output_layer')(x)
        DDQNModel = Model(inputs=states, outputs=qValue, name='DDQN')
        Adam = optimizers.adam_v2.Adam(learning_rate=self.learning_rate)
        DDQNModel.compile(optimizer=Adam, loss='mse')
        return DDQNModel

    def update_target_model(self):
        """update target_model
        """
        self.target_model.set_weights(self.model.get_weights())

    def update_epsilon(self):
        """update epsilon
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def egreedy_action(self, state):
        """ε-greedy
        Arguments:
            state: observation
        Returns:
            action: [snr_index, mod_index, rate_index]
        """
        if np.random.rand() <= self.epsilon:
            return np.array([random.randint(0, len(self.SNR) - 1),
                             random.randint(0, len(self.mod) - 1),
                             random.randint(0, len(self.rate) - 1)])
        else:
            action = np.argmax(self.model.predict_on_batch(state))
            return np.array([np.trunc(action/(len(self.mod)*len(self.rate))),
                             np.trunc((action % (len(self.mod)*len(self.rate)))/len(self.mod)),
                             (action % (len(self.mod)*len(self.rate))) % len(self.rate)], dtype='int16')

    def remember(self, state, action, reward, next_state, done):
        """add data to experience replay.
        Arguments:
            state: observation
            action: action
            reward: reward
            next_state: next_observation
            done: if episode done.
        """
        item = (state, action[0]*len(self.mod)*len(self.rate)+action[1]*len(self.rate)+action[2],
                reward, next_state, done)
        self.memory_buffer.append(item)

    def process_batch(self, batch):
        """process batch data
        Arguments:
            batch: batch size
        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
        # random choice batch data from experience replay.
        # data: [s,a,r,s_,done]
        data = random.sample(self.memory_buffer, batch)
        # Q_target。
        states = np.array([d[0] for d in data]).reshape(-1, 3)
        next_states = np.array([d[3] for d in data]).reshape(-1, 3)

        y = self.model.predict_on_batch(states)
        q = self.target_model.predict_on_batch(next_states)     # Evaluate actions using the target network
        next_action = np.argmax(self.model.predict_on_batch(next_states), axis=1)
        for i, (_, action, reward, _, done) in enumerate(data):
            if not done:
                reward += self.gamma * q[i][next_action[i]]
            y[i][action] = reward  # 第i行第action列
        return states, y

    def train(self, train_episodes, batch):
        TransmitData = []
        EnergyData = []
        episodes = 160
        count = 0
        self.load()
        for repeat in range(train_episodes):
            for episode in range(episodes):
                self.update_epsilon()
                Ts = 0
                DataTrans = 0  # 已传输的数据
                RemEnergy = 0  # 已消耗的能量
                # 第一次不传输数据，只用来获取信道状态，接下来15次传输数据
                # 非对称信道，每个回合有1+15=16次传输，所以一个回合使用了16*2=32条信道
                # 数据集为160*32=5120条信道
                pilot = self.env.reset(episode)
                state, done = self.reset(pilot, Ts, DataTrans)
                # start episode
                while not done:
                    self.env.stateTrans()
                    # choice action from ε-greedy.
                    action = self.egreedy_action(state)  # index of SNR, Mod, Rate
                    EnergyCons = self.emission_energy(action[0])
                    RemEnergy += EnergyCons
                    Ts += 1
                    realBER = self.env.sendData(action)

                    self.env.stateTrans()
                    pilot = self.env.getFeedbackPilot()
                    # 接收到反馈数据(奖励、成功传输的数据)
                    reward, DataTrans = self.feedback(realBER, action, EnergyCons, DataTrans, Ts)
                    nextState, done = self.reset(pilot, Ts, DataTrans)
                    # add data to experience replay.
                    self.remember(state, action, reward, nextState, done)
                    state = nextState
                    if len(self.memory_buffer) > batch:
                        X, y = self.process_batch(batch)
                        self.model.train_on_batch(X, y)
                        count += 1
                        # reduce epsilon pure batch.
                        # update target_model every 20 episode
                        if count != 0 and count % 20 == 0:
                            self.update_target_model()  # 每训练20次model更新一次target_model
                # end episode
                print('Episode: {} | Energy: {:.2f} | Data: {:.2f}'.format
                      ((episode + episodes * repeat) + 1, RemEnergy, DataTrans))
                TransmitData.append(DataTrans)
                EnergyData.append(RemEnergy)
            # end channel episode
        self.save(TransmitData, EnergyData)


if __name__ == '__main__':
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    target_data = 20  # 一个episode的目标吞吐量
    time_limit = 15  # 每15次传输为一个episode
    QoS = 0.001
    snr = [60, 62, 64, 66, 68, 70, 72, 74]
    modulation = [1, 2, 3]
    code_rate = [1/3, 1/2, 2/3]
    fbSNR = 50
    model = DoubleDQN(target_data, time_limit, QoS, snr, modulation, code_rate, fbSNR)
    model.train(30, 64)  # 将整个数据集训练30次
