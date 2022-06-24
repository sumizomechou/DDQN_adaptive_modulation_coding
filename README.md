# DDQN_adaptive_modulation_coding  
underwater communication adaptive modulation and coding based on double deep q network  
在Python中调用Matlab方法参考 http://blog.csdn.net/sunny_xsc1994/article/details/79254196  
  
非对称信道下的水声通信自适应调制  
提出了一种自适应调制和编码 (AMC) 方案，旨在通过联合调度编码速率、调制阶数和传输功率来最大化单个链路的能量效率。  
考虑到UWA信道的复杂性，提出了一种基于深度神经网络（DNN）的误码率（BER）估计方法，可以通过使用反馈链路的固定导频来实现信道估计、特征提取和BER估计。  
设计了一种基于所有调制和编码方式（MCS）的BER的信道分类方法，并将UWA信道进一步建模为具有未知转移概率的有限状态马尔可夫链（FSMC），  
以将 AMC 问题表述为马尔可夫决策过程 (MDP)，并使用了DDQN方法实现了接近最优的能量效率。
