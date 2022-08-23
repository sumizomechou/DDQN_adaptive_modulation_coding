function train_data = fadePilot(SNR, channel, pilot)
    SNR = double(SNR);
    channel = double(channel);
    pilot = double(pilot);
%     load(['data\train_channels\tv_',num2str(channelIndex), '.mat'],'Qchannel')
%     rng('default')
%     rng(SNR)
    chan_order=length(channel)/2;
    chan = channel(1:chan_order) + 1i*channel(chan_order+1:chan_order*2);
    
    Len_x = length(pilot)/2;
    Pilot = pilot(1:Len_x) + 1i*pilot(Len_x+1:Len_x*2);
    %% 噪声能量=0dB作为基准功率
    
    Pilot = Pilot.*(10^(SNR/20));
    fadepilot = filter(chan,1, Pilot);
    noisePower_dB = 0;       % 噪声能量
    noisePower = 10^(noisePower_dB/10); % 转换为线性
    noise=sqrt(noisePower)*sqrt(1/2)*(randn(size(Pilot))+1i*randn(size(Pilot)));
    
    
    %% 噪声能量 =  -SNRdB
%     fadepilot = filter(chan,1, Pilot); % Effect of channel, quasi-static
%     pilotPower = sum(abs(Pilot(:)).^2)/length(Pilot(:)); % 信号能量
%     xtxPower_dB = 10*log10(pilotPower); % 转换为dB
%     noisePower_dB = xtxPower_dB-SNR;       % 噪声能量
%     noisePower = 10^(noisePower_dB/10); % 转换为线性
%     noise=sqrt(noisePower)*sqrt(1/2)*(randn(size(fadepilot))+1i*randn(size(fadepilot)));

    %% 添加噪声
    rnoise  =  fadepilot+ noise;
    train_data=zeros(1,length(rnoise)*2);
    train_data(1,1:length(rnoise))=real(rnoise);
    train_data(1,length(rnoise)+1:length(rnoise)*2)=imag(rnoise);
end
