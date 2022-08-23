% 用来产生调制和信道的关系的数据
% 信道使用随机信道
% 调制方式为 BPSK QPSK 8PSK
% 通信体制为单载波频域均衡
% 接收端首先假设信道已知
function BER_target = BER_generate_new(Len_block, nblock, SNR, M, Rate_code, channel, CP, pilot)
    Len_block = double(Len_block);
    nblock = double(nblock);
    M = double(M);
    SNR = double(SNR);
    channel = double(channel);
    CP = double(CP);
    pilot = double(pilot);
%     rng('default')
%     rng(SNR)
    % 前半为实部，后半为虚部
    chan_order=length(channel)/2;
    chan = channel(1:chan_order) + 1i*channel(chan_order+1:chan_order*2);
    fprintf(['channel size: ', num2str(size(chan,1)), ': ', num2str(size(chan,2)), ' \n'])
    Len_x = length(pilot)/2; % 导频长度1280
    Pilot = pilot(1:Len_x) + 1i*pilot(Len_x+1:Len_x*2);
	% fprintf(['Pilot size: ', num2str(size(Pilot,1)), ': ', num2str(size(Pilot,2)), ' \n'])
    N_cp =length(CP);   % cp的长度
    N_data = Len_block-N_cp;
    PN_add=zeros(N_cp,nblock);
    for i=1:nblock
        PN_add(:,i)=CP.'; % CP:256x1,nblock个CP
    end

    %% 产生发射数据
    M_mod=2^M;
    %% 卷积码
    len_ori_bit = N_data*M*Rate_code;     % 原始bit
    len_coded_bit = N_data*M;     % 编码后bit

    data_info_bit = randi([0,1], len_ori_bit, nblock);
    % 交织
    pos=randperm(len_coded_bit);
    inv_pos=zeros(1,len_coded_bit);
    for i=1: len_coded_bit
        inv_pos(pos(i))=i;
    end
    % 编码
    if Rate_code == 1/3
        trellis = poly2trellis(7, [123,135,157]);
    elseif Rate_code == 1/2
        trellis = poly2trellis(3, [7 5]);
    elseif Rate_code == 2/3
        trellis = poly2trellis([5 4], [23 35 0; 0 5 13]);
    end
    x=zeros(N_data,nblock);
    for i_block =1:nblock
        codedword = convenc(data_info_bit(:,i_block),trellis); 
        conv_interleave=codedword(pos);
        x(:,i_block)=bi2de(reshape(conv_interleave,[],M));
    end
    %% 调制
%     x = randi([0,M_mod-1], N_data, nblock);
    xtx_pload =pskmod(x,M_mod) ; % 2560的数据调制成xPSK
    %         xtx_pload=ones(128,100);

    xtx = [xtx_pload;PN_add];  %% 添加PN
    xtx_temp= reshape(xtx,[],1); % 重构为?x1的矩阵
    xtx_all= [Pilot;xtx_temp];  % 加导频

%% 噪声能量=0dB作为基准功率

    xtx_all = xtx_all.*(10^(SNR/20));
    fadesig = filter(chan,1, xtx_all);
    xtxPower = sum(abs(xtx_all(:)).^2)/length(xtx_all(:)); % 信号能量
    noisePower_dB = 0;       % 噪声能量
    noisePower = 10^(noisePower_dB/10); % 转换为线性
    noise=sqrt(noisePower)*sqrt(1/2)*(randn(size(xtx_all))+1i*randn(size(xtx_all)));
    CP=xtx_all(1:N_cp).';

%% 噪声能量=-SNRdB作为基准功率
%     fadesig = filter(chan,1, xtx_all);
%     xtxPower = sum(abs(xtx(:)).^2)/length(xtx(:)); % 信号能量
%     xtxPower_dB = 10*log10(xtxPower); % 转换为dB
%     noisePower_dB = xtxPower_dB-SNR;       % 噪声能量
%     noisePower = 10^(noisePower_dB/10); % 转换为线性
%     noise=sqrt(noisePower)*sqrt(1/2)*(randn(size(fadesig))+1i*randn(size(fadesig)));

    %%
    rnoise  =  fadesig+ noise;

    %% IPNLMS
%     h_CE=CE_IPNLMS(rnoise(Len_x-N_cp+1:Len_x),CP,chan_order); % 信道估计
    %% LS
    training_length = N_cp-chan_order+1;
    [~,h_CE]=channel_es_LS(rnoise(Len_x-N_cp+1:Len_x),CP,chan_order,training_length);
    h_CE = h_CE.';
    %%
    rnoise = rnoise(Len_x+1:end, :); % 移除Pilot
    rnoise_CP = reshape(rnoise, [], nblock);  %%
    %% 解码
    SER_MMSE_temp =zeros(1,nblock);
    for i_block=1:nblock
        H=fft(h_CE, Len_block, 2); % 未知信道
        rnoise_inter=rnoise_CP(:,i_block);  %% 取一个数据块
        fdein = fft(rnoise_inter,Len_block); % 转换为FDE的频域进行处理

        H_mmse = (H')./((abs(H).^2+ones(size(H))/(xtxPower/noisePower)).');% mmse，发射信号与接收器的噪声功率比
        fdeout_mmse = fdein.*H_mmse;
        xrx_mmse = ifft(fdeout_mmse,Len_block); % 转换回时域
        z_mmse = pskdemod(xrx_mmse,M_mod); % psk解码
        x_temp=reshape(de2bi(z_mmse(1:N_data),M),[],1);   % 去掉CP，转换二进制
        p_cc_from_del =x_temp(inv_pos); % 解交织
        detec_data = vitdec(p_cc_from_del,trellis,2,'trunc','hard');   % 解卷积码
        
        SER_MMSE_temp(i_block)=BER_Cacula(detec_data ,data_info_bit(:,i_block));
%         h_CE=CE_IPNLMS(rnoise_inter(end-N_cp+1:end),CP,chan_order); % 信道估计
        [~,h_CE]=channel_es_LS(rnoise_inter(end-N_cp+1:end),CP,chan_order,training_length);
        h_CE = h_CE.';
    end
    BER_target=mean(SER_MMSE_temp);
end
