% 导入模拟谱图的测试结果数据，分析误差

clc; clear all ;close all;

% load("../experiences/sim-t-500/test/simEvaluate.mat");
load("D:\OneDriveHXH\OneDrive\hxh\xrf baseline calibration based on XRFEGAN\Energy Analysis\XRF Enhance GAN - Local Loss\experiences\2.真实数据分析实验\exp-4\fine-tuning-cv/simEvaluate.mat");

error_xrf = all_results_Genh - all_results_Clean;
fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 MSE 误差 : %f\n", abs(mse(error_xrf)));
t = mean(error_xrf.^2, 2);
fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 std 误差 : %f\n", std(t));
t = sum(t) / length(t);
% fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 RMSE 误差 : %f\n", sum(t) / length(t));
fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 RMSE 误差 : %f\n", sqrt(t));
figure(1);
hold on;
plot(1:1:2048, all_results_Noisy(1, :),'color',[0.3 0.08 0.18])
plot(1:1:2048, all_results_Clean(1, :),'color',[0.64 0.08 0.18]); % 18020715265317081412032120200401170206829xa.dat.mat   样本号：603  GBW07384(GSD-33)
plot(1:1:2048, abs(all_results_Genh(1, :)),'color',[0.00 0.00 1.00]);
plot(1:1:2048, abs(error_xrf(1, :)),'color',[1.00 0.00 1.00]);
legend('original spectrum', 'clean spectrum', 'EGAN spectrum', 'mse error');
box on;
hold off;

error_xrf_log = all_results_Genh_log - all_results_Clean_log;
fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 log MSE 误差 : %f\n", abs(mse(error_xrf_log)));
t = mean(error_xrf_log.^2, 2);
fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 log std 误差 : %f\n", std(t));
t = sum(t) / length(t);
% fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 RMSE 误差 : %f\n", sum(t) / length(t));
fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 log RMSE 误差 : %f\n", sqrt(t));
% fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 log RMSE 误差 : %f\n", abs(rms(error_xrf_log)));
% fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 log std 误差 : %f\n", abs(std(error_xrf_log)));
figure(2);
hold on;
plot(1:1:2048, all_results_Noisy_log(1, :),'color',[0.63 0.08 0.18]);
plot(1:1:2048, all_results_Clean_log(1, :),'color',[0.64 0.08 0.18]);
plot(1:1:2048, abs(all_results_Genh_log(1, :)),'color',[0.00 0.00 1.00]);
plot(1:1:2048, abs(error_xrf_log(1, :)),'color',[1.00 0.00 1.00]);
legend('original spectrum', 'clean spectrum', 'EGAN spectrum', 'mse error');
box on;
hold off;
% error_background = noisy_npy - results_Genh;
% fprintf("GAN 去除的基底和真实本底 MSE 误差 : %f\n", mse(error_background));
% figure(2);
% hold on;
% plot(1:1:2048, results_Genh(1, :));
% plot(1:1:2048, noisy_npy(1, :));
% plot(1:1:2048, error_background(1, :));
% legend('GAN', 'Noisy', 'Backgorund Error');
% hold off;

fprintf("============================================\n");
% ========================================================================
% ========================= sample =================================
fprintf("GAN 去基底和清洁数据的 XRF 谱图 MSE 误差 : %f\n", abs(mse(error_xrf(1, :))));
t = mean(error_xrf(1, :).^2, 2);
fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 RMSE 误差 : %f\n", sqrt(t));

fprintf("GAN 去基底和清洁数据的 XRF 谱图 log MSE 误差 : %f\n", abs(mse(error_xrf_log(1, :))));
t = mean(error_xrf_log(1, :).^2, 2);
fprintf("GAN 去基底和清洁数据的 XRF 谱图全局 log RMSE 误差 : %f\n", sqrt(t));






% =======================================================================
load('soil_chz.mat');
energy = spectrums(:,1:2048);
contents = spectrums(:,2049:end);

energy_GAN = csvread('.\exp-4\fine-tuning-cv\cv_genh.csv', 1, 1);

% 1:Cu 2:Zn 3:Pb 4:As 5:Mn 6:Cr 7:Cd 8:V 9:Mo
element_type = 1;
data(:, 1) = contents(:, 29);
data(:, 2) = contents(:, 30);
data(:, 3) = contents(:, 82);
data(:, 4) = contents(:, 33);
data(:, 5) = contents(:, 25);
data(:, 6) = contents(:, 24);
data(:, 7) = contents(:, 48);
data(:, 8) = contents(:, 23);
data(:, 9) = contents(:, 42);

if element_type == 1
    % Cu
    window_low = 229;                 %卡窗左界
    window_high = 235;                %卡窗右界
elseif element_type == 2
    % Zn
%     window_low = 242;
%     window_high = 254;
    window_low = 245;
    window_high = 253;
elseif element_type == 3
    % Pb
%     window_low = 296; % ka
%     window_high = 309;
    window_low = 358; % beta 
    window_high = 373;
elseif element_type == 4
    % As
    window_low = 296;
    window_high = 309;
elseif element_type == 5
    % Mn
    window_low = 165;
    window_high = 176;
elseif element_type == 6
    % Cr
    window_low = 154;
    window_high = 160;
%     window_low = 158;
%     window_high = 159;
elseif element_type == 7
    % Cd
%     window_low = 648;
%     window_high = 690;
    window_low = 650;
    window_high = 666;
elseif element_type == 8
    % V
    window_low = 140;
    window_high = 148;
elseif element_type == 9
    % Mo
    window_low = 494;
    window_high = 510;
end


% 原谱图.
energy_expand = zeros(size(spectrums, 1), (window_high - window_low + 1));
% GAN
energy_GAN_expand = zeros(size(spectrums, 1), (window_high - window_low + 1));
for i = 1:size(spectrums, 1)
    for j = 1:(window_high - window_low + 1)
        % 原谱图.
        energy_expand(i,j) = energy(i, window_low+j-1 );
        % GAN 
        energy_GAN_expand(i,j) = energy_GAN(i, window_low+j-1 );
    end
end


ag_window_low = 572;
ag_window_high = 624;
% origin 康普顿.
energy_ag_expand = zeros(size(spectrums, 1), (ag_window_high - ag_window_low + 1));
for i = 1:size(spectrums, 1)
    for j = 1:(ag_window_high - ag_window_low + 1)
        energy_ag_expand(i,j) = energy(i, ag_window_low+j-1);
    end
end
% gan 康普顿.
energy_gan_ag_expand = zeros(size(spectrums, 1), (ag_window_high - ag_window_low + 1));
for i = 1:size(spectrums, 1)
    for j = 1:(ag_window_high - ag_window_low + 1)
        energy_gan_ag_expand(i,j) = energy_GAN(i, ag_window_low+j-1);
    end
end


% 原谱图.
energy_area = zeros(1,size(spectrums, 1));
% GAN
energy_GAN_area = zeros(1, size(spectrums, 1));
% 康普顿峰面积.
energy_Ag_area = zeros(1,size(spectrums, 1));
energy_gan_Ag_area = zeros(1,size(spectrums, 1));
for i = 1:size(spectrums, 1)
    % 康普顿.
    energy_Ag_area(i) = sum(energy_ag_expand(i,:));
    energy_gan_Ag_area(i) = sum(energy_gan_ag_expand(i,:));
    % 原谱图.
    energy_area(i) = sum(energy_expand(i,:)) / energy_Ag_area(i);
    % GAN 
    energy_GAN_area(i) = sum(energy_GAN_expand(i,:)) / energy_gan_Ag_area(i);
end

fprintf("原始谱图康普顿峰净峰面积：%f\n", energy_Ag_area(1));
fprintf("EGAN 校准后谱图康普顿峰净峰面积：%f\n", energy_gan_Ag_area(1));
fprintf("原始谱图特征净峰面积：%f\n", sum(energy_expand(1,:)));
fprintf("EGAN 校准后谱图特征峰净峰面积：%f\n", sum(energy_GAN_expand(1,:)));
fprintf("原始：%f    EGAN：%f\n",  energy_area(1), energy_GAN_area(1));

% GAN，算相关系数
contents = zeros(size(spectrums, 1),2);
for i = 1:size(spectrums, 1)
    contents(i,1) = energy_GAN_area(i);
    contents(i,2) = data(i, element_type);
end
R = corrcoef(contents);
R_2 = (R(1,2)^2);
fprintf("XRF-EGAN： R2=%f\n", R_2);