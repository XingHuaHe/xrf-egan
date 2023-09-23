clc; clear; close all;

load('soil_chz.mat');
energy = spectrums(:,1:2048);
contents = spectrums(:,2049:end);

energy_GAN = csvread('D:\OneDriveHXH\OneDrive\hxh\xrf baseline calibration based on XRFEGAN\Energy Analysis\XRF Enhance GAN - Local Loss\experiences\2.真实数据分析实验\exp-4\fine-tuning-cv\cv_genh.csv', 1, 1);

% 1:Cu 2:Zn 3:Pb 4:As 5:Mn 6:Cr 7:Cd 8:V 9:Mo
element_type = 6;
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
%     window_low = 154;
%     window_high = 160;
    window_low = 158;
    window_high = 159;
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

% ======================================================================
figure(1);
hold on;
% 原信号，算相关系数
contents = zeros(size(spectrums, 1),2);
for i = 1:size(spectrums, 1)
    contents(i,1) = energy_area(i);
    contents(i,2) = data(i, element_type);
end
R = corrcoef(contents);
R_2 = (R(1,2)^2);
fprintf("原信号： R2=%f\n", R_2);
% save('./Result save/orgin.mat', 'contents');
scatter(contents(:, 2), contents(:, 1), 25, 'filled');

x = [ones(length(contents(:, 2)), 1), contents(:, 2)];
[b, bint, r, rint, stats] = regress(contents(:, 1), x);
y = b(1) + b(2)*contents(:, 2);
plot(contents(:, 2), y);

% GAN，算相关系数
contents = zeros(size(spectrums, 1),2);
for i = 1:size(spectrums, 1)
    contents(i,1) = energy_GAN_area(i);
    contents(i,2) = data(i, element_type);
end
R = corrcoef(contents);
R_2 = (R(1,2)^2);
fprintf("XRF-EGAN： R2=%f\n", R_2);
% save('./Result save/orgin.mat', 'contents');
scatter(contents(:, 2), contents(:, 1), 25, 'filled');
x = [ones(length(contents(:, 2)), 1), contents(:, 2)];
[b, bint, r, rint, stats] = regress(contents(:, 1), x);
y = b(1) + b(2)*contents(:, 2);
plot(contents(:, 2), y);

legend('original spectrum', 'original r2', 'XRF-EGAN spectrum', 'XRF-EGAN r2');
box on;
hold off;