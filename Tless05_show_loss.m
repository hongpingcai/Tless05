%% added by Hongping Cai, 05/05/2017
% Tless05_show_loss.m

addpath('../Tless02');
Tless02_init;

%% Tless05, fix5, lr001
figure(3);clf;
subplot(4,1,1);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_solver_lr001_w5.log');
plot_learning_curve(txt_log);axis([0 15000 0 1]); 
title('lr:0.001,w:0.5,fix 5L');
subplot(4,1,2);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_solver_lr001_w05.log');
plot_learning_curve(txt_log);axis([0 15000 0 1]); 
title('lr:0.001,w:0.05,fix 5L');
subplot(4,1,3);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_solver_lr001_w005.log');
plot_learning_curve(txt_log);axis([0 15000 0 1]); 
title('lr:0.001,w:0.005,fix 5L');
subplot(4,1,4);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_solver_lr001_w0005.log');
plot_learning_curve(txt_log);axis([0 15000 0 1]); 
title('lr:0.001,w:0.0005,fix 5L');
fig_name = fullfile(dir_DATA,'Hongping/Tless05/caffenet-log/fix5_lr001.png');
if ~exist(fig_name,'file')
    export_fig(fig_name);
end;

%% Tless05, fix5, lr01
figure(4);clf;
subplot(4,1,1);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_solver_lr01_w5.log');
plot_learning_curve(txt_log);axis([0 8000 0 1]); 
title('lr:0.01,w:0.5,fix 5L');
subplot(4,1,2);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_solver_lr01_w05.log');
plot_learning_curve(txt_log);axis([0 8000 0 1]); 
title('lr:0.01,w:0.05,fix 5L');
subplot(4,1,3);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_solver_lr01_w005.log');
plot_learning_curve(txt_log);axis([0 8000 0 1]); 
title('lr:0.01,w:0.005,fix 5L');
subplot(4,1,4);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_solver_lr01_w0005.log');
plot_learning_curve(txt_log);axis([0 8000 0 1]); 
title('lr:0.01,w:0.0005,fix 5L');

fig_name = fullfile(dir_DATA,'Hongping/Tless05/caffenet-log/fix5_lr01.png');
if ~exist(fig_name,'file')
    export_fig(fig_name);
end;

%% Tless05, fix2, lr001
figure(5);clf;
subplot(4,1,1);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix2_solver_lr001_w5.log');
plot_learning_curve(txt_log);axis([0 10000 0 1]); 
title('lr:0.001,w:0.5,fix 2L');
subplot(4,1,2);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix2_solver_lr001_w05.log');
plot_learning_curve(txt_log);axis([0 10000 0 1]); 
title('lr:0.001,w:0.05,fix 2L');
subplot(4,1,3);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix2_solver_lr001_w005.log');
plot_learning_curve(txt_log);axis([0 10000 0 1]); 
title('lr:0.001,w:0.005,fix 2L');
subplot(4,1,4);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix2_solver_lr001_w0005.log');
plot_learning_curve(txt_log);axis([0 10000 0 1]); 
title('lr:0.001,w:0.0005,fix 2L');
fig_name = fullfile(dir_DATA,'Hongping/Tless05/caffenet-log/fix2_lr001.png');
if ~exist(fig_name,'file')
    export_fig(fig_name);
end;


%% Tless05, fix0, lr001
figure(6);clf;
subplot(4,1,1);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix0_solver_lr001_w5.log');
plot_learning_curve(txt_log);axis([0 10000 0 1]); 
title('lr:0.001,w:0.5,fix 0L');
subplot(4,1,2);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix0_solver_lr001_w05.log');
plot_learning_curve(txt_log);axis([0 10000 0 1]); 
title('lr:0.001,w:0.05,fix 0L');
subplot(4,1,3);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix0_solver_lr001_w005.log');
plot_learning_curve(txt_log);axis([0 10000 0 1]); 
title('lr:0.001,w:0.005,fix 0L');
subplot(4,1,4);
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix0_solver_lr001_w0005.log');
plot_learning_curve(txt_log);axis([0 10000 0 1]); 
title('lr:0.001,w:0.0005,fix 0L');
fig_name = fullfile(dir_DATA,'Hongping/Tless05/caffenet-log/fix0_lr001.png');
if ~exist(fig_name,'file')
    export_fig(fig_name);
end;
