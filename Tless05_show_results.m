%% Added by Hongping Cai, 19/06/2017
% Tless05_show_results.m
%
%

%%
addpath('../Tless02');
Tless02_init;
cluster_method = 'kmeans';
cluster_K = 15;
%feat_type = 'caffenet';
%train_dataset = 'tless05';%'
fix_layer = 5;
lr = 0.001;
weight_decay = 0.0005;
aa = num2str(lr);bb = num2str(weight_decay);

%% -------- train_d128
which_set = 'train';%'test';%%'val';%%%%%%%%%%%
dim = 128; %%%%%%
dir_output = fullfile(dir_Tless05,[which_set 'set']);
all_ites = [100 200 300 500 2000];
all_ACC_train_d128 = zeros(1,length(all_ites));
all_nmi_train_d128 = zeros(1,length(all_ites));
all_los_train_d128 = zeros(1,length(all_ites));
for i=1:length(all_ites)
    ite = all_ites(i);%500;%100;%2000;%8000;
    str_para = ['fix' int2str(fix_layer) '_d' int2str(dim) '_lr' aa(3:end) '_w' bb(3:end) ...
        '_ite' int2str(ite)];
    mat_result = fullfile(dir_output,['results_' cluster_method ...
        '_' int2str(cluster_K) '_' str_para '.mat']);
    load(mat_result,'ACC','nmi_score','rec','pre','tp','acc_fm','tp_fm','loss_McClain',...
        'vec_diff_ele', 'vec_R','r');
    all_ACC_train_d128(i) = ACC;
    all_nmi_train_d128(i) = nmi_score;
    all_los_train_d128(i) = loss_McClain;
    clear ACC;clear loss_McClain;clear nmi_score;
end;

%% -------- test_d128
which_set = 'test';%'test';%%'val';%%%%%%%%%%%
dim = 128; %%%%%%
dir_output = fullfile(dir_Tless05,[which_set 'set']);
all_ites = [100 200 300 500 2000];
all_ACC_test_d128 = zeros(1,length(all_ites));
all_nmi_test_d128 = zeros(1,length(all_ites));
all_los_test_d128 = zeros(1,length(all_ites));
for i=1:length(all_ites)
    ite = all_ites(i);%500;%100;%2000;%8000;
    str_para = ['fix' int2str(fix_layer) '_d' int2str(dim) '_lr' aa(3:end) '_w' bb(3:end) ...
        '_ite' int2str(ite)];
    mat_result = fullfile(dir_output,['results_' cluster_method ...
        '_' int2str(cluster_K) '_' str_para '.mat']);
    load(mat_result,'ACC','nmi_score','rec','pre','tp','acc_fm','tp_fm','loss_McClain',...
        'vec_diff_ele', 'vec_R','r');
    all_ACC_test_d128(i) = ACC;
    all_nmi_test_d128(i) = nmi_score;
    all_los_test_d128(i) = loss_McClain;
    clear ACC;clear loss_McClain;clear nmi_score;
end;
    

%% -------- train_d512
which_set = 'train';%'test';%%'val';%%%%%%%%%%%
dim = 512; %%%%%%
dir_output = fullfile(dir_Tless05,[which_set 'set']);
all_ites = [100 200 300 500 2000];
all_ACC_train_d512 = zeros(1,length(all_ites));
all_nmi_train_d512 = zeros(1,length(all_ites));
all_los_train_d512 = zeros(1,length(all_ites));
for i=1:length(all_ites)
    ite = all_ites(i);%500;%100;%2000;%8000;
    str_para = ['fix' int2str(fix_layer) '_d' int2str(dim) '_lr' aa(3:end) '_w' bb(3:end) ...
        '_ite' int2str(ite)];
    mat_result = fullfile(dir_output,['results_' cluster_method ...
        '_' int2str(cluster_K) '_' str_para '.mat']);
    load(mat_result,'ACC','nmi_score','rec','pre','tp','acc_fm','tp_fm','loss_McClain',...
        'vec_diff_ele', 'vec_R','r');
    all_ACC_train_d512(i) = ACC;
    all_nmi_train_d512(i) = nmi_score;
    all_los_train_d512(i) = loss_McClain;
    clear ACC;clear  loss_McClain;clear nmi_score;
end;

%% -------- test_d512
which_set = 'test';%'test';%%'val';%%%%%%%%%%%
dim = 512; %%%%%%
dir_output = fullfile(dir_Tless05,[which_set 'set']);
all_ites = [100 200 300 500 2000];
all_ACC_test_d512 = zeros(1,length(all_ites));
all_nmi_test_d512 = zeros(1,length(all_ites));
all_los_test_d512 = zeros(1,length(all_ites));
for i=1:length(all_ites)
    ite = all_ites(i);%500;%100;%2000;%8000;
    str_para = ['fix' int2str(fix_layer) '_d' int2str(dim) '_lr' aa(3:end) '_w' bb(3:end) ...
        '_ite' int2str(ite)];
    mat_result = fullfile(dir_output,['results_' cluster_method ...
        '_' int2str(cluster_K) '_' str_para '.mat']);
    load(mat_result,'ACC','nmi_score','rec','pre','tp','acc_fm','tp_fm','loss_McClain',...
        'vec_diff_ele', 'vec_R','r');
    all_ACC_test_d512(i) = ACC;
    all_nmi_test_d512(i) = nmi_score;
    all_los_test_d512(i) = loss_McClain;
    clear ACC;clear  loss_McClain;
end;

figure(1);clf;
plot(all_ites,all_ACC_train_d128,'r-*','Linewidth',2);
hold on;
plot(all_ites,all_ACC_test_d128,'b-*','Linewidth',2);
hold on;
plot(all_ites,all_ACC_train_d512,'r-.o','Linewidth',2);
hold on;
plot(all_ites,all_ACC_test_d512,'b-.o','Linewidth',2);
hold on;
plot(all_ites, 0.473.*ones(1,length(all_ites)),'b:','Linewidth',1);
hold on;
plot(all_ites, 0.511.*ones(1,length(all_ites)),'r:','Linewidth',1);
legend('train-d128','test-d128','train-d512','test-d512','train-withouttrain','test-withouttrain');
title('Clustering accuracy, Fix5L');
xlabel('iteration');
ylabel('ACC');


figure(2);clf;
plot(all_ites,all_los_train_d128,'r-*','Linewidth',2);
hold on;
plot(all_ites,all_los_test_d128,'b-*','Linewidth',2);
hold on;
plot(all_ites,all_los_train_d512,'r-.o','Linewidth',2);
hold on;
plot(all_ites,all_los_test_d512,'b-.o','Linewidth',2);
hold on;
plot(all_ites, 0.642.*ones(1,length(all_ites)),'b:','Linewidth',1);
hold on;
plot(all_ites, 0.647.*ones(1,length(all_ites)),'r:','Linewidth',1);
legend('train-d128','test-d128','train-d512','test-d512','train-withouttrain','test-withouttrain');
title('McClainIndexLoss, Fix5L');
xlabel('iteration');
ylabel('Loss');

figure(3);clf;
plot(all_ites,all_nmi_train_d128,'r-*','Linewidth',2);
hold on;
plot(all_ites,all_nmi_test_d128,'b-*','Linewidth',2);
hold on;
plot(all_ites,all_nmi_train_d512,'r-.o','Linewidth',2);
hold on;
plot(all_ites,all_nmi_test_d512,'b-.o','Linewidth',2);
hold on;
plot(all_ites, 0.614.*ones(1,length(all_ites)),'b:','Linewidth',1);
hold on;
plot(all_ites, 0.630.*ones(1,length(all_ites)),'r:','Linewidth',1);
legend('train-d128','test-d128','train-d512','test-d512','train-withouttrain','test-withouttrain');
title('Clustering performance, Fix5L');
xlabel('iteration');
ylabel('NMI');
    