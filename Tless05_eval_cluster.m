%% added by Hongping Cai, 09/05/2017
% Tless05_eval_cluster.m
% Do evaluation on the clustering performance
% 
%OUTPUT:
% 

%%
addpath('../Tless02');
Tless02_init;
cluster_method = 'kmeans';
cluster_K = 15;
feat_type = 'caffenet';
feat_blob = 'fc6';

%%%%%%%%%%%%%%%%%%%%%%%%%
% change parameters below
fix_layer = 0; 
lr = 0.001;
weight_decay = 0.5;
ite = 2000;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%
aa = num2str(lr);bb = num2str(weight_decay);
str_para = ['lr' aa(3:end) '_w' bb(3:end)];
be_show = 0;

%% NOTE: Since Tless03, the test images have been shuffled, so should load the label.txt
% mat_ids_shuffle = fullfile(dir_Tless03,'test_shuffle','ids_shuffle.mat');
% if ~exist(mat_ids_shuffle,'file')
%     error(sprintf('%s does not exist. Pls run Tless03_shuffle_test.m first.',mat_ids_shuffle));
% end;
% load(mat_ids_shuffle,'ids_shuffle','labels_shuffle','im_files_shuffle');
txt_label = fullfile(dir_Tless03,'test_shuffle/label.txt');
labels = dlmread(txt_label);

dir_test_shuffle = fullfile(dir_Tless05,['test_shuffle_ite' int2str(ite)]);

%% load the cluster mat
mat_test_feat = fullfile(dir_test_shuffle, ['fix' int2str(fix_layer) '_test_feat_' str_para '.mat']);
if ~exist(mat_test_feat,'file')
    error('mat_test_feat does not exist. Run Tless05_test.m first.');
else
    fprintf(1,'** Load the test features frm %s....\n',mat_test_feat);
    load(mat_test_feat,'test_feats');
end;

%% load the cluster mat
mat_cluster = fullfile(dir_test_shuffle,['fix' int2str(fix_layer) '_' cluster_method ...
    '_' int2str(cluster_K) '_' str_para '.mat']);
if ~exist(mat_cluster,'file')
    error('mat_cluster does not exist. Run Tless05_test.m first.');
else
    fprintf(1,'** Load clustering file: %s ....\n',mat_cluster);
    load(mat_cluster,'ids_cluster');
end;


fprintf(1,'Clustering performance(%s,fix_layer=%d):\n',str_para,fix_layer);
mat_result = fullfile(dir_test_shuffle,['results_fix' int2str(fix_layer) '_' cluster_method ...
    '_' int2str(cluster_K) '_' str_para '.mat']);
if ~exist(mat_result,'file')
    theta_group_purity = 0.8;
    [ACC] = eval_cluster1(ids_cluster, labels');%
    [nmi_score] = nmi(ids_cluster,double(labels'));
    [rec,pre,tp,acc_fm,tp_fm] = eval_cluster2(ids_cluster, labels, theta_group_purity);
    loss_McClain = McClainIndexLoss(test_feats, labels);    
    save(mat_result,'ACC','nmi_score','rec','pre','tp','acc_fm','tp_fm','loss_McClain');
else
    load(mat_result,'ACC','nmi_score','rec','pre','tp','acc_fm','tp_fm','loss_McClain');
end;
fprintf(1,'** ACC: %.4f\n',ACC);
fprintf(1,'** NMI: %.4f\n',nmi_score);
fprintf(1,'** Obj-wise: rec: %.4f, pre: %.4f, tp:%d\n',rec,pre,tp);
fprintf(1,'** Frm-wise: acc_fm: %.4f, tp_fm: %d\n', acc_fm,tp_fm);
fprintf(1,'** McClainIndex Loss: %.4f\n',loss_McClain);

