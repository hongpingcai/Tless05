%% added by Hongping Cai, 04/05/2017
% Tless05_test.m
% Using the caffenet+McClainIndexLoss to do clustering on Tless-test set 
%
%INPUT:
% 
%OUTPUT:
% Tless05/test_shuffle/fix5_kmeans_15_lr01_w0005.mat
% Tless05/test_shuffle/centres.txt
% Tless05/test_shuffle/fix5_test_feats_lr01_w0005.mat

%%

addpath('../Tless02');
Tless02_init;

cluster_method = 'kmeans';
cluster_K = 15;
feat_type = 'caffenet';
feat_blob = 'fc6';
feat_dim  = 4096;
caffe_input_w = 227; 
caffe_input_h = 227;
%%%%%%%%%%%%%%%%%%%%%%%%%
% change parameters below
fix_layer = 5; 
lr = 0.001;
weight_decay = 0.0005;
ite = 8000;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%
aa = num2str(lr);bb = num2str(weight_decay);
str_para = ['lr' aa(3:end) '_w' bb(3:end)];
be_show = 0;

dir_test_shuffle = fullfile(dir_Tless05,['test_shuffle_ite' int2str(ite)]);%%%%%%%%%%%%%%%%%
if ~exist(dir_test_shuffle,'dir')
    mkdir(dir_test_shuffle)
end;

mat_ids_shuffle = fullfile(dir_Tless03,'test_shuffle','ids_shuffle.mat');
if ~exist(mat_ids_shuffle,'file')
    error(sprintf('%s does not exist. Pls run Tless03_shuffle_test.m first.',mat_ids_shuffle));
end;
load(mat_ids_shuffle,'ids_shuffle','labels_shuffle','im_files_shuffle');


%% SPTE1: generate the test features
mat_test_feat = fullfile(dir_test_shuffle, ['fix' int2str(fix_layer) '_test_feat_' str_para '.mat']);
fprintf(1,'SPTE1: generate the test features...\n');
if ~exist(mat_test_feat,'file')
    net_prototxt = fullfile(dir_Tless05,'caffenet-prototxt','deploy.prototxt');
    net_caffemodel = fullfile(dir_Tless05,'caffenet-model',...
            ['fix' int2str(fix_layer) '-caffenet_' str_para '_iter_' int2str(ite) ...
            '.caffemodel']);
    mat_mean = fullfile(dir_Tless02,'ilsvrc_2012_mean_227.mat');
    disp('** Generate the testing features....');
    caffe.set_mode_gpu();
    gpu_id = 0;  % we will use the first gpu in this demo
    caffe.set_device(gpu_id);
    net = caffe.Net(net_prototxt, net_caffemodel,'test');
    
    mat_test_patch = fullfile(dir_Tless01,'mode0','test_patch.mat');
    load(mat_test_patch,'patches');
    %mat_test_label = fullfile(dir_Tless01,'mode0','test_label.mat');
    patches = patches(ids_shuffle,:,:,:);
    
    
    n_im_test = size(patches,1);
    test_feats = zeros(n_im_test,4096);
    for i=1:n_im_test
        if mod(i,50)==1
            fprintf(1,'%d ',i);
        end;
        im = squeeze(patches(i,:,:,:));
        input_data = {prepare_image(im,caffe_input_w,caffe_input_h,mat_mean)};
        scores = net.forward(input_data);
        cur_feat = net.blobs(feat_blob).get_data();
        test_feats(i,:) = cur_feat';%%%%%%%%
    end;
    fprintf(1,'\n  Save features into %s\n',mat_test_feat);
    caffe.reset_all();
    fprintf(1,'Compute the distance matrix...\n');
    dis = pdist2(test_feats,test_feats);
    save(mat_test_feat,'test_feats','dis');
else
    fprintf(1,'** Load the test features frm %s....\n',mat_test_feat);
    load(mat_test_feat,'test_feats','dis');
end;

%% STEP3: clustering
fprintf(1, 'STEP3: clustering...\n');
mat_cluster = fullfile(dir_test_shuffle,['fix' int2str(fix_layer) '_' cluster_method ...
    '_' int2str(cluster_K) '_' str_para '.mat']);
if ~exist(mat_cluster,'file')
    rng(1); % For reproducibility
    [ids_cluster,centres_cluster,sumd,D] = kmeans(test_feats,cluster_K,'MaxIter',1000,...
        'start','cluster','Display','final','Replicates',10);
    % find the cluster centre images
    ids_centre = zeros(1, cluster_K);
    for i=1:cluster_K
        ids_cur = find(ids_cluster == i);
        dis_cur = D(ids_cur,i);
        [v,d] = min(dis_cur);
        ids_centre(i) = ids_cur(d);
    end;
    save(mat_cluster,'ids_cluster','D','centres_cluster','ids_centre');
else
    fprintf(1,'** Load clustering file: %s ....\n',mat_cluster);
    load(mat_cluster,'ids_cluster');
end;

%% STEP4: clustering performance
fprintf(1,'STEP4: clustering performance(%s,fix_layer=%d)...\n',str_para,fix_layer);
mat_result = fullfile(dir_test_shuffle,['results_fix' int2str(fix_layer) '_' cluster_method ...
    '_' int2str(cluster_K) '_' str_para '.mat']);
if ~exist(mat_result,'file')
    theta_group_purity = 0.8;
    labels = labels_shuffle';
    [ACC] = eval_cluster1(ids_cluster, labels');%
    [nmi_score] = nmi(ids_cluster,double(labels'));
    [rec,pre,tp,acc_fm,tp_fm] = eval_cluster2(ids_cluster, labels, theta_group_purity);
    loss_McClain = McClainIndexLossFrmDis(dis, labels);
    txt_elev = fullfile(dir_Tless03,'test_shuffle',['test_with_elev.txt']);
    if ~exist(txt_elev,'file')
        error('txt_elev file does not exist. Run Tless05_loss_with_without_train.m first.');
    else
        [im_files, labels, elevs] = textread(txt_elev,'%s %d %d');
    end;
    [vec_diff_ele, vec_R,r] = eval_viewpoint_invariance(dis,labels,elevs);
    save(mat_result,'ACC','nmi_score','rec','pre','tp','acc_fm','tp_fm','loss_McClain',...
        'vec_diff_ele', 'vec_R','r');
else
    load(mat_result,'ACC','nmi_score','rec','pre','tp','acc_fm','tp_fm','loss_McClain',...
        'vec_diff_ele', 'vec_R','r');
end;
fprintf(1,'** ACC: %.4f\n',ACC);
fprintf(1,'** NMI: %.4f\n',nmi_score);
fprintf(1,'** Obj-wise: rec: %.4f, pre: %.4f, tp:%d\n',rec,pre,tp);
fprintf(1,'** Frm-wise: acc_fm: %.4f, tp_fm: %d\n', acc_fm,tp_fm);
fprintf(1,'** McClainIndex Loss: %.4f\n',loss_McClain);
fprintf(1,'** intra-inter dis ratio at elev-diff=50 is : %.2f\n',vec_R(6));




%     
%     new_centre_feats = test_feats(ids_centre,:);
%     new_centre_ids = ids_centre;
%     dis_centres = pdist2(old_centre_feats,new_centre_feats);
%     [assignment,cost] = munkres(dis_centres);    
%     if be_show
%         im_all = zeros(64*2,64*15,3,'uint8');
%         for k=1:cluster_K
%             im_all(1:64,(k-1)*64+1:k*64,:)   = imresize(imread(im_files_shuffle{old_centre_ids(k)}),[64 64]);
%             im_all(65:end,(k-1)*64+1:k*64,:) = imresize(imread(im_files_shuffle{new_centre_ids(assignment(k))}),[64 64]);
%         end;
%         figure(1);clf;
%         imshow(im_all);
%         title(sprintf('Up: centres of ite=%d, Down: centres of ite=%d',ite-1,ite));
%         fig_name = fullfile(dir_Tless04, 'fig',sprintf('lamda%d-fix%d-centres_ite%d-ite%d.png',lamda*10,fix_layer,ite-1,ite));
%         if ~exist(fig_name)
%             export_fig(fig_name);
%         end;
%     end
%         
%     %% STEP5: write the new centres into txt file
%     txt_centres = fullfile(dir_test_shuffle,['fix' int2str(fix_layer) '_centres_ite' int2str(ite) '.txt']);
%     fprintf(1,'STEP5: Write the centres into %s ....\n',txt_centres);
%     if ~exist(txt_centres,'file')
%         fid = fopen(txt_centres,'w');
%         for i=1:cluster_K
%             fprintf(fid,'%s %d\n',im_files_shuffle{ids_centre(i)}, i-1); %% 0,1,....
%             fprintf(1,'%s %d\n',im_files_shuffle{ids_centre(i)}, i-1);
%             if be_show
%                 figure(2);clf;
%                 imshow(im_files_shuffle{ids_centre(i)});
%                 tmp_ = findstr(im_files_shuffle{ids_centre(i)},'/');
%                 title(['C' int2str(i) '_centre image (' ...
%                     im_files_shuffle{ids_centre(i)}(tmp_(end-1)+1:end) ')']);
%                 pause;
%             end;
%         end;
%         fclose(fid);
%     end;
%     if cost<1e4%%%%%%%%
%         fprintf(1,'STOP.');
%         break;
%     end;


