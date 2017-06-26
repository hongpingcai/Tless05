%% Added by Hongping Cai,23/05/2017
% Tless05_BN_loss_with_without_train.m
% Compute the features, clusters before and after training.
% compare the clustering performance and the McClainIndex loss and
% intra-inter distance ratio.
% Using the Batch normalization caffemodel
%
%INPUT:
% Tless01/train_val/train_with_elev.txt
% Tless01/train_val/val_with_elev.txt
% Tless03/test_shuffle/test_with_elev.txt
%
%OUTPUT:
% Tless05-BN/trainset/*.mat
% Tless05-BN/valset/*.mat
% Tless05-BN/testset/*.mat
%

%%
addpath('../Tless02');
Tless02_init;
cluster_method = 'kmeans';
cluster_K = 15;
feat_type = 'caffenet';
train_dataset = 'imagenet';%'tless05';%'tless01';%%%%%%%%%
caffe_input_w = 227; 
caffe_input_h = 227;
be_show = 0;

dir_Tless05 = fullfile(dir_DATA,'Hongping/Tless05_BN');%%%%%%%%%%%%%%%

which_set = 'val';%'train';%'test';%%'val';%%%%%%%%%%%
dir_output = fullfile(dir_Tless05,[which_set 'set']);
if ~exist(dir_output,'dir');
    mkdir(dir_output);
end;


%% load the image and label, elevation
if strcmp(which_set,'test')==1
    txt_elev = fullfile(dir_Tless03,'test_shuffle',[which_set '_with_elev.txt']);    
    txt_list = fullfile(dir_Tless03,'test_shuffle',[which_set '.txt']);
    objs = objs_test;
else
    txt_elev = fullfile(dir_Tless01,'train_val',[which_set '_with_elev.txt']);    
    txt_list = fullfile(dir_Tless01,'train_val',[which_set '.txt']);
    objs = objs_train;
end;
if ~exist(txt_elev,'file')
    [im_files, labels] = textread(txt_list,'%s %d');
    all_im_id = []; all_obj_id = [];
    all_mode = [];  all_elev = [];
    for c=1:length(objs)
        cur_obj_id = objs(c);
        cur_gt_mat = fullfile(dir_Tless01,'gt_mats',sprintf('%02d.mat',cur_obj_id));
        load(cur_gt_mat,'tless_gt');
        
        all_im_id = [all_im_id;cat(1,tless_gt.im_id)];
        all_obj_id = [all_obj_id;cat(1,tless_gt.obj_id)];
        all_mode = [all_mode;cat(1,tless_gt.mode)];
        all_elev = [all_elev;cat(1,tless_gt.elev)];
    end;
    elevs = zeros(size(labels));
    for i=1:length(im_files)
        tmp_ = findstr(im_files{i},'/');
        tmp2 = findstr(im_files{i},'.jpg');
        obj_id = str2num(im_files{i}(tmp_(end-1)+1:tmp_(end)-1));
        im_id = str2num(im_files{i}(tmp_(end)+1:tmp2(1)-1));
        which_gt = find(all_im_id==im_id & all_obj_id==obj_id & all_mode==0);
        if length(which_gt)~=1
            error('which_gt should be only 1 number.');
        end;
        elevs(i) = all_elev(which_gt);
    end;
    fid = fopen(txt_elev,'w');
    for i=1:length(im_files)
        fprintf(fid,'%s %d %d\n',im_files{i},labels(i),elevs(i));
    end;
    fclose(fid);
else
    [im_files, labels, elevs] = textread(txt_elev,'%s %d %d');
end;
n_im = length(labels);
    
       
%% STEP1: generate the training features
switch lower(train_dataset)
    case 'tless01'
%         opt_split_trainval = 1;
%         net_prototxt = fullfile(dir_Tless01,'caffenet-prototxt','deploy.prototxt');
%         net_caffemodel = fullfile(dir_Tless01,'caffenet-model',...
%             ['m' int2str(opt_split_trainval) '_Tless-caffenet_iter_10000.caffemodel']);
%         feat_blob = 'fc7';        
%         str_para = [feat_type '_' feat_blob '_' train_dataset];
%         dim = 4096;
    case 'imagenet'
        net_prototxt = fullfile(dir_DATA,'Hongping/model-caffenet-BN/deploy.prototxt');
        net_caffemodel = fullfile(dir_DATA,'Hongping/model-caffenet-BN/alexnet_cvgj_iter_320000.caffemodel');
        feat_blob = 'fc7/bn';        
        str_para = [feat_type '_' train_dataset];%'_' feat_blob 
        dim = 4096;
    case 'tless05'
        %%%%%%%%%%%%%%%%%%%%%%%%%
        % change parameters below
        fix_layer = 5;
        lr = 0.001;
        weight_decay = 0.0005;
        ite = 500;%100;%2000;%8000;
        dim = 128; %%%%%%
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        feat_blob = 'fc7_tless';%'fc6';
        aa = num2str(lr);bb = num2str(weight_decay);
        str_para_short = ['lr' aa(3:end) '_w' bb(3:end)];
        % str_para = ['fix' int2str(fix_layer) '_lr' aa(3:end) '_w' bb(3:end) ...
        %     '_ite' int2str(ite)];
        str_para = ['fix' int2str(fix_layer) '_d' int2str(dim) '_lr' aa(3:end) '_w' bb(3:end) ...
            '_ite' int2str(ite) '_dropout'];  %%%%%%%%
        net_prototxt = fullfile(dir_Tless05,'caffenet-prototxt',['deploy_d' int2str(dim) '.prototxt']);
        net_caffemodel = fullfile(dir_Tless05,'caffenet-model',...
            ['fix' int2str(fix_layer) '_d' int2str(dim) '_caffenet_' str_para_short '_iter_' int2str(ite) '.caffemodel']);
            % ['fix' int2str(fix_layer) '-caffenet_' str_para_short '_iter_' int2str(ite) '.caffemodel']);
            
    otherwise
        error('No such train_dataset.');
end;
mat_feat = fullfile(dir_output, ['feats_' str_para '.mat']);
if ~exist(mat_feat,'file')
    %% the feat has been generated in Tless05_test.m
%     if strcmp(which_set,'test')==1 & strcmp(train_dataset,'tless05')
%         old_mat_feat = fullfile(dir_Tless05,['test_shuffle_ite' int2str(ite)],...
%             ['fix' int2str(fix_layer) '_test_feat_' str_para_short '.mat']);
%         if ~exist(old_mat_feat)
%             error('old_mat_feat does not exist. Run Tless05_test.m first.');
%         else
%             load(old_mat_feat,'test_feats');
%             feats = test_feats; 
%             fprintf(1,'Compute the distance matrix...\n');
%             dis = pdist2(feats,feats);
%             save(old_mat_feat,'test_feats','dis','-v7.3');
%             clear test_feats;
%             fprintf(1,'\n  Save features and dis into %s\n',mat_feat);
%             save(mat_feat,'feats','dis','-v7.3'); %% also add the distance matrix
%         end;
%     else        
        fprintf(1,'** STEP1: Generate the %s features (%s)....', which_set,str_para);
        mat_mean = '/media/deepthought/DATA/Hongping/Tless02/ilsvrc_2012_mean_227.mat';
        caffe.set_mode_gpu();
        gpu_id = 0;  % we will use the first gpu in this demo
        caffe.set_device(gpu_id);
        net = caffe.Net(net_prototxt, net_caffemodel,'test');
        feats = zeros(n_im,dim);%4096);
        for i=1:n_im
            if mod(i,50)==1
                fprintf(1,'%d ',i);
            end;
            im = imread(im_files{i});
            %input_data = {prepare_image(im,caffe_input_w,caffe_input_h,mat_mean)};
            im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
            im_data = permute(im_data, [2, 1, 3]);  % flip width and height
            im_data = single(im_data);  % convert from uint8 to single
            im_data = imresize(im_data, [caffe_input_w caffe_input_h], 'bilinear');  % resize im_data
            
            scores = net.forward({im_data});
            cur_feat = net.blobs(feat_blob).get_data();
            feats(i,:) = cur_feat';%%%%%%%%
        end;
        caffe.reset_all();
        fprintf(1,'Compute the distance matrix...\n');
        dis = pdist2(feats,feats);
        fprintf(1,'\n  Save features and dis into %s\n',mat_feat);
        save(mat_feat,'feats','dis'); %% also add the distance matrix
   % end;
else
    fprintf(1,'** Load the features frm %s....\n',mat_feat);
    load(mat_feat,'feats','dis');
end;

%% STEP2: clustering
fprintf(1, 'STEP2: clustering on %s set (%s)....', which_set,str_para);
mat_cluster = fullfile(dir_output,[cluster_method '_' int2str(cluster_K) ...
    '_' str_para '.mat']);
if ~exist(mat_cluster,'file')    
    %% the feat has been generated in Tless05_test.m
%     if strcmp(which_set,'test')==1 & strcmp(train_dataset,'tless05')
%         old_mat_cluster = fullfile(dir_Tless05,['test_shuffle_ite' int2str(ite)],...
%             ['fix' int2str(fix_layer) '_kmeans_' int2str(cluster_K) '_' str_para_short '.mat']);
%         if ~exist(old_mat_cluster)
%             error('old_mat_feat does not exist. Run Tless05_test.m first.');
%         else
%             copyfile(old_mat_cluster,mat_cluster);
%             load(mat_cluster);
%         end;
%     else        
        rng(1); % For reproducibility
        [ids_cluster,centres_cluster,sumd,D] = kmeans(feats,cluster_K,'MaxIter',1000,...
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
 %   end;
else
    fprintf(1,'** Load clustering file: %s ....\n',mat_cluster);
    load(mat_cluster);
end;

%% STEP3: clustering performance and loss
disp(['STEP3: clustering performance and loss on ' which_set ' set(' str_para ')']);
mat_result = fullfile(dir_output,['results_' cluster_method ...
    '_' int2str(cluster_K) '_' str_para '.mat']);
if ~exist(mat_result,'file')
    theta_group_purity = 0.8;
    [ACC] = eval_cluster1(ids_cluster, labels');%
    [nmi_score] = nmi(ids_cluster,double(labels'));
    [rec,pre,tp,acc_fm,tp_fm] = eval_cluster2(ids_cluster, labels, theta_group_purity);
    loss_McClain = McClainIndexLossFrmDis(dis, labels);
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

if be_show
    figure(5);clf;
    ma = {'b','r','y','m','g','b:','r:','y:','m:','k:','g:','b-.','r-.','y-.','m-.','k-.'};
    for i=1:size(r,1)
        plot(vec_diff_ele,r(i,:),ma{i},'LineWidth',1.5);
        hold on;
    end;
    hold on;
    plot(vec_diff_ele,vec_R,'k','LineWidth',3);
    hold on;
    if strcmp(which_set,'test')==1
        legend('Obj02','Obj04','Obj06','Obj08','Obj10','Obj12','Obj14',...
            'Obj16','Obj18','Obj20','Obj22','Obj24','Obj26','Obj28','Obj30','Avg');
    else
        legend('Obj01','Obj03','Obj05','Obj07','Obj09','Obj11','Obj13',...
            'Obj15','Obj17','Obj19','Obj21','Obj23','Obj25','Obj27','Obj29','Avg');
    end
    line([50 50],[0 1.4]);
    axis([0 80 0 1.4]);
    str_para_ = strrep(str_para,'_','-')
    title(['intra-inter ratio, ' which_set ' set , ' str_para_]);
end;


