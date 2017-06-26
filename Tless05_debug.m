addpath('../Tless02');
Tless02_init;
dir_DATA = '/media/deepthought/DATA';
txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/fix5_d512_solver_lr001_w0005.log');
%txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/tmpdebug_fix5_solver_lr001_w0005.log');
%txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/classification_fix5_lr001_w0005.log');
%txt_log = fullfile(dir_DATA, 'Hongping/Tless05/caffenet-log/classification_flickr_style.log');
figure(3);clf;
[its_tr, loss_tr, its_te, loss_te] = plot_learning_curve(txt_log);
axis([0 10000 0 1]); 
title('lr:0.001,w:0.5,fix 5L');


fid = fopen(txt_log,'r');
tline = fgetl(fid);
count = 0;
all_loss_train = [];
all_loss_test = [];
while ischar(tline)
    id0 = findstr(tline,'N: 64');
    if ~isempty(id0)
        tline = fgetl(fid);
        id1 = findstr(tline,'loss: ');
        id2 = findstr(tline,']');
        if ~isempty(id1) &length(tline)<22
            cur_loss = str2num(tline(id1(1)+9:id2(1)-1));
            all_loss_test = [all_loss_test cur_loss];
        end;
    end;
    
    id0 = findstr(tline,'N: 128');
    if ~isempty(id0)        
        tline = fgetl(fid);
        id1 = findstr(tline,'loss: ');
        id2 = findstr(tline,']');
        if ~isempty(id1) &length(tline)<22
            cur_loss = str2num(tline(id1(1)+9:id2(1)-1));
            all_loss_train = [all_loss_train cur_loss];
        end;
    end;
    
    tline = fgetl(fid);
        
        
%     id1 = findstr(tline,'loss: ');
%     id2 = findstr(tline,']');
%     if ~isempty(id1) &length(tline)<22
%         cur_loss = str2num(tline(id1(1)+9:id2(1)-1));
%         all_loss = [all_loss cur_loss];
%     end;
%     tline = fgetl(fid);
end;
fclose(fid);
% figure(2);clf;
% plot(all_loss);
figure(4);clf;
plot(all_loss_train,'r');
hold on;
plot(all_loss_test,'b');
