clc;
% set the root_dir to the directory where the Code and visualization data is located 
root_dir = 'fig45';
% Choose a subject {'NY742', 'NY717', 'NY749'} -- we don't have subject brains other than these but MNI should work for all subjs
Subj = 'NY717';
% get paths to all visualization files
paths = get_path(root_dir,Subj);
% load electrode (channel) table
channel_info_all = get_channel_info(paths);
% get the relevant plotting options by choosing {'mni' or 'subj'} and hemisphere {'lh' or 'rh'} 
plot_data_all_mni = get_plot_data(paths, 'mni', 'lh');
% plot_data_all_T1 = get_plot_data(paths, 'subj', 'lh');
attrs={'spec'};
%% 
elec_range = 1:128;
elec = table2array(channel_info_all(elec_range,plot_data_all_mni.coord_ind));
elecname = table2array(channel_info_all(elec_range,11));
% set radius of the elec circle
radius = 1*ones(length(elec_range));
% choose a color map and set indecies of colors for each electrode
color_map = hot(256);%afmhot;%hot;
color_map = color_map(end:-1:1,:);
% [color_map]=cbrewer('seq', 'Blues', 256);
[color_map_div]=cbrewer('div', 'RdBu', 256,'PCHIP');
[colorset1] = cbrewer('qual', 'Set1', 9,'PCHIP');
color_map_div = color_map_div(end:-1:1,:);
color_ind = value2colorind(ones(128,1),'hot',[0,1]); 
color_ind = [color_ind,color_ind,color_ind];
colors = {color_ind, color_map};
ylims_min = [0,0,0,0,0,0,0];
ylims_max = [1,1,1,1,1,1.5,1];

wreg=true;
temp=true;
tau=true;
return_curve=true;

global win_len 
win_len = 1;%11;
entire_period = true;
post = true;
visdiff = false;
postabs = true;%post || ~visdiff; % sparse if true
max_att = false;%post || ~visdiff;
max_anker = true;
clrmap = color_map;

subjs = {'717','742','749','829','798'};
SUB = length(subjs);
plot_on_same_brain = true;
above_median=false;
[found,ind]=ismember('freq_formants_hamon',attrs);

if found
    row=size(attrs,2)+2;
else
    row=size(attrs,2)+1;
end
N_tau = 16;

if temp
    col = 13;
else
    if plot_on_same_brain
        col=1;
    else
        col=SUB;
    end
end

invest_regions = {{'cSTG','rSTG','mMTG','cMTG','rMTG',...
    'inferiorprecentral','superiorprecentral','postcentral','supramarginal',...
    'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'}};

all_regions = {'cSTG','rSTG','mMTG','cMTG','rMTG',...
    'inferiorprecentral','superiorprecentral','postcentral','supramarginal',...
    'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'};

rows=1;
invest_regions = {{'mMTG','rMTG',...
                    'inferiorprecentral','superiorprecentral','postcentral','supramarginal',...
                    'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'},...
                    {'cSTG','rSTG',...
                    'inferiorprecentral','superiorprecentral','postcentral',...
                    'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'},...
                    {'cSTG','mMTG',...
                    'inferiorprecentral','superiorprecentral','postcentral','supramarginal',...
                    'parsopercularis','parstriangularis','caudalmiddlefrontal'}};


reference = {'causalpre','causal','anticausal'};
dataname = {'causal','causal','anticausal'};
is_causal={true,true,false};
%       
for ir=1:3
    
    fig1=figure();
    set(fig1,'position',[0,0,2560,270])
    TT=7;
    haxis2 = tight_subplot(rows,TT,[0,0],0,0);
    dumm = 1;
    onregion = {'mSTG','rSTG','cSTG','rMTG','mMTG','cMTG','inferiorprecentral','superiorprecentral','postcentral','caudalmiddlefrontal','rostralmiddlefrontal','parstriangularis','parsopercularis','supramarginal'};  

    win=hann(win_len); win = win./sum(win);
    [frames,frames_brain,err,tticks] = attr_vis_tau('spec',post,visdiff,postabs,max_att,plot_on_same_brain,above_median,temp,wreg,entire_period,true,invest_regions{ir},reference{ir},dataname{ir},is_causal{ir});

    for t=1:length(frames_brain)
        axcp = copyobj(frames_brain{t}.CurrentAxes,fig1);
        set(axcp,'Position',get(haxis2(sub2ind([TT,rows],t,dumm)),'position'));
%                 set(axcp,'Position',get(haxis2(sub2ind([TT,rows],TT-t+1,dumm)),'position'));
        close(frames_brain{t})
        set(haxis2(sub2ind([TT,rows],t,dumm)),'visible','off');
        pos = get(haxis2(sub2ind([TT,rows],t,dumm)),'position');
        currentax=haxis2(sub2ind([TT,rows],t,dumm));
    end

end
set(gca,'visible','off')


% ---------------------------------------------------------------------------
% ----------------------------Relevant Functions ----------------------------
% ---------------------------------------------------------------------------
function y = myconv(a,b,reflectdir)
    filter_size=floor(size(b,1)/2);
    switch reflectdir
        case 'left'
%             a = [a(filter_size:-1:1,:);a;zeros(size(a(end:-1:end-filter_size+1,:)))];
            a = [a(filter_size:-1:1,:);a;a(end:-1:end-filter_size+1,:)];
        case 'right'
%             a = [zeros(size(a(filter_size:-1:1,:)));a;a(end:-1:end-filter_size+1,:)];
            a = [a(filter_size:-1:1,:);a;a(end:-1:end-filter_size+1,:)];
    end
    y = conv2(a,b,'valid');
end

function [frames,frames_brain,err,tticks] = PlotAtt(Att1_cell, mask_cell,coord_cell,regions,causal,brain,colormap,Att_c,Att_cpre,Att_a,ref,investreg)
        name4plot = {'cSTG','rSTG','mMTG','cMTG','rMTG','vPreCG','dPreCG','PostCG','supramarginal','pOp','pTri','rMFG','cMFG'};
        regions4barplot =investreg;
        dosmooth = true;
    %     regions4barplot = {'inferiorprecentral','superiorprecentral','postcentral','supramarginal'};
    %     regions4barplot = {'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'};
        att=[];
        att_ref = [];
        att_nt = [];
        mask=[];
        region = [];
        coord = [];
        ratio = 1;%0.47;
        switch ref
            case 'causal'
                Att_ref = Att_c;
            case 'causalpre'
                Att_ref = Att_cpre;
    %             Att_ref = Att_c;
            case 'anticausal'
                Att_ref = Att_a;
        end
        
        for sub = 1:length(Att1_cell)
            att = cat(3,att,Att1_cell{sub});
            att_ref = cat(3,att_ref,Att_ref{sub});
            mask = cat(1,mask,mask_cell{sub});
            region = cat(1,region,regions{sub});
            coord = cat(2,coord,coord_cell{sub});
        end
        col=144;%13;
        offset=16;
        rate = 8;
        N_tau=16;%16*rate;
        if causal
            if not(strcmp(ref,'causalpre'))
                tau0=5;
    %             tau1=11;
                tau1=8;
            else
                tau0=7;
    %             tau0=5;
                tau1=11;
            end
            t1=-6 + offset/rate;
            t2=0 + offset/rate;
        else
            tau0=5;
            tau1=7;
            t1=-1 + offset/rate;
            t2=5 + offset/rate;
        end
        switch ref
            case 'causal'
                tticks = [-55:0]/125*1000;
            case 'causalpre'
                tticks = [-55:0]/125*1000;
            case 'anticausal'
                tticks = [0:55]/125*1000;
        end
    % % %     frames = zeros(tau2-tau1+1,length(regions4barplot));
        frames = zeros((t2-t1+1)*rate,length(regions4barplot));
        err = zeros((t2-t1+1)*rate,length(regions4barplot));
        
        att = reshape(att,[col,N_tau,size(att,3),size(att,4)]);
        region_value=cell(length(regions4barplot),1);
        region_samples=cell(length(regions4barplot),1);

        for r=1:length(region_samples)
            region_samples{r} = 0;
        end
        
        att_norm = 0;
        for tau=tau0:tau1
            att_norm = att_norm + att(rate*(tau-1)+1+t1*rate:rate*tau+t2*rate,tau,:,:);
        end
        att_norm = att_norm/(tau1-tau0+1);
        att_norm = mean(att_norm,1);
        att = att./(att_norm+1e-10).*att_ref*ratio;
        
    
        global win_len
        win=hann(win_len); win = win./sum(win);

        if tticks(1)==0
            reflectdir = 'right';
        else
            reflectdir = 'left';
        end
        
        for m=1:size(mask,1)
            for n=1:size(mask,2)
                reg = reshape(region(m,n,:),1,[]);
                [found,ind]=ismember(reg(find(reg~=0)),regions4barplot);
                [containmstg,~] = ismember('mSTG',regions4barplot);
                if containmstg
                    [ismstg,~]=ismember(reg(find(reg~=0)),{'mSTG'});
                    if ismstg
                        if coord(2,m,n)>=(0.553486529*coord(3,m,n)-2.527049117)
                            [~,ind]=ismember('rSTG',regions4barplot);
                        else
                            [~,ind]=ismember('cSTG',regions4barplot);
                        end
                    end
                end
                if found && mask(m,n)
    %                     region_count{ind} = region_count(ind)+1;
                    for tau=tau0:tau1
    %                 avg_data = avg_data+data_region(t,t+tau1:t+tau2);
                        switch ref
                            case 'causalpre'
                                region_value{ind} = [region_value{ind},[att(rate*(tau-1)+1+t1*rate:rate*tau0+t2*rate,tau,m,n);zeros(rate*tau+t2*rate-(rate*tau0+t2*rate),1,1,1)]];
                                region_samples{ind} = region_samples{ind}+[ones(rate*tau0+t2*rate-(rate*(tau-1)+1+t1*rate)+1,1,1,1);zeros(rate*tau+t2*rate-(rate*tau0+t2*rate),1,1,1)];
                            otherwise
                                region_value{ind} = [region_value{ind},att(rate*(tau-1)+1+t1*rate:rate*tau+t2*rate,tau,m,n)];
                        end
                    end
                end
            end
        end
        
        if tticks(1)==0
            reflectdir = 'right';
        else
            reflectdir = 'left';
        end
        for r = 1:length(regions4barplot)
            if dosmooth
                region_value{r} = myconv(region_value{r},reshape(win,size(win,1),1),reflectdir);
            end
        end
        for r = 1:length(regions4barplot)
            switch  ref
                case 'causalpre'
                    frames(:,r) = sum(region_value{r},2)./(region_samples{r}+1e-10);
                    err(:,r) = sqrt(sum((region_value{r}-frames(:,r)).^2,2)./(sum(region_samples{r},2)+1e-10))./(sqrt(sum(region_samples{r},2)*50)+1e-10)*1.96; % each data point reflect the average of 50 trials of test speech
                otherwise
                    frames(:,r) = mean(region_value{r},2);
                    err(:,r) = std(region_value{r},1,2)/sqrt(size(region_value{r},2)*50)*1.96; % each data point reflect the average of 50 trials of test speech
            end
        end
        
        
        frames_brain = zeros(t2*rate-t1*rate+rate,tau1-tau0+1,size(att,3),size(att,4));
        mask_reshape = reshape(mask,1,1,size(att,3),size(att,4));
        masked_att = mask_reshape.*att;
        summ=0;
        for tau=tau0:tau1
            switch ref
                case 'causalpre'
                    frames_brain(:,tau-tau0+1,:,:) = [masked_att(rate*(tau-1)+1+t1*rate:rate*tau0+t2*rate,tau,:,:);zeros(rate*tau+t2*rate-(rate*tau0+t2*rate),1,size(masked_att,3),size(masked_att,4))];
                    summ = summ+[ones(rate*tau0+t2*rate-(rate*(tau-1)+1+t1*rate)+1,1,size(masked_att,3),size(masked_att,4));zeros(rate*tau+t2*rate-(rate*tau0+t2*rate),1,size(masked_att,3),size(masked_att,4))];
                otherwise
                    frames_brain(:,tau-tau0+1,:,:) = masked_att(rate*(tau-1)+1+t1*rate:rate*tau+t2*rate,tau,:,:);
                    summ = summ+ones(rate*tau+t2*rate-(rate*(tau-1)+t1*rate),1,size(masked_att,3),size(masked_att,4));
            end
    %         frames_brain(:,tau-tau0+1,:,:) = masked_att(rate*(tau-1)+1+t1*rate:rate*tau+t2*rate,tau,:,:);
        end
        
        frames_brain = permute(frames_brain,[2,1,3,4]);
        frames_brain = reshape(frames_brain,size(frames_brain,1)*rate,size(frames_brain,2)/rate,size(frames_brain,3),size(frames_brain,4));
    %     frames_brain = mean(frames_brain,1)/1.5;
        summ = permute(summ,[2,1,3,4]);
        summ = reshape(summ,size(summ,1)*rate,size(summ,2)/rate,size(summ,3),size(summ,4));
        frames_brain = sum(frames_brain,1)./sum(summ,1)/1.5;

        [frames_brain] = VisualAtt({frames_brain}, {mask}, {region}, {coord}, brain,colormap,1,true,false,false,false,true,[0,0.7],4);

    end

function [frames,prob,prob0] = VisualAtt(Att, Mask, Region, coord, brain,color_map,maxv,alpha,div,annotation,bar_plot,gaussianplot,scale,gausskernel,causal)
    if ~exist('annotation','var')
        annotation = [];
    end
    if ~exist('gausskernel','var')
        gausskernel = 10;
    end
    if ~exist('gaussianplot','var')
        gaussianplot = true;
    end
    if ~exist('bar_plot','var')
        bar_plot = false;
    end
    if ~exist('causal','var')
        causal = true;
    end
    
    if nargin<5
        color_map = 'hot';
        color_map = eval([color_map,'(256)']);
    end
    annot = brain.annot;
    regions4barplot = {'rSTG','cSTG','rMTG','mMTG','cMTG','inferiorprecentral','superiorprecentral','postcentral','caudalmiddlefrontal','rostralmiddlefrontal','parstriangularis','parsopercularis','supramarginal'};
    att=[];
    mask=[];
    cod = [];
    region = [];
    for ii = 1:length(Att)
        att = cat(3,att,Att{ii});
        mask = cat(1,mask,Mask{ii});
        cod = cat(2,cod,coord{ii});
        region = cat(1,region,Region{ii});
    end
    load('annot_regions.mat');
    if bar_plot
        annot_regions = regions4barplot;
    end
%     mask = Mask{ii};
    if length(size(att))==4
        sizes = size(att);
        Att_avg = reshape(mean(att,1),sizes(2:end));
    else 
        Att_avg = att;
    end
    if nargin<6
        maxv = max(Att_avg(:));
    end

    Att_avg = (Att_avg)/maxv;

    time_step = size(Att_avg,1);
    if gaussianplot
        frames={};
    else
        frames = [];
    end
    for i=1:time_step
        Att_frame = squeeze(Att_avg(i,:,:));
        region_value = cell(length(annot_regions),1);
        ii_value = cell(length(annot_regions),1);

        for m=1:size(mask,1)
            for n=1:size(mask,2)
                reg = reshape(region(m,n,:),1,[]);
                [found,ind]=ismember(reg(find(reg~=0)),annot_regions);
                [ismstg,~]=ismember(reg(find(reg~=0)),{'mSTG'});
                if ismstg
                    if cod(2,m,n)>=0.553486529*cod(3,m,n)-2.527049117
                        [~,ind]=ismember('rSTG',annot_regions);
                    else
                        [~,ind]=ismember('cSTG',annot_regions);
                    end
                end
                if found && mask(m,n)
%                     region_count{ind} = region_count(ind)+1;
                    region_value{ind} = [region_value{ind},Att_frame(m,n)];
                    ii_value{ind} = [ii_value{ind},ceil(m/15)];
                end
            end
        end
        for m=1:length(region_value)
            region_value_mean(m) = mean(region_value{m});
            region_value_stderr(m) = std(region_value{m})/sqrt(length(region_value{m}));
        end
       
        region_value = region_value_mean;
        ind_active = find(mask==1);

        if div
            color_ind = value2colorind(Att_frame(ind_active), 'hot',[0,1]);
            color_ind_region = value2colorind(region_value, 'hot',[0,1]);
        else
            color_ind = value2colorind(Att_frame(ind_active), 'hot',[0,1]);
            color_ind_region = value2colorind(region_value, 'hot',[0,1]);
        end
        if annotation
            regioncolor = color_map(color_ind_region,:);
            regioncolor(isnan(region_value),:) = repmat([0.92 0.92 0.92],length(find(isnan(region_value))),1);
        end
%             colors = {color_ind, color_map};
        if size(color_map,1)<100
            if size(color_map,1)==1
                colors = {color_map};
            else
                colors = {color_map(ii,:)};
            end
        else
            colors = {color_ind, color_map};
        end
        elecname = 1:length(ind_active);
        elec = zeros(length(ind_active),3);
        Temp = squeeze(cod(1,:,:));
        elec(:,1) = Temp(ind_active);
        Temp = squeeze(cod(2,:,:));
        elec(:,2) = Temp(ind_active);
        Temp = squeeze(cod(3,:,:));
        elec(:,3) = Temp(ind_active);

        if alpha
            if div
                point_alpha = 2*abs(max(min(Att_frame(ind_active),1),0)-0.5);
%                 point_alpha = max(min(abs(Att_frame(ind_active)),1),0);
            else
                point_alpha = max(min(Att_frame(ind_active),1),0);
            end
            point_alpha = double(point_alpha>0.1);
        else
            point_alpha = double(abs(Att_frame(ind_active)-0.5)>0.07);
        end
        if annotation
            braincolors = regioncolor;
            point_alpha = point_alpha*0;
        else
            braincolors = [];

            brain.annot = [];
        end
        regionreshape = reshape(region,size(region,1)*size(region,2),size(region,3));
        validregion = regionreshape(ind_active,:);
        if gaussianplot
            load('S.mat')
            Lcrtx = load('ch2_template_lh_pial.mat');
            Lcrtx.tri = Lcrtx.faces;Lcrtx.vert = Lcrtx.coords;
            if div
                data = (Att_frame(ind_active)-0.5)*2;
            else
                data = Att_frame(ind_active);
            end
            [fig,prob_all]=ctmr_gauss_plot_edited(Lcrtx,elec,ones(size(elec,1),1),S.cax,0,S.cm,S.gsp*gausskernel,[],annot,validregion); 
            color_map(1:int64(size(color_map,1)*0.21/S.cax(2)),:)=1;
            prob0 = prob_all;
            close(fig)
            if div
                S.cax=[-0.7,0.7];
            else
                S.cax = [0,0.7];
            end
            if exist('scale','var')
                S.cax = scale;
            end
            [fig,prob]=ctmr_gauss_plot_edited(Lcrtx,elec,data,S.cax,0,color_map,S.gsp*gausskernel,prob_all,annot,validregion); 
            viewangle = 'l';   
            pltshowplanes = 0;
            cameratoolbar('setmode',''); 
            litebrain(viewangle,.8); 
            hold on; colormap(gca,color_map);
            axis off;
            frames{i}=fig;
        else
            fig = nyu_plot_whitebackground(brain.surf_brain,...
                                           brain.sph,...
                                           elec, elecname,...
                                           colors, 0, brain.annot, 2, 1, 1, point_alpha,braincolors);
            frame = getframe(fig);
            frames(i,:,:,:) = frame.cdata;
            prob = frames;
            prob0 = frames;
        end
    end
end

function [rgb_inds] = value2colorind(value, c_map, c_range)
    if ~exist('c_map','var') || isempty(c_map)
        c_map = 'hot';
    end
    if ~exist('c_range','var') || isempty(c_range)
        c_range = [mean(value(:)), max(value(:))];
    end
    cmap = eval([c_map,'(256)']);
    % clip the out of range values
    value(value<c_range(1)) = c_range(1);
    value(value>c_range(2)) = c_range(2);
    % normalize the values
    rgb_inds = round(((value-c_range(1))/(c_range(2)-c_range(1)))*255)+1;
    rgb_inds(isnan(rgb_inds)) = 1;
end
function [plot_data_all] = get_plot_data(paths, visual_mode, sph)
    if strcmp(lower(visual_mode), 'mni')
        if strcmp(lower(sph), 'lh')
            plot_data_all.surf_brain = load(paths.MNI_lh);
            plot_data_all.sph = 'lh';
            plot_data_all.annot = paths.MNI_lh_annot;
            plot_data_all.coord_ind = 2:4;
        elseif strcmp(lower(sph), 'rh')
            plot_data_all.surf_brain = load(paths.MNI_rh);
            plot_data_all.sph = 'rh';
            plot_data_all.annot = paths.MNI_rh_annot;
            plot_data_all.coord_ind = 2:4;
        else
            error('sph mode not supported! use lh or rh');
        end
    elseif strcmp(lower(visual_mode), 'subj')
         if strcmp(lower(sph), 'lh')
            plot_data_all.surf_brain = load(paths.Subj_lh);
            plot_data_all.sph = 'lh';
            plot_data_all.annot = paths.Subj_lh_annot;
            plot_data_all.coord_ind = 6:8;
        elseif strcmp(lower(sph), 'rh')
            plot_data_all.surf_brain = load(paths.Subj_rh);
            plot_data_all.sph = 'rh';
            plot_data_all.annot = paths.Subj_rh_annot;
            plot_data_all.coord_ind = 6:8;
        else
            error('sph mode not supported! use lh or rh');
        end
    else
        error('visual_mode not supported! use mni or subj');
    end
end
function [channel_info_all] = get_channel_info(paths)
    % load coordinate files
    coordinates = readtable(paths.coordinate_file);
    ind = [1:8,10,11];
    coordinates = coordinates(:,ind);
    coordinates.channel= [1:size(coordinates,1)]';
    Subj.sebject = repmat(paths.Subj,size(coordinates,1),1);
    Subj = struct2table(Subj);
%     Subj = table('Size',[size(coordinates,1),1],...
%                  'VariableTypes', {'string'},...
%                  'VariableNames', {'subject'});
%     Subj.subject = paths.Subj;
    channel_info_all = [coordinates, Subj];
end
function [paths] = get_path(root_dir, Subj)
    addpath(genpath(root_dir));
    paths.root_dir = root_dir;
    paths.Subj = Subj;
    paths.Subj_path = [root_dir, filesep, 'Data', filesep, Subj];
    paths.coordinate_file = [paths.Subj_path, filesep, 'coordinates.csv'];
    paths.MNI_lh = [paths.root_dir, filesep, 'Data', filesep, 'MNI', filesep, 'ch2_template_lh_pial_120519.mat'];
    paths.MNI_rh = [paths.root_dir, filesep, 'Data', filesep, 'MNI', filesep, 'ch2_template_rh_pial_120519.mat'];
    paths.MNI_lh_annot = [paths.root_dir, filesep, 'Data', filesep, 'MNI', filesep, 'ch2_template.lh.aparc.split_STG_MTG.annot'];
    paths.MNI_rh_annot = [paths.root_dir, filesep, 'Data', filesep, 'MNI', filesep, 'ch2_template.rh.aparc.split_STG_MTG.annot'];
    lh_pial_file = dir([paths.Subj_path,filesep,'*lh_pial_surf.mat']);
    lh_annot_file = dir([paths.Subj_path,filesep,'lh*_STG_MTG.annot']);
    rh_pial_file = dir([paths.Subj_path,filesep,'*rh_pial_surf.mat']);
    rh_annot_file = dir([paths.Subj_path,filesep,'rh*_STG_MTG.annot']);
    if (length(lh_pial_file)~=0)
        paths.Subj_lh = [paths.Subj_path, filesep, lh_pial_file(1).name];
    else
        paths.Subj_lh = [];
    end
    if (length(rh_pial_file)~=0)
        paths.Subj_rh = [paths.Subj_path, filesep, rh_pial_file(1).name];
    else
        paths.Subj_rh = [];
    end
    if (length(lh_annot_file)~=0)
        paths.Subj_lh_annot = [paths.Subj_path, filesep, lh_annot_file(1).name];
    else
        paths.Subj_lh_annot = [];
    end
    if (length(rh_annot_file)~=0)
        paths.Subj_rh_annot = [paths.Subj_path, filesep, rh_annot_file(1).name];
    else
        paths.Subj_rh_annot = [];
    end
end

function [Att1,Att1_temp] = gather_att(data,postabs,maxv,dsrate,tau)
if nargin<5
    tau=false;
end
if tau
    data_reshape = reshape(data,size(data,1),dsrate,size(data,2)/dsrate,1,size(data,4),size(data,5),size(data,6));
    % data_temp = max(data_reshape,[],3);
    data_temp = repmat(data_reshape,[1,2,1,1,1,1,1]);
else
    data_reshape = reshape(data,size(data,1),dsrate,size(data,2)/dsrate,1,size(data,4),size(data,5),size(data,6));
    % data_temp = max(data_reshape,[],3);
    data_temp = repmat(data_reshape,[1,2,1,1,1,1]);
end
if ~maxv
    if ~postabs
        Att1 = mean(mean(abs(squeeze(data)),2),1); 
        Att1_temp = mean(abs(squeeze(data_temp)),1);
        Att1_temp = mean(Att1_temp,2);
        sizes = size(Att1_temp);
        Att1_temp = reshape(Att1_temp,[sizes(1),sizes(3:end)]);
    else
        Att1 = mean(abs(mean(squeeze(data),2)),1);
%         Att1 = mean(mean(abs(mean(squeeze(data),2)),1),5);
        Att1_temp = abs(mean(squeeze(data_temp),1));
        Att1_temp = mean(Att1_temp,2);
        sizes = size(Att1_temp);
        Att1_temp = reshape(Att1_temp,[sizes(1),sizes(3:end)]);
    end
else
    if ~postabs
        Att1 = max(mean(abs(squeeze(data)),1),[],2);
        Att1_temp = mean(abs(squeeze(data_temp)),1);
        Att1_temp = max(Att1_temp,[],2);
        sizes = size(Att1_temp);
        Att1_temp = reshape(Att1_temp,[sizes(1),sizes(3:end)]);
    else
        Att1 = max(abs(mean(squeeze(data),1)),[],2);
        Att1_temp = abs(mean(squeeze(data_temp),1));
        Att1_temp = max(Att1_temp,[],2);
        sizes = size(Att1_temp);
        Att1_temp = reshape(Att1_temp,[sizes(1),sizes(3:end)]);
    end
end
if tau
    Att1_temp = permute(Att1_temp,[1,2,5,3,4]);
    sz = size(Att1_temp);
    Att1_temp = reshape(Att1_temp,[sz(1),sz(2)*sz(3),sz(4:end)]);
end
end
function [att,att_temp,min25,max75] = robust_rescale(data,data_temp,mask,median_baseline,regions,tau)
    if nargin<4
        median_baseline=true;
    end
    if nargin<5
        regions = false;
    end
    if nargin<6
        tau = false;
    end
    if regions
        att_temp = zeros(size(data_temp));
        att = zeros(size(data));
        region = reshape(regions(repmat(mask,1,1,20)==1),length(find(mask==1)),20);
        unique_region = unique(region,'rows');
        sz = size(regions);
        mask_region = zeros(sz(1:length(sz)-1));
        for regs=1:size(unique_region)
            for i =1:size(regions,1)
                for j=1:size(regions,2)
                    if strcmp(reshape(regions(i,j,:),1,length(unique_region(regs,:))),unique_region(regs,:))
                        mask_region(i,j) = regs;
                    end
                end
            end 
        end
        for regs=1:size(unique_region)
            mask_temp = repmat(reshape(mask_region,[1,1,size(mask_region,1),size(mask_region,2)]),size(data_temp,1),size(data_temp,2),1,1)==regs;
            mask_rep = repmat(reshape(mask_region,[1,1,size(mask_region,1),size(mask_region,2)]),size(data,1),size(data,2),1,1)==regs;
            sorted = sort(data(mask_rep));
            min25 = sorted(int32(0.05*length(sorted))+1);
            max75 = sorted(int32(0.95*length(sorted)));
            md = median(sorted);
            
            sorted_temp = sort(data_temp(mask_temp));
            min25_temp = sorted_temp(int32(0.05*length(sorted_temp))+1);
            max75_temp = sorted_temp(int32(0.95*length(sorted_temp)));
            md_temp = median(sorted_temp);
            if median_baseline
                att(mask_rep) = (data(mask_rep)-md)/(max75-min25);
                att_temp(mask_temp) = (data_temp(mask_temp)-md)/(max75-min25);
%                 att_temp(mask_temp) = (data_temp(mask_temp)-md_temp)/(max75_temp-min25_temp);
            else
                att = (data-min25)/(max75-min25);
                att_temp(mask_temp) = (data_temp(mask_temp)-min25)/(max75-min25);
%                 att_temp(mask_temp) = (data_temp(mask_temp)-min25_temp)/(max75_temp-min25_temp);
            end
        end
    else
%         if tau
%             sorted = sort(data(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2),1]),size(data,1),size(data,2),1,1,size(data,5))==1));
%         else
%             sorted = sort(data(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data,1),size(data,2),1,1)==1));
%         end
        sorted = sort(data(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data,1),size(data,2),1,1)==1 & ~isnan(data)));
%         min25 = sorted(int32(0.05*length(sorted))+1);
%         max75 = sorted(int32(0.95*length(sorted)));
        min25 = sorted(int32(0.01*length(sorted))+1);
        max75 = sorted(int32(0.99*length(sorted)));
        md = median(sorted);
        
%         if tau
%             sorted_temp = sort(data_temp(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2),1]),size(data_temp,1),size(data_temp,2),1,1,size(data_temp,5))==1));
%         else
%             sorted_temp = sort(data_temp(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data_temp,1),size(data_temp,2),1,1)==1));
%         end
        sorted_temp = sort(data_temp(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data_temp,1),size(data_temp,2),1,1)==1 & ~isnan(data_temp)));
%         min25_temp = sorted_temp(int32(0.05*length(sorted_temp))+1);
%         max75_temp = sorted_temp(int32(0.95*length(sorted_temp)));
        min25_temp = sorted_temp(int32(0.01*length(sorted_temp))+1);
        max75_temp = sorted_temp(int32(0.99*length(sorted_temp)));
        md_temp = median(sorted_temp);

        if median_baseline
            att = (data-md)/(max75-min25);
%             att_temp = (data_temp-md)/(max75-min25);
            att_temp = (data_temp-md_temp)/(max75_temp-min25_temp);
            att_temp(isnan(att_temp))=0;
            att_temp(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data_temp,1),size(data_temp,2),1,1)==0) = 0;
        else
            att = (data-min25)/(max75-min25);
%             att_temp = (data_temp-min25)/(max75-min25);
            att_temp = (data_temp-min25_temp)/(max75_temp-min25_temp);
            att_temp(isnan(att_temp))=0;
            att_temp(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data_temp,1),size(data_temp,2),1,1)==0) = 0;
        end
    end
end


function [frames,frames_brain,err,tticks] = attr_vis_tau(attr,post,visdiff,postabs,max_att,plot_on_same_brain,above_median,temp,wreg,entire_period,return_curve,investreg,reference,dataname,is_causal)
    % choose a color map and set indecies of colors for each electrode
    color_map = hot(256);%afmhot;%hot;
    color_map = color_map(end:-1:1,:);

    color_ind = value2colorind(ones(128,1),'hot',[0,1]); 
    color_ind = [color_ind,color_ind,color_ind];
    colors = {color_ind, color_map};
    clrmap = color_map;
    if strcmp(attr,'ecog')
        dsrate_org = 1;
    else
        dsrate_org = 8;
    end
    if entire_period
        on = 16/dsrate_org+1;
        off = 120/dsrate_org;
    else
        if post
            on = (16+32)/dsrate_org+1;
            off = 120/dsrate_org;
        else
            on = 16/dsrate_org+1;
            off = (16+32)/dsrate_org;
            on_post = (16+32)/dsrate_org+1;
            off_post = 120/dsrate_org;
        end
    end
    on_ttau=1;
    off_ttau=144;
    dsrate = 8/dsrate_org;
    SUB = 3;
    % ind = [1:10,21:30];
    ind = [1:2];%[1:50];
    Att1_cell = {};
    att_cell_c = {};
    att_cell_cpre = {};
    att_cell_a = {};
    mask_cell = {};
    coord_cell = {};
    region_cell = {};
    subjs = {'717','742','749','829','798'};
%     subjs = {'717','742','749'};
    root_dir = 'fig45';
    
    for sub=1:SUB
        
        load(['fig45/NY_',subjs{sub},'_elec_entiregrid.mat']);
        region_cell{sub} = regions;

        load(['fig45/entiregrid_',subjs{sub},'_normxdim_',dataname,'_covnorm_NOVISWEIGHT_step1/occ_covnorm_temporal_tstep1_tau_8.mat',]);
        load(['fig45/contrib_ref_',subjs{sub},'.mat']);
        color_map = hot(256);%afmhot;%hot;
        color_map = color_map(end:-1:1,:);
        clrmap = color_map;
        eval([attr,'=',attr,';']);
        causal=is_causal;
        ref = reference;%'anticausal';'causalpre';'causal';'percept';
        do_t_ds=false;
        Subj = ['NY',subjs{sub}];
        % get paths to all visualization files
        paths = get_path(root_dir,Subj);
        channel_info_all = get_channel_info(paths);
        plot_data_all_mni = get_plot_data(paths, 'mni', 'lh');
        coord = mni;

        att = eval(attr);
        if do_t_ds
            att = att(:,1:8:end,:,:,:,:);
        end

        data = att(ind,on_ttau:off_ttau,1,:,:,:);


        att_cell_c{sub} = Att1_c;
        att_cell_cpre{sub} = Att1_cpre;
        att_cell_a{sub} = Att1_a;

% 
        sorted = sort(data(:));
        sorted = sorted(end-1);

        [Att1,Att1_temp] = gather_att(data,postabs,max_att,dsrate,true);

        sorted = sort(Att1(:));
%                 sorted = sorted(end-1);
        % get all plot frames



        if wreg
            [Att1,Att1_temp,m25,m75] = robust_rescale(Att1,Att1_temp,mask,above_median,regions,true);
        else
            [Att1,Att1_temp,m25,m75] = robust_rescale(Att1,Att1_temp,mask,above_median,false,true);
%                                 Att1_temp = (max(Att1_temp,0)).^thepower;
        end
        if temp
            Att1_cell{sub} = Att1_temp;
        else
            Att1_cell{sub} = Att1;
        end
        mask_cell{sub} = mask;
        coord_cell{sub} = coord;
        if plot_on_same_brain 
            if sub == SUB
                if ~return_curve
                    [frames] = VisualAtt(Att1_cell, mask_cell,region_cell, coord_cell, plot_data_all_mni,clrmap,1,true,false);
                else
                    [frames,frames_brain,err,tticks] = PlotAtt(Att1_cell, mask_cell,coord_cell,region_cell,causal,plot_data_all_mni,clrmap,att_cell_c,att_cell_cpre,att_cell_a,ref,investreg);
                end
            end
        else
            [frames] = VisualAtt({Att1}, {mask}, {coord}, plot_data_all_mni,color_map,1,false,false);
        end

    end

end
