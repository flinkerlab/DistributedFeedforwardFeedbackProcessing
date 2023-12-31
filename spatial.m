clc;
addpath('Visuallization_matlab/Code');
addpath('cbrewer');
% set the root_dir to the directory where the Code and visualization data is located 
root_dir = 'fig3';
% Choose a iiject {'NY742', 'NY717', 'NY749'} -- we don't have iiject brains other than these but MNI should work for all Ss
Sj = 'NY717';
% get paths to all visualization files
paths = get_path(root_dir,Sj);
% load electrode (channel) table
channel_info_all = get_channel_info(paths);
% get the relevant plotting options by choosing {'mni' or 'iij'} and hemisphere {'lh' or 'rh'} 
plot_data_all_mni = get_plot_data(paths, 'mni', 'lh');

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

attrs = {'spec'};
ylims_min = [0,0,0,0,0,0,0];
% ylims_max = [1,1.5,1.5,1.5,1,2.0,1];
ylims_max = [1,1,1,1,1,1.5,1];

wreg=true;
temp=false;
tau=false;
return_curve=false;
dosmooth = true;
global win_len 
win_len = 1;%11;
entire_period = true;
post = true;
visdiff = false;
postabs = true;%post || ~visdiff; % sparse if true
max_att = false;%post || ~visdiff;
max_anker = true;
clrmap = color_map;

Ss = {'717','742','749','829','798'};
S_len = length(Ss);
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
        col=S_len;
    end
end

invest_regions = {{'cSTG','rSTG','mMTG','cMTG','rMTG',...
    'inferiorprecentral','superiorprecentral','postcentral','supramarginal',...
    'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'}};

all_regions = {'cSTG','rSTG','mMTG','cMTG','rMTG',...
    'inferiorprecentral','superiorprecentral','postcentral','supramarginal',...
    'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'};

%causalpre
% invest_regions = {{'mMTG','rMTG',...
%     'inferiorprecentral','superiorprecentral','postcentral','supramarginal',...
%     'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'}};

%causal
% invest_regions = {{'cSTG','rSTG',...
%     'inferiorprecentral','superiorprecentral','postcentral',...
%     'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'}};

% %anticausal
invest_regions = {{'cSTG','mMTG',...
    'inferiorprecentral','superiorprecentral','postcentral','supramarginal',...
    'parsopercularis','parstriangularis','caudalmiddlefrontal'}};

% invest_regions = {{'cSTG','inferiorprecentral','superiorprecentral','postcentral','parsopercularis','parstriangularis','caudalmiddlefrontal'}};

% invest_regions = {{'mSTG'},...
%     {'inferiorprecentral','superiorprecentral','postcentral','supramarginal'},...
%     {'parsopercularis','parstriangularis','rostralmiddlefrontal','caudalmiddlefrontal'}};
invest_regions_legend = {{'cSTG','vPrCG','dPrCG','PsCG','pOp','pTri','cMFG'}};
ir = 1;
[~,investregionindex] = ismember(invest_regions{ir},all_regions);


% colors{1}=[givemecolor('Purples',2,0.6,1);givemecolor('Blues',3,0.4,1);givemecolor('YlGn',4);givemecolor('OrRd',4)];

fig=figure();
dumm = 1;
avg_cell = {0,0,0};
regioncolor=false;
bar_plot=false;
do_gaussian=true;
norm_contr_elec=false;
if bar_plot
    haxis = tight_subplot(row-1,col,[0,0],[0.3,0],[0.1,0]);
else
%     haxis = tight_subplot(row-1,col,[0,0],0,[0.1,0]); 
    haxis = tight_subplot(row-1,col,[0,0],0,[0,0]); 
end
for a = 1:length(attrs)
    [frames1,frames2,frames,avg] = attr_vis_contrast(attrs{a},post,visdiff,postabs,max_att,plot_on_same_brain,above_median,temp,wreg,entire_period,avg_cell,regioncolor,bar_plot,norm_contr_elec);
end
set(gca,'visible','off')

linkaxes(haxis,'xy');

% ---------------------------------------------------------------------------
% ----------------------------Relevant Functions ----------------------------
% ---------------------------------------------------------------------------

function [frames,prob,prob0] = VisualAtt(Att, Mask, Region, coord, brain,color_map,maxv,alpha,div,annotation,bar_plot,gaussianplot,scale,gausskernel,causal,plot)
    if ~exist('annotation','var')
        annotation = [];
    end
    if ~exist('plot','var')
        plot=true;
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
        disp('------');
       
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
        gsp=18;
        if gaussianplot
%             load('S.mat')
            Lcrtx = load('ch2_template_lh_pial.mat');
            Lcrtx.tri = Lcrtx.faces;Lcrtx.vert = Lcrtx.coords;
            data = Att_frame(ind_active);
            cax=[0,0.8];
            [fig,prob_all]=ctmr_gauss_plot_edited(Lcrtx,elec,ones(size(elec,1),1),cax,0,hot(65),gsp*gausskernel,[],annot,validregion); 
           
            prob0 = prob_all;
            close(fig)
            if div
                cax=[-0.7,0.7];
            else
                cax = [0,0.7];
            end
            if exist('scale','var')
                cax = scale;
            end
            if ~div
                color_map(1:int64(size(color_map,1)*0.19/cax(2)),:)=1;
            end
%             color_map(1:int64(size(color_map,1)*0.03516/cax(2)),:)=1;
            [fig,prob]=ctmr_gauss_plot_edited(Lcrtx,elec,data,cax,0,color_map,gsp*gausskernel,prob_all,annot,validregion,plot); 
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
    elseif strcmp(lower(visual_mode), 'iij')
         if strcmp(lower(sph), 'lh')
            plot_data_all.surf_brain = load(paths.Sj_lh);
            plot_data_all.sph = 'lh';
            plot_data_all.annot = paths.Sj_lh_annot;
            plot_data_all.coord_ind = 6:8;
        elseif strcmp(lower(sph), 'rh')
            plot_data_all.surf_brain = load(paths.Sj_rh);
            plot_data_all.sph = 'rh';
            plot_data_all.annot = paths.Sj_rh_annot;
            plot_data_all.coord_ind = 6:8;
        else
            error('sph mode not supported! use lh or rh');
        end
    else
        error('visual_mode not supported! use mni or iij');
    end
end
function [channel_info_all] = get_channel_info(paths)
    % load coordinate files
    coordinates = readtable(paths.coordinate_file);
    ind = [1:8,10,11];
    coordinates = coordinates(:,ind);
    coordinates.channel= [1:size(coordinates,1)]';
    Sj.sebject = repmat(paths.Sj,size(coordinates,1),1);
    Sj = struct2table(Sj);
    channel_info_all = [coordinates, Sj];
end
function [paths] = get_path(root_dir, Sj)
    addpath(genpath(root_dir));
    paths.root_dir = root_dir;
    paths.Sj = Sj;
    paths.Sj_path = [root_dir, filesep, 'Data', filesep, Sj];
    paths.coordinate_file = [paths.Sj_path, filesep, 'coordinates.csv'];
    paths.MNI_lh = [paths.root_dir, filesep, 'Data', filesep, 'MNI', filesep, 'ch2_template_lh_pial_120519.mat'];
    paths.MNI_rh = [paths.root_dir, filesep, 'Data', filesep, 'MNI', filesep, 'ch2_template_rh_pial_120519.mat'];
    paths.MNI_lh_annot = [paths.root_dir, filesep, 'Data', filesep, 'MNI', filesep, 'ch2_template.lh.aparc.split_STG_MTG.annot'];
    paths.MNI_rh_annot = [paths.root_dir, filesep, 'Data', filesep, 'MNI', filesep, 'ch2_template.rh.aparc.split_STG_MTG.annot'];
    lh_pial_file = dir([paths.Sj_path,filesep,'*lh_pial_surf.mat']);
    lh_annot_file = dir([paths.Sj_path,filesep,'lh*_STG_MTG.annot']);
    rh_pial_file = dir([paths.Sj_path,filesep,'*rh_pial_surf.mat']);
    rh_annot_file = dir([paths.Sj_path,filesep,'rh*_STG_MTG.annot']);
    if (length(lh_pial_file)~=0)
        paths.Sj_lh = [paths.Sj_path, filesep, lh_pial_file(1).name];
    else
        paths.Sj_lh = [];
    end
    if (length(rh_pial_file)~=0)
        paths.Sj_rh = [paths.Sj_path, filesep, rh_pial_file(1).name];
    else
        paths.Sj_rh = [];
    end
    if (length(lh_annot_file)~=0)
        paths.Sj_lh_annot = [paths.Sj_path, filesep, lh_annot_file(1).name];
    else
        paths.Sj_lh_annot = [];
    end
    if (length(rh_annot_file)~=0)
        paths.Sj_rh_annot = [paths.Sj_path, filesep, rh_annot_file(1).name];
    else
        paths.Sj_rh_annot = [];
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
        sorted = sort(data(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data,1),size(data,2),1,1)==1 & ~isnan(data)));
        min25 = sorted(int32(0.01*length(sorted))+1);
        max75 = sorted(int32(0.99*length(sorted)));
        md = median(sorted);
        sorted_temp = sort(data_temp(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data_temp,1),size(data_temp,2),1,1)==1 & ~isnan(data_temp)));
        min25_temp = sorted_temp(int32(0.01*length(sorted_temp))+1);
        max75_temp = sorted_temp(int32(0.99*length(sorted_temp)));
        md_temp = median(sorted_temp);

        if median_baseline
            att = (data-md)/(max75-min25);
            att_temp = (data_temp-md_temp)/(max75_temp-min25_temp);
            att_temp(isnan(att_temp))=0;
            att_temp(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data_temp,1),size(data_temp,2),1,1)==0) = 0;
        else
            att = (data-min25)/(max75-min25);
            att_temp = (data_temp-min25_temp)/(max75_temp-min25_temp);
            att_temp(isnan(att_temp))=0;
            att_temp(repmat(reshape(mask,[1,1,size(mask,1),size(mask,2)]),size(data_temp,1),size(data_temp,2),1,1)==0) = 0;
        end
    end
end

function [frames1,frames2,frames,Att1_cell] = attr_vis_contrast(attr,post,visdiff,postabs,max_att,plot_on_same_brain,above_median,temp,wreg,entire_period,avg_data,annotation,bar_plot,norm_contr_elec)    
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

    dsrate = 8/dsrate_org;
    S_len = 4;
    Att1_cell = {};
    Att2_cell = {};
    diff_cell = {};
    diff2_cell = {};
    att_cell1 = {};
    att_cell2 = {};
    mask_cell = {};
    coord_cell = {};
    region_cell = {};
    Ss = {'717','742','749','798','829'};
%     Ss = {'717'};
    cmax=1.2;
    cax=[-0.7,0.7];
    root_dir = 'fig3';
    attrs = {'spec'};

    [color_map_div]=cbrewer('div', 'RdBu', 256,'PCHIP');
    color_map_div_ = color_map_div(end:-1:1,:);
    color_map = hot(256);
    color_map_ = color_map(end:-1:1,:);
    color_ind = value2colorind(ones(128,1),'hot',[0,1]); 
    color_ind = [color_ind,color_ind,color_ind];
    for ii=1:S_len
        load(['fig3/NY_',Ss{ii},'_elec_entiregrid.mat']);
        load(['fig3/contrib_',Ss{ii},'.mat']);
%         onregion = {'cSTG','mSTG','parstriangularis','parsopercularis','precentral','postcentral','inferiorparietal','supramarginal'};
        onregion = {'cSTG','mSTG','parstriangularis','parsopercularis','precentral','postcentral'};
        
        region_cell{ii} = regions;

        Sj = ['NY',Ss{ii}];
        % get paths to all visualization files
        paths = get_path(root_dir,Sj);
        channel_info_all = get_channel_info(paths);
        plot_data_all_mni = get_plot_data(paths, 'mni', 'lh');
        coord = mni;
        
        Att_cell1{ii} = Att1_causal;
        Att_cell2{ii} = Att1_anticausal;

        mask_cell{ii} = mask;
        coord_cell{ii} = coord;
        if plot_on_same_brain 
            if ii == S_len
                [frames1,prob1,prob0] = VisualAtt(Att_cell1, mask_cell, region_cell,coord_cell, plot_data_all_mni,color_map_,cmax,true,false,annotation,bar_plot,true,[0,0.858]); title(gca,'Anti-causal contribution'); % anticausal
                [frames2,prob2,prob0] = VisualAtt(Att_cell2, mask_cell, region_cell,coord_cell, plot_data_all_mni,color_map_,cmax,true,false,annotation,bar_plot,true,[0,0.858]); title(gca,'causal contibution'); % causal
                frames={};Lcrtx = load('ch2_template_lh_pial.mat'); Lcrtx.tri = Lcrtx.faces;Lcrtx.vert = Lcrtx.coords; frames{1}=tripatch(Lcrtx, 'nofigure', 3*(prob1'-prob2')); shading interp;colormap(gca,color_map_div_);material dull;lighting gouraud;light;axis off;viewangle = 'l';litebrain(viewangle,.8);set(gca,'CLim',[cax(1) cax(2)]);title(gca,'contribution contrast'); 
            end
        else
            [frames] = VisualAtt({Att1}, {mask}, {regions}, {coord}, plot_data_all_mni,color_map,1,false,false,annotation,bar_plot);

        end

    end

end
