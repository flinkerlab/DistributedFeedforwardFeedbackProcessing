load('smoothcurve_active.mat');
load('smoothcurve_passive.mat');
load('smoothcurve_imagine.mat');
load('smootherr_active.mat');
load('smootherr_passive.mat');
load('smootherr_imagine.mat');
load('name4plot.mat');
invest_regions = name4plot;
% invest_regions={'rMTG','mMTG','cMTG','vPreCG','supramarginal','rMFG','pOb'};
colors{1}=[givemecolor('Purples',2,0.6,1);givemecolor('Blues',3,0.4,1);givemecolor('YlGn',4);givemecolor('OrRd',5)];
figure();
rows = ceil(length(invest_regions)/2);
cols=2;
tticks = [0:55]*8;
% haxis = tight_subplot(rows,cols,[0.03,0],[0.07,0.02],[0.02,0]); 
haxis = tight_subplot(rows,cols,[0.005,0.1],[0.02,0.04],[0.1,0]); 
for ir=1:length(invest_regions)
   [~,ind]=ismember(invest_regions(ir),name4plot);
   axes(haxis(sub2ind([rows,cols],ir)));
   p=plot(tticks,smoothcurve_passive(:,ind),'.','LineWidth',2,'Color',[colors{1}(ind,1:3),0.7]); hold on;
   plot(tticks,smoothcurve_passive(:,end),'.','LineWidth',2,'Color',[0.75,0.75,0.75,0.7]); 
   plot(tticks,smoothcurve_active(:,ind),'LineWidth',2,'Color',[colors{1}(ind,1:3),0.7]); 
   plot(tticks,smoothcurve_active(:,end),'LineWidth',2,'Color',[0.75,0.75,0.75,0.7]); 
   ax=gca;ax.FontSize=20;  set(gca,'linewidth',2);
   set(haxis(sub2ind([rows,cols],ir)),'box','off','TickDir','out');
   ff = fill([tticks,fliplr(tticks)] , [smoothcurve_passive(:,ind)'-smootherr_passive(:,ind)',fliplr(smoothcurve_passive(:,ind)'+smootherr_passive(:,ind)')],colors{1}(ind,1:3),'FaceAlpha',0.3,'EdgeAlpha',0); 
   ff = fill([tticks,fliplr(tticks)] , [smoothcurve_active(:,ind)'-smootherr_active(:,ind)',fliplr(smoothcurve_active(:,ind)'+smootherr_active(:,ind)')],colors{1}(ind,1:3),'FaceAlpha',0.3,'EdgeAlpha',0);
   ff = fill([tticks,fliplr(tticks)] , [smoothcurve_passive(:,end)'-smootherr_passive(:,end)',fliplr(smoothcurve_passive(:,end)'+smootherr_passive(:,end)')],[0.75,0.75,0.75],'FaceAlpha',0.3,'EdgeAlpha',0);
   ff = fill([tticks,fliplr(tticks)] , [smoothcurve_active(:,end)'-smootherr_active(:,end)',fliplr(smoothcurve_active(:,end)'+smootherr_active(:,end)')],[0.75,0.75,0.75],'FaceAlpha',0.3,'EdgeAlpha',0);
   pbaspect([5 1 1])
   if ir<(length(invest_regions)-1)
      set(gca,'xticklabel',[])
   end
   xlim(haxis(sub2ind([rows,cols],ir)),[tticks(1),tticks(end)]);
end


%% 
load('tau_elec_fdiff_active.mat');
region_value_active=region_value{1};
load('tau_elec_fdiff_passive.mat');
region_value_passive=region_value{1};

rows = size(region_value_active,2)/6;
cols=6;
figure;
haxis = tight_subplot(rows,cols,[0,0.0],[0.02,0],[0.1,0.1]); 

for ir=1:size(region_value_active,2)
   axes(haxis(sub2ind([rows,cols],ir)));
   p=plot(tticks,region_value_passive(:,ir),'.','LineWidth',2,'Color',[colors{1}(1,1:3),0.7]); hold on;
   plot(tticks,smoothcurve_passive(:,end),'.','LineWidth',2,'Color',[0.75,0.75,0.75,0.7]); 
   plot(tticks,region_value_active(:,ir),'LineWidth',2,'Color',[colors{1}(1,1:3),0.7]); 
   plot(tticks,smoothcurve_active(:,end),'LineWidth',2,'Color',[0.75,0.75,0.75,0.7]); 
   ax=gca;ax.FontSize=20;  set(gca,'linewidth',2);
   set(haxis(sub2ind([rows,cols],ir)),'box','off','TickDir','out');
   pbaspect([2 1 1])
   if ir<(size(region_value_active,2)-cols)
      set(gca,'xticklabel',[])
   end
   xlim(haxis(sub2ind([rows,cols],ir)),[tticks(1),tticks(end)]);
end


%% 

function [clr] = givemecolor(name,num,min,max)
if ~exist('min','var')
    min=0.35;
end
if ~exist('max','var')
    max=1;
end
[color_map_]=getPyPlot_cMap(name, 256);
if num==1
    clr = color_map_(int64(256*max),:);
else
    clr = color_map_(int64(linspace(256*max,256*min,num)),:);
end

end