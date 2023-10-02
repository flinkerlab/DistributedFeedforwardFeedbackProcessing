load('region_value_f0_percept.mat'); region_value_f0_percept=region_value;
load('regions4barplot_f0_percept.mat'); regions4barplot_f0_percept=regions4barplot;
load('region_value_fdiff_percept.mat'); region_value_fdiff_percept=region_value;
load('regions4barplot_fdiff_percept.mat'); regions4barplot_fdiff_percept=regions4barplot;
load('region_value_f0_prod.mat'); region_value_f0_prod=region_value;
load('regions4barplot_f0_prod.mat'); regions4barplot_f0_prod=regions4barplot;
load('region_value_fdiff_prod.mat'); region_value_fdiff_prod=region_value;
load('regions4barplot_fdiff_prod.mat'); regions4barplot_fdiff_prod=regions4barplot;

[f0_percept_dorsal,f0_percept_ventral]=get_dorsal_ventral_division(region_value_f0_percept,regions4barplot_f0_percept);
[fdiff_percept_dorsal,fdiff_percept_ventral]=get_dorsal_ventral_division(region_value_fdiff_percept,regions4barplot_fdiff_percept);
[f0_prod_dorsal,f0_prod_ventral]=get_dorsal_ventral_division(region_value_f0_prod,regions4barplot_f0_prod);
[fdiff_prod_dorsal,fdiff_prod_ventral]=get_dorsal_ventral_division(region_value_fdiff_prod,regions4barplot_fdiff_prod);

f0_percept_dorsal_=f0_percept_dorsal(f0_percept_dorsal>fdiff_percept_dorsal-0);
fdiff_percept_dorsal=fdiff_percept_dorsal(f0_percept_dorsal>fdiff_percept_dorsal-0);
f0_percept_dorsal = f0_percept_dorsal_;

% fdiff_percept_ventral_=fdiff_percept_ventral(fdiff_percept_ventral>f0_percept_ventral-0.05);
% f0_percept_ventral=f0_percept_ventral(fdiff_percept_ventral>f0_percept_ventral-0.05);
% fdiff_percept_ventral = fdiff_percept_ventral_;


get_plot_data_meanerror({f0_percept_dorsal,fdiff_percept_dorsal,f0_percept_ventral,fdiff_percept_ventral});
set(gca,'fontweight','bold');
get_plot_data_meanerror({f0_prod_dorsal,fdiff_prod_dorsal,f0_prod_ventral,fdiff_prod_ventral});
set(gca,'fontweight','bold');

vec_for_anova = [f0_percept_dorsal,fdiff_percept_dorsal,f0_percept_ventral,fdiff_percept_ventral];
feature=[ones(1,size(f0_percept_dorsal,2)),2*ones(1,size(fdiff_percept_dorsal,2)),ones(1,size(f0_percept_ventral,2)),2*ones(1,size(fdiff_percept_ventral,2))];
region=[ones(1,size(f0_percept_dorsal,2)),ones(1,size(fdiff_percept_dorsal,2)),2*ones(1,size(f0_percept_ventral,2)),2*ones(1,size(fdiff_percept_ventral,2))];
[p,~,stats] = anovan(vec_for_anova,{feature,region},'model',2,'varnames',{'feature','region'});
[results,means,~,gnames] = multcompare(stats,'Dimension',[1 2]);
tbl = array2table(results,'VariableNames', ...
    {'GroupA','GroupB','LowerLimit','AminusB','UpperLimit','P_value'});
tbl.('GroupA')=gnames(tbl.('GroupA'));
tbl.('GroupB')=gnames(tbl.('GroupB'));


vec_for_anova = [f0_prod_dorsal,fdiff_prod_dorsal,f0_prod_ventral,fdiff_prod_ventral];
feature=[ones(1,size(f0_prod_dorsal,2)),2*ones(1,size(fdiff_prod_dorsal,2)),ones(1,size(f0_prod_ventral,2)),2*ones(1,size(fdiff_prod_ventral,2))];
region=[ones(1,size(f0_prod_dorsal,2)),ones(1,size(fdiff_prod_dorsal,2)),2*ones(1,size(f0_prod_ventral,2)),2*ones(1,size(fdiff_prod_ventral,2))];
[p,~,stats] = anovan(vec_for_anova,{feature,region},'model',2,'varnames',{'feature','region'});
[results,means,~,gnames] = multcompare(stats,'Dimension',[1 2]);
tbl = array2table(results,'VariableNames', ...
    {'GroupA','GroupB','LowerLimit','AminusB','UpperLimit','P_value'});
tbl.('GroupA')=gnames(tbl.('GroupA'));
tbl.('GroupB')=gnames(tbl.('GroupB'));





[clrmap]=cbrewer('div', 'PRGn', 256,'PCHIP');
clrp3 = clrmap(int32(length(clrmap)*0.85),:);
clrn3 = clrmap(int32(length(clrmap)*0.15),:);
figure();
xlabes = {'f0 preception','f2/f1 preception', 'f0 production', 'f2/f1 production'};
[dorsal_data,dorsal_region] = get_plot_data({f0_percept_dorsal,fdiff_percept_dorsal,f0_prod_dorsal,fdiff_prod_dorsal},xlabes);
boxplot(dorsal_data,dorsal_region,'PlotStyle','compact','Colors',clrp3);   set(gca,'XTickLabel',{' '});
xticks([1:length(xlabes)]);xticklabels(xlabes);xtickangle(45); set(gca,'fontweight','bold');
figure();
[ventral_data,ventral_region] = get_plot_data({f0_percept_ventral,fdiff_percept_ventral,f0_prod_ventral,fdiff_prod_ventral},xlabes);
boxplot(ventral_data,ventral_region,'PlotStyle','compact','Colors',clrn3);   set(gca,'XTickLabel',{' '});
xticks([1:length(xlabes)]);xticklabels(xlabes);xtickangle(45); set(gca,'fontweight','bold');

 
function [reorg_data,region] = get_plot_data(data,label)
    reorg_data = [];
    region = [];
    for bin=1:length(data)
        reorg_data = [reorg_data, data{bin}];
        l = ['                   '];
        l(1:length(label{bin})) = label{bin};
        region = [region; repmat(l,length(data{bin}),1)];
    end
end

function get_plot_data_meanerror(data)
    reorg_data = [];
    reorg_err = [];
    for bin=1:length(data)
        reorg_data = [reorg_data, mean(data{bin})];
        reorg_err = [reorg_err, sqrt(var(data{bin})/length(data{bin}))];
    end
    figure;
    errorbar([1:length(data)], reorg_data, reorg_err,'o');
    xlim([0,length(data)+1]);
end


function [dorsal,ventral] = get_dorsal_ventral_division(data,regions)
    dorsal = []; ventral =[];
    dorsal = [ dorsal, data{find(ismember(regions,'superiorprecentral'))}];
    dorsal = [ dorsal, data{find(ismember(regions,'caudalmiddlefrontal'))}];
    ventral = [ ventral, data{find(ismember(regions,'parstriangularis'))}];
    ventral = [ ventral, data{find(ismember(regions,'parsopercularis'))}];
    ventral = [ ventral, data{find(ismember(regions,'parsorbitalis'))}];
    ventral = [ ventral, data{find(ismember(regions,'rostralmiddlefrontal'))}];
end
