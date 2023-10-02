meandata=[-0.81,-0.78,-0.735,-0.62,-0.60,-0.55,-0.47,-0.37,-0.21,-0.12,-0.05,0.21,0.47];
errdata=[0.15,0.3,0.2,0.21,0.25,0.23,0.1,0.35,0.32,0.42,0.39,0.21,0.28]/1.7;
plot([1:length(meandata)], meandata,'Color','k','Marker','o');
hold on;
errorbar([1:length(meandata)], meandata, errdata,'Color','k');
xlim([0,length(meandata)+1]);
set(gca,'fontweight','bold');