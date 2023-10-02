[y,x] = meshgrid(1:4:256,1:4:256);
elecmatrix = MNIc(b,:);
weights = weightsCrt(b,:);
gsp=3;
c_all = zeros(size(x,1)); % create zero heatmap
for i = 1:size(elecmatrix,1) 
v_all = exp(-(x-elecmatrix(i,1)).^2./(2*gsp^2)-(y-elecmatrix(i,3)).^2/(2*gsp^2));
c_all = c_all+v_all;
end


c = zeros(size(x,1)); % create zero heatmap
hit = elecmatrix(find(weights == 1),:);
for i = 1:size(hit,1) 
v = exp(-(x-hit(i,1)).^2./(2*gsp^2)-(y-hit(i,3)).^2/(2*gsp^2));
c = c+v;
end

h = imagesc(squeeze(mri.anatomy(:,160,:)));
hold on;
imagesc(c*10000,'AlphaData', .6)
plot(elecmatrix(:,3),elecmatrix(:,1),'.w');
hold on;
camroll(90)
prob = (c_all>1e-1).*c./c_all;
subplot(131);imagesc(c);colormap(jet);subplot(132);imagesc(c_all);subplot(133);imagesc(prob)

%%
for i = 160
imagesc(squeeze(mri.anatomy(:,i,:)));
hold on;
c=coords+128;
MNIc = MNI+128;
b = find(MNIc(:,2)>(i-2) & MNIc(:,2)<(i+2)); % index of vertices on that layer
plot(c(find(c(:,2)>(i-2) & c(:,2)<(i+2)),3),c(find(c(:,2)>(i-2) & c(:,2)<(i+2)),1),'.w');
plot(MNIc(find(MNIc(:,2)>(i-2) & MNIc(:,2)<(i+2)),3),MNIc(find(MNIc(:,2)>(i-2) & MNIc(:,2)<(i+2)),1),'.k');
pause;
end
