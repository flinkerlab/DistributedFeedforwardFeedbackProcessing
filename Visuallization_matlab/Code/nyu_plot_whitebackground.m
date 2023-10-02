function fig = nyu_plot(surf_brain,sph,elec,elecname,color,label,aparc_annot,radius,alpha,add_figure,pointalpha,braincolors)

if ~exist('color','var')
    color = 'w';
end
if ~exist('label','var')
    label = 0;
end
if ~exist('alpha','var')
    alpha = 1;
end
if ~exist('radius','var')
    radius = 1;
end
if ~exist('aparc_annot','var')
    aparc_annot = [];
end

if ~exist('add_figure','var')
    add_figure=1;
end

if ~exist('pointalpha','var')
    pointalpha=ones(10000,1);
end

if add_figure
    fig = figure;
end   
% col = [.7 .7 .7];
% col = [0.85 0.85 0.85];
col = [0.92 0.92 0.92];
load('s1.mat');
crv = cscvn(s1');
yy1 = ppval(crv, linspace(0,crv.breaks(end),200));
load('s2.mat');
crv = cscvn(s2');
yy2 = ppval(crv, linspace(0,crv.breaks(end),200));
if strcmp(sph,'both')
    sub_sph1.vert = surf_brain.sph1.coords;
    sub_sph1.tri = surf_brain.sph1.faces;

    sub_sph2.vert = surf_brain.sph2.coords;
    sub_sph2.tri = surf_brain.sph2.faces;
    
    if isempty(aparc_annot) || ~exist(aparc_annot.hemi1,'file') || ~exist(aparc_annot.hemi2,'file')
        col1=repmat(col(:)', [size(sub_sph1.vert, 1) 1]);
        col2=repmat(col(:)', [size(sub_sph2.vert, 1) 1]);
    else
        [~,albl1,actbl1]=fs_read_annotation(aparc_annot.hemi1);
        [~,aa] = ismember(albl1,actbl1.table(:,5));
        aa(aa==0) = 1;
        col1 = actbl1.table(aa,1:3)./255;
        
        [~,albl2,actbl2]=fs_read_annotation(aparc_annot.hemi2);
        [~,bb] = ismember(albl2,actbl2.table(:,5));
        bb(bb==0) = 1;
        col2 = actbl2.table(bb,1:3)./255;
                
        % re-define the electrode color if plot with aparc
        color = 'w';
    end    
    
    trisurf(sub_sph1.tri, sub_sph1.vert(:, 1), sub_sph1.vert(:, 2),sub_sph1.vert(:, 3),...
        'FaceVertexCData', col1,'FaceColor', 'interp','FaceAlpha',alpha);
    hold on;
    trisurf(sub_sph2.tri, sub_sph2.vert(:, 1), sub_sph2.vert(:, 2), sub_sph2.vert(:, 3),...
        'FaceVertexCData', col2,'FaceColor', 'interp','FaceAlpha',alpha);
else    
    if isfield(surf_brain,'coords')==0
        sub.vert = surf_brain.surf_brain.coords;
        sub.tri = surf_brain.surf_brain.faces;
    else
        sub.vert = surf_brain.coords;
        sub.tri = surf_brain.faces;
    end
    
    if isempty(aparc_annot) || ~exist(aparc_annot,'file')
        col=repmat(col(:)', [size(sub.vert, 1) 1]);
    else
        [~,albl,actbl]=fs_read_annotation(aparc_annot);
        [~,cc] = ismember(albl,actbl.table(:,5));
        in = inpolygon(sub.vert(:,2),sub.vert(:,3),yy1(1,:),yy1(2,:));
        cc(in) = 25;
        in = inpolygon(sub.vert(:,2),sub.vert(:,3),yy2(1,:),yy2(2,:));
        cc(in) = 23;
        cc(cc==0) = 1;
        if ~isempty(braincolors)
            col = braincolors(cc,:);
            col(sub.vert(:,3)>40 & cc==25,:) = repmat(braincolors(end,:),length(find(sub.vert(:,3)>40 & cc==25)),1);
        else
            col = actbl.table(cc,1:3)./255;
        end
        % re-define the electrode color if plot with aparc
        %color = 'w';
    end    
    trisurf(sub.tri, sub.vert(:, 1), sub.vert(:, 2), sub.vert(:, 3),...
        'FaceVertexCData', col,'FaceColor', 'interp','FaceAlpha',alpha);
end

shading interp;
lighting gouraud;
material dull;
light;
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
left = outerpos(1);
bottom = outerpos(2); 
ax_width = outerpos(3); 
ax_height = outerpos(4);
ax.Position = [left bottom ax_width ax_height];
axis off;
hold on;

if length(radius) == 1 % same radius for all electrodes
    radius = repmat(radius,1,size(elec,1));
end
for i=1:size(elec,1)
    if length(color)==1
        plotSpheres(elec(i,1),elec(i,2),elec(i,3),radius(i),color{1},pointalpha(i));
    else
       cm = color{2};
       plotSpheres(elec(i,1),elec(i,2),elec(i,3),radius(i),cm(color{1}(i),:),pointalpha(i));
    end
    if label==1
        [x, y, z] = adjust_elec_label(elec(i,:),radius(i));
        text('Position',[x y z],'String',elecname(i,:),'Color','w','VerticalAlignment','top');
    end
end
set(light,'Position',[-1 0 1]); 
    if strcmp(sph,'lh')
        view(270, 0);      
    elseif strcmp(sph,'rh')
        view(90,0);        
    elseif strcmp(sph,'both')
        view(90,90);
    end
    

set(gcf, 'color','white','InvertHardCopy', 'off');
axis tight;
axis equal;
end
%% adjust_elec_label
function [x, y, z] = adjust_elec_label(elec,radius)

if ~exist('radius','var')
    radius = 2;
end

if elec(1)>0
    x = elec(1)+radius;
else
    x = elec(1)-radius;
end

if elec(3)>0
    z = elec(3)+radius;
else
    z = elec(3)-radius;
end

y = elec(2);

end

%% plotSpheres
function [shand]=plotSpheres(spheresX, spheresY, spheresZ, spheresRadius,varargin)

if nargin>4,
    col=varargin{1};
end
if nargin>5,
    alpha=varargin{2};
end
spheresRadius = ones(length(spheresX),1).*spheresRadius;
% set up unit sphere information
numSphereFaces = 25;
[unitSphereX, unitSphereY, unitSphereZ] = sphere(numSphereFaces);

% set up basic plot
sphereCount = length(spheresRadius);

% for each given sphere, shift the scaled unit sphere by the
% location of the sphere and plot
for i=1:sphereCount
sphereX = spheresX(i) + unitSphereX*spheresRadius(i);
sphereY = spheresY(i) + unitSphereY*spheresRadius(i);
sphereZ = spheresZ(i) + unitSphereZ*spheresRadius(i);
shand=surface(sphereX, sphereY, sphereZ,'FaceColor',col,'EdgeColor','none','AmbientStrength',1,'SpecularStrength',0,'DiffuseStrength',0,'FaceAlpha',alpha);
end

end
