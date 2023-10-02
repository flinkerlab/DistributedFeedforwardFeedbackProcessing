function [c_h,c] = ctmr_gauss_plot_edited(cortex,elecmatrix,weights,cax,addl,CM,gsp,prob_all,aparc_annot,region,plot)
% function [c_h]=ctmr_gauss_plot(cortex,elecmatrix,weights)
%
% projects electrode locationsm (elecmatrix) onto their cortical spots in 
% the left hemisphere and plots about them using a gaussian kernel
% for only cortex use:
% ctmr_gauss_plot(cortex,[0 0 0],0)
% rel_dir=which('loc_plot');
% rel_dir((length(rel_dir)-10):length(rel_dir))=[];
% addpath(rel_dir)

%     Copyright (C) 2009  K.J. Miller & D. Hermes, Dept of Neurology and Neurosurgery, University Medical Center Utrecht
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

%   Version 1.1.0, released 26-11-2009
%
% Modified in 2016 by Liberty Hamilton and then by Jon Kleen in 2017-2021
% for omni-planar and surface casting of epileptiform activity (OPSCEA).
if ~exist('plot','var')
    plot=true;
end

if isempty(elecmatrix); 
    elecmatrix = [0 0 0];
end
if isempty(weights); 
    weights = zeros(size(elecmatrix,1),1);
end

if ~exist('addl','var'); addl=0; end

if exist('CM','var') && ~isempty(CM); cm=CM; else cm=cmSz3D; end

brain=cortex.vert;
% brain=cortex.coords;

if length(weights)~=length(elecmatrix(:,1))
    error('You sent a different number of weights than electrodes in elecmatrix (perhaps a whole matrix instead of vector)')
end

if ~exist('gsp','var') || isempty(gsp); gsp=10; end %default 10

% elecmatrix = elecmatrix(1:640,:);
% weights = weights(1:640,:);
% brain_expand = reshape(brain,[size(brain,1),1,3]);
% elecmatrix_expand = reshape(elecmatrix,[1,size(elecmatrix,1),3]);
% weights_expand = reshape(weights,[1,size(weights,1),1]);
% bb = abs(brain_expand-elecmatrix_expand);
% cc = sum(weights_expand.*exp((-sum(bb.^2,3))/gsp),2);
% cc=cc';

[~,albl1,actbl1]=fs_read_annotation(aparc_annot);
[~,aa] = ismember(albl1,actbl1.table(:,5));
aa(brain(:,3)>40 & aa==25) = 75;
aa(aa==41 & brain(:,2)<0.553486529*brain(:,3)-2.527049117) = 40; %cstg
aa(aa==41 & brain(:,2)>=0.553486529*brain(:,3)-2.527049117) = 42; %rstg
load('annot_regions.mat');
load('s1.mat');
crv = cscvn(s1');
yy1 = ppval(crv, linspace(0,crv.breaks(end),200));
load('s2.mat');
crv = cscvn(s2');
yy2 = ppval(crv, linspace(0,crv.breaks(end),200));
load('s3.mat');
crv = cscvn(s3');
yy3 = ppval(crv, linspace(0,crv.breaks(end),200));
load('s4.mat');
crv = cscvn(s4');
yy4 = ppval(crv, linspace(0,crv.breaks(end),200));
in = inpolygon(brain(:,2),brain(:,3),yy1(1,:),yy1(2,:));
aa(in) = 25;
in = inpolygon(brain(:,2),brain(:,3),yy2(1,:),yy2(2,:));
aa(in) = 23;
in = inpolygon(brain(:,2),brain(:,3),yy3(1,:),yy3(2,:));
aa(in) = 19;
in = inpolygon(brain(:,2),brain(:,3),yy4(1,:),yy4(2,:));
aa(in) = 19;
c=zeros(length(cortex(:,1)),1);
for i=1:length(elecmatrix(:,1))
    b_z=abs(brain(:,3)-elecmatrix(i,3));
    b_y=abs(brain(:,2)-elecmatrix(i,2));
    b_x=abs(brain(:,1)-elecmatrix(i,1));
    reg = region(i,:);
    [~,ind]=ismember(reg(find(reg~=0)),annot_regions);
    if ind==41
        if elecmatrix(i,2)>=(0.553486529*elecmatrix(i,3)-2.527049117)
            ind=42;
        else
            ind = 40;
        end
    end
%     if ind==23
%         d=weights(i)*exp((-(b_x.^2+b_y.^2+b_z.^2))/gsp).*double(aa==ind); %gaussian
%     else
%         d=weights(i)*exp((-(b_x.^2+b_y.^2+b_z.^2))/gsp); %gaussian
%     end
    if ind==40 || ind==42
        d=weights(i)*exp((-(b_x.^2+b_y.^2+b_z.^2))/gsp).*double((aa==40) | (aa==42)); %gaussian
    else
        d=weights(i)*exp((-(b_x.^2+b_y.^2+b_z.^2))/gsp).*double(aa==ind); %gaussian
    end
    c=c+d';
end
if ~isempty(prob_all)
c = c./max(prob_all,1);
end
if plot
    c_h=tripatch(cortex, 'nofigure', c');
    if ~addl; shading interp; end
    a=get(gca);

    d=a.CLim;
    if exist('cax','var') && ~isempty(cax); set(gca,'CLim',[cax(1) cax(2)]); 
    else set(gca,'CLim',[-max(abs(d)) max(abs(d))]); 
    end
    colormap(gca,cm)
    material dull;
    lighting gouraud;
    light;
    axis off
    litebrain('a',.1) %just so it is visible; will replace lighting soon
else
    c_h=figure;
    if ~addl; shading interp; end
    a=get(gca);

    d=a.CLim;
    if exist('cax','var') && ~isempty(cax); set(gca,'CLim',[cax(1) cax(2)]); 
    else set(gca,'CLim',[-max(abs(d)) max(abs(d))]); 
    end
    colormap(gca,cm)
    material dull;
    lighting gouraud;
    light;
    axis off
    litebrain('a',.1) %just so it is visible; will replace lighting soon
end