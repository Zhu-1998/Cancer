clc;
clear all;
close all;


dir = 'G:\EMT\cell2022\normal_AT2like\landscape\WT\';
dir = 'G:\EMT\benchmarking\15cancer\landscape\';

dir = 'G:\EMT\cell2022\cancer_envolution\landscape\0.0001\';
grid=100;
D = 0.0001;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Enter input file name below
filename1= [dir, 'Xgrid.csv'];
Xgrid = readmatrix(filename1);

filename2= [dir, 'Ygrid.csv'];
Ygrid = readmatrix(filename2);

filename3= [dir, 'pot_U.csv'];
pot_U = readmatrix(filename3);

filename4= [dir, 'mean_Fx.csv'];
mean_Fx = readmatrix(filename4);

filename5= [dir, 'mean_Fy.csv'];
mean_Fy = readmatrix(filename5);

filename6= [dir, 'p_tra.csv'];
p_tra = readmatrix(filename6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%plot landscape%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
U = pot_U';
P = p_tra';
mean_Fx = mean_Fx';
mean_Fy = mean_Fy';

mean_Fx(~isfinite(mean_Fx))=0;
mean_Fy(~isfinite(mean_Fy))=0;

l = figure(1);
U(~isfinite(U))=22;
U(U>15)=15;
U(U>18)=18;
U(U>19)=Inf;
surf(Xgrid, Ygrid, U, 'LineStyle', '-', 'FaceColor', 'interp', 'FaceAlpha', 1.0);

xlabel('UMAP1');
ylabel('UMAP2');
zlabel('Potential');
colormap([jet(256)]);
colormap Turbo;
set(gca, 'FontName', 'Arial')
set(gca,'FontSize',20, 'LabelFontSizeMultiplier', 1, 'TitleFontSizeMultiplier', 1)
% pbaspect([1 1 0.8])

set(gca,'TickDir', 'out', 'TickLength', [0.02 0.02])
set(gca, 'LineWidth', 2, 'Color', [0 0 0])
set(gca, 'XColor', [0.00 0.00 0.00])
set(gca, 'YColor', [0.00 0.00 0.00])
set(gca, 'ZColor', [0.00 0.00 0.00])
box on 
box off
colorbar

alpha(1)
shading interp
% camlight;
lighting gouraud;
lighting flat;
lighting phong;

xlim([0.5, 4])  %normal_AT2like
ylim([-3.8, 0.4])  %normal_AT2like
zlim([8.5, 10.2])  %normal_AT2like

xlim([-6, 4.4])  %cancer_envolution
ylim([-8.0, 5])  %cancer_envolution
zlim([8.5, 10.2])  %cancer_envolution
zlim([-1, 19])  %cancer_envolution

xticks([-6, -3, 0, 3]) %
yticks([-6, -3, 0, 3]) %
zticks([5, 15]) %

view([12, 50])
set(gca, 'color', 'white')
hold on

mesh(Xgrid(1:5:end, 1:5:end), Ygrid(1:5:end, 1:5:end), U(1:5:end, 1:5:end)+0.05, 'LineStyle', '-', 'LineWidth', 1.5, 'EdgeColor', 'k', 'FaceColor', 'none')
mesh(Xgrid(1:3:end, 1:3:end), Ygrid(1:3:end, 1:3:end), U(1:3:end, 1:3:end)+0.12, 'LineStyle', '-', 'LineWidth', 1.0, 'EdgeColor', [.2 .2 .2], 'FaceColor', 'none')
mesh(Xgrid(1:3:end, 1:3:end), Ygrid(1:3:end, 1:3:end), U(1:3:end, 1:3:end)+0.02, 'LineStyle', '-', 'LineWidth', 0.5, 'EdgeColor', 'k', 'EdgeAlpha', 0.8, 'FaceColor', 'none')
mesh(Xgrid(1:2:end, 1:2:end), Ygrid(1:2:end, 1:2:end), U(1:2:end, 1:2:end)+0.75, 'LineStyle', '-', 'LineWidth', 1.0, 'EdgeColor', [.2 .2 .2], 'FaceColor', 'none')

hold on
saveas( l, [dir, 'landscape.fig']); 
print(l, [dir, 'landscape.tif'],'-r600','-dtiff');
print(l, '-r600', '-dpdf', [dir, 'landscape.pdf']);


s = figure(2);
pcolor(Xgrid, Ygrid, U);
pbaspect([1 1 1])
dx = (max(max(Xgrid))-min(min(Xgrid)))/200;
dy = (max(max(Ygrid))-min(min(Ygrid)))/200;
[GUx,GUy] = gradient(U,dx,dy);
[GPx,GPy] = gradient(P,dx,dy);
Jx = mean_Fx.*P - D*GPx ;
Jy = mean_Fy.*P - D*GPy ;

mg = 1:5:200;
ng = mg;
E=Jy.^2+Jx.^2;
JJx=Jx./(sqrt(E)+eps);
JJy=Jy./(sqrt(E)+eps);

hold on
quiver(xx(mg,ng),yy(mg,ng),JJx(mg,ng),JJy(mg,ng),0.5,'color','k', 'LineWidth',1);

Fgradx = -D*GUx;
Fgrady = -D*GUy;
EEE=Fgradx.^2+Fgrady.^2;
FFgradx=Fgradx./(sqrt(EEE)+eps);
FFgrady=Fgrady./(sqrt(EEE)+eps);

hold on;
quiver(xx(mg,ng),yy(mg,ng),FFgradx(mg,ng),FFgrady(mg,ng),0.5,'color','w', 'LineWidth',1);

saveas(figure(2), [dir, 'figure/flux_grad_path.fig']); 
print(figure(1), '-r600', '-dpdf', [dir, 'figure/flux_grad_path.pdf']);

