% clear all;
close all;

imgnum = 201;
% s1 = sprintf('/media/ynki9/DATA/dev2/Amir Code/OSD-dl/OSD-0.2-depth/disparity/test%d.png',imgnum);
%s1 = '/home/ynki9/Documents/occ/2020-07-18_22-44-37/generatedDepthFrame.png';
s1 = sprintf('/home/ynki9/Dev/ucf_research/project_edge/data/images/depth/test%d.png', imgnum);
depth_img = depth; % imread(s1);
depth_img = double(depth_img);
dd = edge(depth_img, 'canny', 0.05);
dst_img = zeros(size(depth_img));
depth_img = padarray(depth_img, [2, 2]);
% depth_img(depth_img < 300 & depth_img > 1500) = 0;
noise = 4.046285945910389e+04;

src_size = size(dst_img);
k1 = ones(1, 5) * -1;
k2 = ones(1, 5).' * -1;
for r = 3:src_size(1)
    for c = 3:(src_size(2))
        g = 0;
        local_mean = 0;
        local_variance = 0;
        
%         col_conv1 = zeros(5, 1);
%         col_conv2 = zeros(5, 1);
%         for u = 0:4
%             col_conv1(u+1) = sum(depth_img((r-2):(r+2), c+u-2) .*  k2, 'all');
%             col_conv2(u+1) = sum((depth_img((r-2):(r+2), c+u-2).^2) .* k2, 'all');
%         end
%         for v = 0:4            
%             local_mean = sum(col_conv1 .* k1, 'all');
%             local_variance = sum(col_conv2 .* k1, 'all');
%         end
        
        for u = 0:4
            for v = 0:4
                pixel = depth_img(r+u-2, c+v-2);
                local_mean = local_mean + pixel;
                local_variance = local_variance + (pixel * pixel);
            end
        end
        
        local_mean = local_mean / 25;
        local_variance = local_variance / 25.0;
        local_variance = local_variance - (local_mean.^2);
        
        g = depth_img(r, c);
        f = g - local_mean;
        g = local_variance - noise;
        g = max(g, 0);
        local_variance = max(local_variance, noise);
        f = f / local_variance;
        f = f * g;
        f = f + local_mean;
        
        dst_img(r-2, c-2) = f;
        
    end
end
figure('Name', 'Stage 1 Filtering');
imshow(dst_img, []);
Idf = dst_img;

% t = wiener2(depth_img, [5 5]);
% figure;
% imshow(t, []);


[Gx, Gy] = imgradientxy(Idf);
Gx = wiener2(Gx,[5 5]) ;
Gy = wiener2(Gy,[5 5]) ;
% figure('Name', 'Stage 2 X Gradient');
% imshow(Gx, []);
% figure('Name', 'Stage 2 Y Gradient');
% imshow(Gy, []);

%%
[Gmag, Gdir] = imgradient(Gx, Gy);
Gxm = Gx;
Gym = Gy;
indx11 = find(abs(Gxm)<2);
indx12 = find(abs(Gym)<2);
indx13 = intersect(indx11,indx12) ;
Gdir(indx13) =  0;
GdirLR = Gdir ;
GdirLR(Gdir>90) = 180-GdirLR(Gdir>90);
GdirLR(Gdir<-90) = -(180+GdirLR(Gdir<-90));

GdirUD = Gdir ;
index3= find(GdirUD<0) ;
GdirUD(index3) = abs(Gdir(index3));
GdirLR = wiener2(GdirLR,[5 5]);
GdirUD = wiener2(GdirUD,[5 5]);
figure('Name', 'Stage 3 X Grad Mirror');
imshow(GdirLR, []);
figure('Name', 'Stage 3 Y Grad mirror');
imshow(GdirUD, []);

%%
[BW_GLR,~] = edge(GdirLR,'canny',0.25);
[BW_GUD,~] = edge(GdirUD,'canny',0.25);
BW30 = or(BW_GLR,BW_GUD) ;
figure('Name', 'CD Results');
imshow(BW30, []);