% clear
% close all
% 
% 
% nimg = imread('~/Downloads/normals.png');
% for i = 1:3
%     nimg(:, :, i) = wiener2(nimg(:, :, i), [15, 15]);
% end
vimg = single(nimg()); %340:391, 430:464,:));


hori_img = zeros([size(vimg, 1), size(vimg, 2)]);
vert_img = zeros([size(vimg, 1), size(vimg, 2)]);
sv = [1 2 1];
for j=2:(size(vimg, 1)-1)
    for i=2:(size(vimg, 2)-1)
        hori_vals = [];
        vert_vals = [];
        res = squeeze(vimg(j, i, :));
        if vecnorm(res) == 0
            continue
        end
        res = res/vecnorm(res);
        for v=0:2
                tres1 = squeeze(vimg(j+v-1, i-1, :));                
                tres1 = tres1/vecnorm(tres1);
                tres2 = squeeze(vimg(j+v-1, i+1, :));
                tres2 = tres2/vecnorm(tres2);
                
                %tres = tres2 - tres1;
                val = (1-mean([dot(tres2, res), dot(tres1, res)]))*sv(v+1)*500;
                hori_vals = [hori_vals, val];
        end
%         for u=-1:1              
%             tres1 = squeeze(vimg(j-1, i+u, :));
%             tres1 = tres1/vecnorm(tres1);
%             tres2 = squeeze(vimg(j+1, i+u, :));
%             tres2 = tres2/vecnorm(tres2);
% 
%             tres = tres2 - tres1;
% %             val = 1-mean([dot(tres2, res), dot(tres1, res)]);
% %             vert_vals = [vert_vals, val];
%             val = (1 - dot(tres, res))*1000;
%             vert_vals = [vert_vals, val];  
%         end        
        tval = abs(sum(hori_vals)/5);
        hori_img(j, i) = tval;
        if tval > 1
            hori_img(j, i) = 100;
        end
%         vert_img(j, i) = mean(abs(vert_vals));
    end
end
imtool(hori_img, []);