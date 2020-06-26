clear all;
close all;
img = imread('/home/ynki9/Dev/ucf_research/project_edge/data/images/depth/test12.png');
timg = img;


start_pos = [265 160];
end_pos = [228 165];
lvec = end_pos - start_pos;
mag = norm(lvec);
lvec = lvec / mag;
line_params = [start_pos; end_pos];
window_size = 4;

ang = atan2(lvec(2), lvec(1));
rot = [cos(ang) -sin(ang); sin(ang) cos(ang)];

roi_1 = [];
roi_2 = [];
for i = 0:mag
    for j = 0:window_size        
        tpoint = [i, j^2+1];%(rot*[j^2+1, 0].').'
        point = int32((rot*tpoint')' + start_pos); %int32(tpoint + ((lvec * i) + start_pos))
        depth_val = timg(point(2), point(1));
        img(point(2), point(1)) = 3000;
        if depth_val ~= 0
            roi_1 = [roi_1, depth_val];
        end   
        tpoint = [i, -1*(j^2+1)]; %(rot*[-(j^2+1), 0].').';
        point = int32((rot*tpoint')' + start_pos); %int32(tpoint + ((lvec * i) + start_pos));
        depth_val = timg(point(2), point(1));
%         img(point(2), point(1)) = 3000;
        if depth_val ~= 0
            roi_2 = [roi_2, depth_val];
        end
    end
end

imshow(img, []);
% imtool(img, []);

roi_r = mean(roi_1);
roi_l = mean(roi_2);