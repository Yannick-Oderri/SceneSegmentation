% ty = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testy.png');
% tx = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testx.png');
% tx_angle = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testx.png');
depth = imread('/home/ynki9/Dev/ucf_research/project_edge/data/images/depth/test301.png');

%% Load yml
width = 640;
height = 576;
data = fopen('synthetic_contours.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, width, height).';
fdepth = gv;

%% Load yml Color Image
data = fopen('fdep.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, 4, width, height);
gv = gv(1:3, :, :);
normals = permute(gv, [3, 2, 1]);

%% Load yml Image
data = fopen('hangle.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, width, height).';
% 
hangle= gv;

data = fopen('lrdir.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, width, height).';
lrdir= gv;


%% Generate Gradient Map
width = 640;
height = 576;

data = fopen('ygrad.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, width, height).';
ygrad = gv;

data = fopen('xgrad.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, width, height).';
xgrad = gv;

txgrad = reshape(xgrad, [1 width*height]);
tygrad = reshape(ygrad, [1 width*height]);
uy = repelem(1:height, width);
tty =reshape(uy, [width height]);
uy = reshape(tty.', [1 width*height]);
ux = repelem(1:width, height);
ttx =reshape(ux, [height width]);
ux = reshape(ttx, [1 width*height]);
figure;
quiver(ux, uy*-1, txgrad, tygrad, 30);
magrad = sqrt(xgrad.^2+ygrad.^2);

%% Generate Gradient From depth
width = 640;
height = 480;



% data = fopen('avg_depth.yml');
% f = textscan(data, '%s', 'Delimiter', ',');
% v = str2double(f{:});
% gv = reshape(v, width, height).';
% depth = gv;

[xgrad, ygrad] = imgradientxy(depth, 'central');
ygrad = ygrad * -1;

txgrad = reshape(xgrad, [1 width*height]);
tygrad = reshape(ygrad, [1 width*height]);
uy = repelem(1:height, width);
tty =reshape(uy, [width height]);
uy = reshape(tty.', [1 width*height]);
ux = repelem(1:width, height);
ttx =reshape(ux, [height width]);
ux = reshape(ttx, [1 width*height]);
figure;
quiver(ux, uy*-1, txgrad, tygrad, 30);
magrad = sqrt(xgrad.^2+ygrad.^2);
imtool(magrad, []);


%%
val = arrayfun(@(dxu, dyu, dxv, dyv) dot([dxu dyv], [dyu, dxv])/(norm([dxu, dyv])*norm([dyv, dxu])), dxu, dyu, dxv, dyv)
