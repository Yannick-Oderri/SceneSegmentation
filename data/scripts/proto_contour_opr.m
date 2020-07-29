% ty = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testy.png');
% tx = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testx.png');
% tx_angle = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testx.png');
depth = imread('/home/ynki9/Dev/ucf_research/project_edge/data/images/depth/test65.png');


%% Load yml Color Image
data = fopen('normals.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, 4, 640, 480);
gv = gv(1:3, :, :);
normals = permute(gv, [3, 2, 1]);

%% Load yml Image
% data = fopen('hangle.yml');
% f = textscan(data, '%s', 'Delimiter', ',');
% v = str2double(f{:});
% gv = reshape(v, 640, 480).';
% % 
% hangle= gv;

data = fopen('lrdir.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, 640, 480).';
lrdir= gv;


%% Generate Gradient Map

% data = fopen('ygrad.yml');
% f = textscan(data, '%s', 'Delimiter', ',');
% v = str2double(f{:});
% gv = reshape(v, 640, 480).';
% ygrad = gv;
% 
% 
% data = fopen('xgrad.yml');
% f = textscan(data, '%s', 'Delimiter', ',');
% v = str2double(f{:});
% gv = reshape(v, 640, 480).';
% xgrad = gv;

txgrad = reshape(xgrad, [1 640*480]);
tygrad = reshape(ygrad, [1 640*480]);
uy = repelem(1:480, 640);
tty =reshape(uy, [640 480]);
uy = reshape(tty.', [1 640*480]);
ux = repelem(1:640, 480);
ttx =reshape(ux, [480 640]);
ux = reshape(ttx, [1 640*480]);
figure;
q = quiver(ux, uy*-1, txgrad, tygrad);
