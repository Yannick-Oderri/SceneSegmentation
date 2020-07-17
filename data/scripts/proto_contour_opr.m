% ty = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testy.png');
% tx = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testx.png');
% tx_angle = imread('/home/ynki9/Dev/ucf_research/project_edge/cmake-build-debug/test/results/testx.png');
depth = imread('/home/ynki9/Dev/ucf_research/project_edge/data/images/depth/test64.png');

% data = fopen('normals.yml');
% f = textscan(data, '%s', 'Delimiter', ',');
% v = str2double(f{:});
% gv = permute(v, [640, 480, 3]);
% normals = gv;

data = fopen('hangle.yml');
f = textscan(data, '%s', 'Delimiter', ',');
v = str2double(f{:});
gv = reshape(v, 640, 480).';
hori_angle = gv;
