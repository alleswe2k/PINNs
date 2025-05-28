SetFactory("OpenCASCADE");

lc = 0.05;
lc_min = lc / 20;
distance_mesh = 0.5;

l = 2.0;
h = 1.0;
r = 0.15;

hole1_x = l/2;
hole1_y = h/2;

Point(1) = {0, 0, 0};
Point(2) = {l, 0, 0};
Point(3) = {l, h, 0};
Point(4) = {0, h, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Circle(5) = {l/2, h/2, 0, r};

Curve Loop(6) = {5};
Curve Loop(7) = {1, 2, 3, 4};

Plane Surface(8) = {7, -6};

Physical Curve("wall_left", 9) = {4};
Physical Curve("free_bot", 10) = {1};
Physical Curve("force_right", 11) = {2};
Physical Curve("free_top", 12) = {3};
Physical Curve("hole1", 13) = {5};

Field[1] = Distance;
Field[1].CurvesList = {5};
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc_min;
Field[2].SizeMax = lc;
Field[2].DistMin = 0.0;
Field[2].DistMax = distance_mesh;

Background Field = 2;

Mesh 2;


