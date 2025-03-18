
lc = 0.1;
lc_min = 0.005;
l = 2.0;
h = 1.0;
r = 0.1;

Point(1) = {0, 0, 0, lc};
Point(2) = {l, 0, 0, lc};
Point(3) = {l, h, 0, lc};
Point(4) = {0, h, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Point(5) = {l/2, h/2, 0, lc_min};
Point(6) = {l/2 - r, h/2, 0, lc_min};
Point(7) = {l/2 + r, h/2, 0, lc_min};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 6};

Curve Loop(7) = {1, 2, 3, 4, -5, -6};
Plane Surface(8) = {7};

Field[1] = Distance;
Field[1].CurvesList = {5, 6};
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc_min;
Field[2].SizeMax = lc;
Field[2].DistMin = 0.0;
Field[2].DistMax = 0.5;

Background Field = 2;

Mesh 2;


