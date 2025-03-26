
lc = 0.1;

l = 2.0;
h = 1.0;

Point(1) = {0, 0, 0, lc};
Point(2) = {l, 0, 0, lc};
Point(3) = {l, h, 0, lc};
Point(4) = {0, h, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

Physical Curve("wall_left", 9) = {4};
Physical Curve("free_bot", 10) = {1};
Physical Curve("force_right", 11) = {2};
Physical Curve("free_top", 12) = {3};