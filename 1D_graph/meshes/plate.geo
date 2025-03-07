// Gmsh project created on Fri Mar 07 15:54:22 2025
SetFactory("OpenCASCADE");
//+
Point(1) = {0, -0, 0, 1.0};
//+
Point(2) = {1, -0, 0, 1.0};
//+
Point(3) = {0, 1, 0, 1.0};
//+
Point(4) = {1, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 4};
//+
Line(3) = {4, 3};
//+
Line(4) = {3, 1};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Curve("wall_left", 1) = {4};
//+
Physical Curve("free_top", 2) = {3};
//+
Physical Curve("force_right", 3) = {2};
//+
Physical Curve("free_bottom", 4) = {1};
