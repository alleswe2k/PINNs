# Python Script, API Version = V251

# Set Sketch Plane
selection = Axis1
result = ViewHelper.SetSketchPlane(selection, Info1)
# EndBlock



# Set Sketch Plane
sectionPlane = Plane.PlaneXY
result = ViewHelper.SetSketchPlane(sectionPlane, Info7)
# EndBlock

# Sketch Rectangle
point1 = Point2D.Create(M(0),M(0))
point2 = Point2D.Create(M(0.531),M(0))
point3 = Point2D.Create(M(0.531),M(0.146))
result = SketchRectangle.Create(point1, point2, point3)

baseSel = SelectionPoint.Create(CurvePoint15)
targetSel = SelectionPoint.Create(DatumLine3)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint16)
targetSel = SelectionPoint.Create(DatumLine3)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint17)
targetSel = SelectionPoint.Create(DatumLine3)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint18)
targetSel = SelectionPoint.Create(DatumLine3)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint15)
targetSel = SelectionPoint.Create(DatumLine4)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint19)
targetSel = SelectionPoint.Create(DatumLine4)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint20)
targetSel = SelectionPoint.Create(DatumLine4)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint18)
targetSel = SelectionPoint.Create(DatumLine4)

result = Constraint.CreateCoincident(baseSel, targetSel)
# EndBlock

# 
# Create Length Dimension
dimTarget = Curve9
alignment = DimensionAlignment.Aligned
result = Dimension.CreateLength(dimTarget, alignment)
# EndBlock

# Edit dimension
selDimension = SketchDimension7
newValue = M(2)
result = Dimension.Modify(selDimension, newValue)
# EndBlock

# 
# Create Length Dimension
dimTarget = Curve10
alignment = DimensionAlignment.Aligned
result = Dimension.CreateLength(dimTarget, alignment)
# EndBlock

# Edit dimension
selDimension = SketchDimension8
newValue = M(1)
result = Dimension.Modify(selDimension, newValue)
# EndBlock

# Sketch Line
start = Point2D.Create(M(1), M(1))
end = Point2D.Create(M(1), M(0))
result = SketchLine.Create(start, end)

baseSel = CurvePoint21
targetSel = Curve9
result = Constraint.CreateMidpoint(baseSel, targetSel)

baseSel = SelectionPoint.Create(Curve11)
targetSel = SelectionPoint.Create(Curve9)

result = Constraint.CreatePerpendicular(baseSel, targetSel)

baseSel = CurvePoint22
targetSel = Curve12
result = Constraint.CreateMidpoint(baseSel, targetSel)

baseSel = SelectionPoint.Create(Curve11)
targetSel = SelectionPoint.Create(Curve12)

result = Constraint.CreatePerpendicular(baseSel, targetSel)
# EndBlock

# Sketch Circle
origin = Point2D.Create(M(1), M(0.5))
result = SketchCircle.Create(origin, M(0.140064270961584))
# EndBlock

# 
# Create Diameter Dimension
dimTarget = Curve13
result = Dimension.CreateDiameter(dimTarget)
# EndBlock

# Edit dimension
selDimension = SketchDimension9
newValue = M(0.1)
result = Dimension.Modify(selDimension, newValue)
# EndBlock

# Delete Selection
selection = Curve11
result = Delete.Execute(selection)
# EndBlock

# Solidify Sketch
mode = InteractionMode.Solid
result = ViewHelper.SetViewMode(mode, Info8)
# EndBlock