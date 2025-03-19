# Python Script, API Version = V251

# Set Sketch Plane
selection = DatumPlane1
result = ViewHelper.SetSketchPlane(selection, Info1)
# EndBlock

# Sketch Rectangle
point1 = Point2D.Create(MM(0),MM(0))
point2 = Point2D.Create(MM(1144),MM(0))
point3 = Point2D.Create(MM(1144),MM(738))
result = SketchRectangle.Create(point1, point2, point3)

baseSel = SelectionPoint.Create(CurvePoint1)
targetSel = SelectionPoint.Create(DatumPoint1)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint2)
targetSel = SelectionPoint.Create(DatumPoint1)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint3)
targetSel = SelectionPoint.Create(DatumLine1)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint4)
targetSel = SelectionPoint.Create(DatumLine1)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint5)
targetSel = SelectionPoint.Create(DatumLine2)

result = Constraint.CreateCoincident(baseSel, targetSel)

baseSel = SelectionPoint.Create(CurvePoint6)
targetSel = SelectionPoint.Create(DatumLine2)

result = Constraint.CreateCoincident(baseSel, targetSel)
# EndBlock

# 
# Create Length Dimension
dimTarget = Curve6
alignment = DimensionAlignment.Aligned
result = Dimension.CreateLength(dimTarget, alignment)
# EndBlock

# Edit dimension
selDimension = SketchDimension1
newValue = MM(2000)
result = Dimension.Modify(selDimension, newValue)
# EndBlock

# 
# Create Length Dimension
dimTarget = Curve7
alignment = DimensionAlignment.Aligned
result = Dimension.CreateLength(dimTarget, alignment)
# EndBlock

# Edit dimension
selDimension = SketchDimension2
newValue = MM(1000)
result = Dimension.Modify(selDimension, newValue)
# EndBlock

# Sketch Line
start = Point2D.Create(MM(1000), MM(0))
end = Point2D.Create(MM(1000), MM(1000))
result = SketchLine.Create(start, end)

baseSel = CurvePoint7
targetSel = Curve8
result = Constraint.CreateMidpoint(baseSel, targetSel)

baseSel = SelectionPoint.Create(Curve9)
targetSel = SelectionPoint.Create(Curve8)

result = Constraint.CreatePerpendicular(baseSel, targetSel)

baseSel = CurvePoint8
targetSel = Curve6
result = Constraint.CreateMidpoint(baseSel, targetSel)

baseSel = SelectionPoint.Create(Curve9)
targetSel = SelectionPoint.Create(Curve6)

result = Constraint.CreatePerpendicular(baseSel, targetSel)
# EndBlock

# Sketch Circle
origin = Point2D.Create(MM(1000), MM(500))
result = SketchCircle.Create(origin, MM(100))
# EndBlock

# 
# Create Diameter Dimension
dimTarget = Curve10
result = Dimension.CreateDiameter(dimTarget)
# EndBlock


# Delete Selection
selection = Curve9
result = Delete.Execute(selection)
# EndBlock

# 
selection = Selection.Empty()
secondarySelection = Selection.Empty()
options = FillOptions()
result = Fill.Execute(selection, secondarySelection, options, FillMode.Layout, Info2)
# EndBlock

# Solidify Sketch
mode = InteractionMode.Solid
result = ViewHelper.SetViewMode(mode, Info3)
# EndBlock

# Set Sketch Plane
selection = Face1
result = ViewHelper.SetSketchPlane(selection, Info4)
# EndBlock

# Solidify Sketch
mode = InteractionMode.Solid
result = ViewHelper.SetViewMode(mode, Info5)
# EndBlock

# Create Datum Plane
selection = Face1
result = DatumPlaneCreator.Create(selection, False, Info6)
# EndBlock

# Rotate About X Handle
selection = DatumPlane2
axis = Move.GetAxis(selection, HandleAxis.X)
options = MoveOptions()
result = Move.Rotate(selection, axis, DEG(90), options, Info7)
# EndBlock

# Split Faces
options = SplitFaceOptions()
selection = Face1
cutter = DatumPlane2
result = SplitFace.ByCutter(selection, cutter, options, Info8)
# EndBlock

# Rotate About Y Handle
selection = DatumPlane2
axis = Move.GetAxis(selection, HandleAxis.Y)
options = MoveOptions()
result = Move.Rotate(selection, axis, DEG(90), options, Info9)
# EndBlock

# Split Faces
options = SplitFaceOptions()
selection = FaceSelection.Create(Face2, Face3)
cutter = DatumPlane2
result = SplitFace.ByCutter(selection, cutter, options, Info10)
# EndBlock