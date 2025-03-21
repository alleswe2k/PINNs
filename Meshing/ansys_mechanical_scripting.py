'''NOTE : All workflows will not be recorded, as recording is under development.'''


#region Details View Action
body_20 = DataModel.GetObjectById(20)
body_20.Thickness = Quantity(10, "mm")
#endregion


#region UI Action
with Transaction(True):
    body_20 = DataModel.GetObjectById(20)
    body_20.Material = "e09bacca-d562-4e35-8b19-1774114d8ebb"
#endregion

#region Context Menu Action
mesh_15 = Model.Mesh
sizing_32 = mesh_15.AddSizing()
#endregion

#region Details View Action
selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
selection.Ids = [5]
sizing_32.Location = selection
#endregion

#region Details View Action
sizing_32.ElementSize = Quantity(20, "mm")
#endregion

#region Context Menu Action
sizing_34 = mesh_15.AddSizing()
#endregion

#region Details View Action
selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
selection.Ids = [6]
sizing_34.Location = selection
#endregion

#region Details View Action
sizing_34.Type = SizingType.NumberOfDivisions
#endregion

#region Details View Action
sizing_34.NumberOfDivisions = 40
#endregion

#region Details View Action
mesh_15.DisplayStyle = MeshDisplayStyle.ShellThickness
#endregion

#region Details View Action
mesh_15.DisplayStyle = MeshDisplayStyle.ElementQuality
#endregion

#region Context Menu Action
automatic_method_37 = mesh_15.AddAutomaticMethod()
#endregion

#region Details View Action
selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
selection.Ids = [4]
automatic_method_37.Location = selection
#endregion

#region Details View Action
automatic_method_37.FreeFaceMeshType = 2
#endregion

#region Context Menu Action
mesh_15.GenerateMesh()
#endregion

#region Details View Action
mesh_15.DisplayStyle = MeshDisplayStyle.AspectRatio
#endregion

#region Details View Action
mesh_15.DisplayStyle = MeshDisplayStyle.JacobianRatioCornerNodes
#endregion

#region Details View Action
mesh_15.DisplayStyle = MeshDisplayStyle.GeometrySetting
#endregion

#region Context Menu Action
analysis_24 = DataModel.GetObjectById(24)
fixed_support_40 = analysis_24.AddFixedSupport()
#endregion

#region Details View Action
selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
selection.Ids = [7]
fixed_support_40.Location = selection
#endregion

#region Context Menu Action
force_42 = analysis_24.AddForce()
#endregion

#region Details View Action
selection = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
selection.Ids = [9]
force_42.Location = selection
#endregion

#region Details View Action
force_42.DefineBy = LoadDefineBy.Components
#endregion

#region Details View Action
force_42.XComponent.Output.SetDiscreteValue(0, Quantity(500000, "N"))
#endregion

#region Context Menu Action
solution_25 = DataModel.GetObjectById(25)
equivalent_stress_44 = solution_25.AddEquivalentStress()
#endregion

#region Context Menu Action
structural_error_46 = solution_25.AddStructuralError()
#endregion

