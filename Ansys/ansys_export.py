import csv
import os

# Define output file path
file_name = "hole_offcenter_r10_al.csv"
output_file = os.path.expanduser("~/Documents/Visual Studio Code/Python/Examensarbete/PINNs/Ansys/data/" + file_name)

# Get the current mechanical model
model = ExtAPI.DataModel
solution = model.Project.Model.Analyses[0].Solution

eq_stress = solution.AddEquivalentStress()

normal_x = solution.AddNormalStress()

normal_y = solution.AddNormalStress()	
normal_y.NormalOrientation = NormalOrientationType.YAxis

deform_x = solution.AddDirectionalDeformation()

deform_y = solution.AddDirectionalDeformation()
deform_y.NormalOrientation = NormalOrientationType.YAxis

shear = solution.AddShearStress()

solution.EvaluateAllResults()

nodes = eq_stress.PlotData["Node"]
eq_stress_data = eq_stress.PlotData["Values"]
eq_stress_data = eq_stress.PlotData["Values"]
normal_x_data = normal_x.PlotData["Values"]
normal_y_data = normal_y.PlotData["Values"]
deform_x_data = deform_x.PlotData["Values"]
deform_y_data = deform_y.PlotData["Values"]
shear_data = shear.PlotData["Values"]


# Write results to CSV
with open(output_file, "wb") as file:
    writer = csv.writer(file)
    writer.writerow(["Node ID", "Equivalent Stress", "Stress X", "Stress Y", "Deformation X", "Deformation Y", "Shear"])  # Header
    
    for i in range(0, len(nodes), 2):
        writer.writerow([nodes[i], eq_stress_data[i], normal_x_data[i], normal_y_data[i], deform_x_data[i], deform_y_data[i], shear_data[i]])

