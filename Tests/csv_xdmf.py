import pandas as pd
import numpy as np
import h5py


csv_file = "Tests/test_data.csv"
df = pd.read_csv(csv_file, sep=";", skipinitialspace=True)

# print(df.columns)

points = df[['X', 'Y']].values

s_mises = df['S-Mises'].values
s11 = df['S-S11'].values
s22 = df['S-S22'].values
s12 = df['S-S12'].values
print(df.head)

h5_file = "output.h5"
with h5py.File(h5_file, 'w') as f:
    f.create_dataset("coordinates", data=points)
    f.create_dataset("SMises", data=s_mises)
    f.create_dataset("S11", data=s11)
    f.create_dataset("S22", data=s22)
    f.create_dataset("S12", data=s12)


xdmf_file = "output.xdmf"
xdmf_content = f"""<?xml version="1.0" ?>
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="Mesh" GridType="Uniform">
            <Topology TopologyType="Polyvertex" NumberOfElements="{len(points)}"/>
            <Geometry GeometryType="XY">
                <DataItem Format="HDF" DataType="Float" Dimensions="{len(points)} 2">{h5_file}:/coordinates</DataItem>
            </Geometry>
            <Attribute Name="SMises" AttributeType="Scalar" Center="Node">
                <DataItem Format="HDF" DataType="Float" Dimensions="{len(points)}">{h5_file}:/SMises</DataItem>
            </Attribute>
            <Attribute Name="S11" AttributeType="Scalar" Center="Node">
                <DataItem Format="HDF" DataType="Float" Dimensions="{len(points)}">{h5_file}:/S11</DataItem>
            </Attribute>
            <Attribute Name="S22" AttributeType="Scalar" Center="Node">
                <DataItem Format="HDF" DataType="Float" Dimensions="{len(points)}">{h5_file}:/S22</DataItem>
            </Attribute>
            <Attribute Name="S12" AttributeType="Scalar" Center="Node">
                <DataItem Format="HDF" DataType="Float" Dimensions="{len(points)}">{h5_file}:/S12</DataItem>
            </Attribute>
        </Grid>
    </Domain>
</Xdmf>
"""


with open(xdmf_file, "w") as f:
    f.write(xdmf_content)

print("XDMF file saved as output.xdmf")