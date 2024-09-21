import gdspy
import numpy as np

Si = {"layer": 0, "datatype": 0}
SiO2 = {"layer": 1, "datatype": 1}
pixel = 150e-3      # 默认单位是1um 这里设置一个pixel是150nm

# The GDSII file is called a library, which contains multiple cells.
lib = gdspy.GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')
x_size = 50
y_size = 40

# Create the geometry (a single rectangle) and add it to the cell.
# 首先添加中间区域
opt = np.loadtxt('D:/研究生工作/Photonic-Computing/code/OpTorch_9_13/exp_test/dielec_dist_numpy.csv', delimiter=",")
# print(opt.shape, opt)
for i in range(x_size):
    for j in range(y_size):
        if(opt[i, j] == 1):
            rect = gdspy.Rectangle(((30+i)*pixel, (30+j)*pixel), ((31+i)*pixel, (31+j)*pixel), **Si)
            cell.add(rect)
        else:
            rect = gdspy.Rectangle(((30+i)*pixel, (30+j)*pixel), ((31+i)*pixel, (31+j)*pixel), **SiO2)
            cell.add(rect)
# 然后添加输入波导区域
wg1 = gdspy.Rectangle((0*pixel, 33*pixel), (30*pixel, 37*pixel), **Si)
cell.add(wg1)
wg2 = gdspy.Rectangle((0*pixel, 43*pixel), (30*pixel, 47*pixel), **Si)
cell.add(wg2)
wg3 = gdspy.Rectangle((0*pixel, 53*pixel), (30*pixel, 57*pixel), **Si)
cell.add(wg3)
wg4 = gdspy.Rectangle((0*pixel, 63*pixel), (30*pixel, 67*pixel), **Si)
cell.add(wg4)
# 最后添加输出波导区域
op1 = gdspy.Rectangle((80*pixel, 35*pixel), (100*pixel, 37*pixel), **Si)
cell.add(op1)
op2 = gdspy.Rectangle((80*pixel, 51*pixel), (100*pixel, 53*pixel), **Si)
cell.add(op2)
op3 = gdspy.Rectangle((80*pixel, 67*pixel), (100*pixel, 69*pixel), **Si)
cell.add(op3)

# Save the library in a file called 'first.gds'.
lib.write_gds('test.gds')

# Optionally, save an image of the cell as SVG.
cell.write_svg('test.svg')

# Display all cells using the internal viewer.
gdspy.LayoutViewer()

# # Layer/datatype definitions for each step in the fabrication
# ld_fulletch = {"layer": 1, "datatype": 3}
# ld_partetch = {"layer": 2, "datatype": 3}
# ld_liftoff = {"layer": 0, "datatype": 7}

# p1 = gdspy.Rectangle((-3, -3), (3, 3), **ld_fulletch)
# p2 = gdspy.Rectangle((-5, -3), (-3, 3), **ld_partetch)
# p3 = gdspy.Rectangle((5, -3), (3, 3), **ld_partetch)
# p4 = gdspy.Round((0, 0), 2.5, number_of_points=6, **ld_liftoff)