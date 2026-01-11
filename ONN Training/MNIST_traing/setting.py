import os
import numpy as np
import sys
import lumapi
from scipy.interpolate import RegularGridInterpolator

sys.path.append(os.path.dirname(__file__))
cur_path = os.path.dirname(os.path.realpath(__file__))

def load_from_lsf(script_file_name):

    with open(script_file_name, 'r') as text_file:
        lines = [line.strip().split(sep='#', maxsplit=1)[0] for line in text_file.readlines()]
    script = ''.join(lines)
    if not script:
        raise UserWarning('empty script.')
    return script

def create_new_sim_directory(base_path, new_path):

    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Created base directory: {base_path}")

    sim_number = 1

    while True:
        new_directory = os.path.join(base_path, f"{new_path}_{sim_number}")
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
            print(f"New directory created: {new_directory}")
            return new_directory
        sim_number += 1

def bilinear_interpolation_2d(original_data, x_old, y_old, x_new, y_new):
    x_old, y_old = y_old, x_old

    interp_func = RegularGridInterpolator(
        (y_old, x_old),
        original_data,
        method='nearest',
        bounds_error=False,
        fill_value=None
    )

    X_new, Y_new = np.meshgrid(x_new, y_new)
    points = np.column_stack((Y_new.ravel(), X_new.ravel()))

    interpolated_data = interp_func(points).reshape(len(y_new), len(x_new))

    return interpolated_data

def initialize_optmodel(obj, sub_path, region, index_opt, hide_fdtd, gpu):

    obj.model = lumapi.FDTD(hide=hide_fdtd)

    script = load_from_lsf(obj.filename)
    obj.model.eval(script)
    obj.model.eval('setnamed("FDTD", "express mode", 1);'.format(gpu))
    save_path = os.path.join(sub_path, obj.filename)
    obj.model.save(save_path)

    refresh_design_region_3d(obj, region, index_opt)

    obj.model.run()
    obj.E_fom_desire_mode = np.squeeze(obj.model.getresult("fom_exp_1", "mode profiles")["E1"])

    data_E_set = obj.model.getresult("opt_fields", "E")
    region.sim_x_pos = np.squeeze(data_E_set['x'])
    region.sim_y_pos = np.squeeze(data_E_set['y'])

    obj.model.switchtolayout()
    obj.model.close()

def initialize_simproj(sim_obj, sub_path, hide_fdtd):
    sim_obj.model = lumapi.FDTD(hide=hide_fdtd)

    script = load_from_lsf(sim_obj.filename)
    sim_obj.model.eval(script)
    save_path = os.path.join(sub_path, sim_obj.filename)
    sim_obj.model.save(save_path)

def refresh_design_region_3d(obj, region, index_opt):


    x_non_end = region.x_sim_pos[1:-1]
    x_insert_left = x_non_end - 1e-16
    x_insert_right = x_non_end + 1e-16
    x_combined = np.concatenate([region.x_sim_pos, x_insert_left, x_insert_right])
    x_new = np.unique(x_combined)

    y_non_end = region.y_sim_pos[1:-1]
    y_insert_left = y_non_end - 1e-16
    y_insert_right = y_non_end + 1e-16
    y_combined = np.concatenate([region.y_sim_pos, y_insert_left, y_insert_right])
    y_new = np.unique(y_combined)

    index_opt_inter = bilinear_interpolation_2d(index_opt, region.x_pos, region.y_pos, x_new, y_new)

    index_opt_inter = np.rot90(index_opt_inter, k=-1)

    obj.model.select("import")
    obj.model.delete()
    tensor = np.tile(index_opt_inter, (len(region.z_pos), 1, 1))
    tensor = tensor.transpose(1, 2, 0)

    obj.model.addimport()
    obj.model.putv('tensor', tensor)
    obj.model.putv('x', x_new)
    obj.model.putv('y', y_new)
    obj.model.putv('z', region.z_pos)
    obj.model.eval("importnk2(tensor, x, y, z);")