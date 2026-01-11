import os
import sys
import lumapi
import time
sys.path.append(os.path.dirname(__file__))
cur_path = os.path.dirname(os.path.realpath(__file__))

def load_from_lsf(script_file_name):
    """
       Loads the provided scritp as a string and strips out all comments.

       Parameters
       ----------
       :param script_file_name: string specifying a file name.
    """

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
            return new_directory  # 返回新创建的目录路径
        sim_number += 1

def add_rect(obj, region):

    obj.model.eval(
        "addstructuregroup;" +
        "set('name', 'design_pixel');" +
        "set('construction group', 0);" +
        "set('x', 0);" +
        "set('y', 0);" +
        "set('z', 0);" +

        "set('script', \" " +
        "opt_size_x = {};".format(region.size_x) +
        "opt_size_y = {};".format(region.size_y) +
        "opt_size_z = {};".format(region.size_z) +
        "pixel_size = {};".format(region.pixel_size) +
        "begin_x = -opt_size_x / 2 + pixel_size / 2;" +
        "begin_y = opt_size_y / 2 - pixel_size / 2;" +
        "row = opt_size_x / pixel_size;" +
        "col = opt_size_y / pixel_size;" +
        "for (i=1:row){" +
        "for (j=1:col) {" +
        "   addrect;" +
        "set('name', 'pixel_' + num2str(i) + '_' + num2str(j));" +
        "set('x', begin_x + pixel_size * (j - 1));" +
        "set('x span', pixel_size);" +
        "set('y', begin_y - pixel_size * (i - 1));" +
        "set('y span', pixel_size);" +
        "set('z', 0);" +
        "set('z span', opt_size_z);" +
        "set('index', (3.47 - 2.8) * 0.5 + 2.8);" +
        "}" +
        "}" +
        "\");"
    )

    while True:
        time.sleep(1)
        number = obj.model.getnamednumber(f"design_pixel::pixel_{region.x_points}_{region.y_points}")
        # print(f"pixel num:{number}")
        if number == 1:
            # print(f"Design Pixels Initialized successfully")
            break


def initialize_model_addrect(obj, sub_path, region, hide_fdtd, gpu):

    model = lumapi.FDTD(hide=hide_fdtd)
    script = load_from_lsf(obj.filename)
    model.eval(script)
    model.eval('setnamed("FDTD", "express mode", 1);'.format(gpu))
    save_path = os.path.join(sub_path, obj.filename)
    model.save(save_path)
    obj.model = model
    add_rect(obj, region)

def refresh_design_region_3d(obj, region, index_opt):

    commands_str = ""
    for i in range(1, region.y_points+1):
        for j in range(1, region.x_points+1):
            command = f"setnamed('pixel_{i}_{j}','index',{index_opt[i-1, j-1]:.6f});\n"
            commands_str += command

    commands_str += f"addrect; \n set('name', 'flag');"

    obj.model.switchtolayout()

    obj.model.eval(
        "select('design_pixel');" +
        "set('script', \" " +
        "{}".format(commands_str) +
        "\");"
    )

    # time.sleep(2)
    # number = obj.model.getnamednumber("design_pixel::flag")
    # print(f"flag num : {number}")

    while True:
        time.sleep(1)
        number = obj.model.getnamednumber("design_pixel::flag")
        # print(f"flag num : {number}")
        if number == 1:
            obj.model.eval(
                "select('design_pixel');" +
                "set('script', \" " +
                "\");"
            )
            time.sleep(0.1)
            obj.model.select("design_pixel::flag")
            obj.model.delete()
            number = obj.model.getnamednumber("design_pixel::flag")
            print(f"flag num : {number}")
            # print(f"Design pixels updated successfully")
            break



