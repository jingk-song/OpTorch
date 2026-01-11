######## IMPORTS ########
import os
import sys

sys.path.append(r"../lumerical/v241/api/python")
# sys.path.append("D:/Program Files/Lumerical/v241/api/python/")
sys.path.append(os.path.dirname(__file__))
cur_path = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import setting
import optimization as opt
import simulation as sim
import plot
import quant


class Object:
    def __init__(self, filename, fom_name, mode_exp_monitor_name,
                 forward_source_name, adjoint_source_name,
                 wavelength, forphase, target_fom, weight, field_size):

        self.model = []
        self.filename = filename
        self.fom_name = fom_name
        self.fom = np.zeros((len(fom_name)))
        self.factor = np.zeros((len(fom_name)), dtype=np.complex64)
        self.mode_exp_monitor_name = mode_exp_monitor_name
        self.forward_source_name = forward_source_name
        self.adjoint_source_name = adjoint_source_name
        self.forphase = forphase
        self.target_fom = target_fom
        self.weight = weight
        self.wavelength = wavelength
        self.E_for = np.zeros((field_size[0], field_size[1],  field_size[2], 3), dtype=np.complex64)
        self.E_adj = np.zeros((field_size[0], field_size[1],  field_size[2], 3), dtype=np.complex64)
        self.simfilepath = None


class Region:
    def __init__(self):


        self.size_x = 6e-6
        self.size_y = 6e-6
        self.size_z = 220e-9
        self.pixel_size = 100e-9

        self.x_points = int(self.size_x / self.pixel_size)
        self.y_points = int(self.size_y / self.pixel_size)
        self.z_points = int(self.size_z / 110e-9) + 1

        self.dx = self.pixel_size
        self.dy = self.pixel_size

        self.x_pos = np.linspace(-self.size_x / 2  + self.dx/2, self.size_x / 2  - self.dx/2, self.x_points)
        self.y_pos = np.linspace(-self.size_y / 2  + self.dy/2, self.size_y / 2  - self.dy/2, self.y_points)


        self.z_pos = np.linspace(-self.size_z / 2, self.size_z / 2, self.z_points)
        self.dz = self.z_pos[1] - self.z_pos[0]

        #sim field params
        self.sim_dx = 40e-9
        self.sim_dy = 40e-9
        self.sim_x_points = int(self.size_x / self.sim_dx) + 1
        self.sim_y_points = int(self.size_y / self.sim_dy) + 1
        self.sim_z_points = self.z_points

        self.sim_x_pos = np.linspace(-self.size_x / 2, self.size_x / 2, self.sim_x_points)
        self.sim_y_pos = np.linspace(-self.size_y / 2, self.size_y / 2, self.sim_y_points)


if __name__ == "__main__":

    # sim params
    batch_size = 12
    max_epoch = 1000
    all_error = []
    threshold = 0.1
    # quant parameters
    eta = 0.5
    beta = 1.0
    all_beta = []

    region = Region()

    field_size = []
    field_size.append(region.sim_x_points)
    field_size.append(region.sim_y_points)
    field_size.append(region.sim_z_points)

    V_cell = region.dx * region.dy * region.dz

    eps_min = 1.44 ** 2
    eps_max = 3.47 ** 2
    # opt parameters
    params_min = 0
    params_max = 1

    # #filename, path, number, wavelength, target_fom, weight
    ################################
    #文件保存路径
    #create path
    save_path = setting.create_new_sim_directory(cur_path, "sim_100nm_fom_mesh")
    sub_path = os.path.join(save_path, 'sim_file')
    os.makedirs(sub_path)
    ##################################

    obj_1 = Object("obj_1_3d_coherent_fom.lsf", ["fom_1", "fom_2", "fom_3"], ["fom_exp_1", "fom_exp_2", "fom_exp_3"],
                   ["forward_source_1", "forward_source_2", "forward_source_3", "forward_source_4"],["adjoint_source_1","adjoint_source_2", "adjoint_source_3"],
                    1550e-9, [0, 0, 0, 0], [1, 1, 1], [1, 1, 1], field_size)
    objects = [obj_1]

    num_fom = 0

    for obj in objects:
        num_fom += len(obj.fom_name)

    #initial model

    for obj in objects:
        setting.initialize_model_addrect(obj, sub_path, region, hide_fdtd=True, gpu=1)

    ############################################################################
    # # initial guess
    eps_initial = 0.5 * np.ones((region.x_points, region.y_points)) * (eps_max - eps_min) + eps_min
    index_opt = np.sqrt(eps_initial)
    eps_opt = eps_initial
    params = (eps_initial - eps_min) / (eps_max - eps_min)
    #定义adam优化算法所需要的中间变量
    m = np.zeros_like(eps_opt)
    v = np.zeros_like(eps_opt)
    iteration = 0
    ##############################################################

    data = np.load('training_data.npz')
    keys = data.files
    data_length = len(keys)
    data_batch_num = int(data_length / batch_size)

    for i in range(1, max_epoch+1):
        print(f"Interation {i}")

        grad_all = np.zeros((region.x_points, region.y_points))
        every_fom = np.zeros((batch_size, len(obj_1.fom_name)))
        all_fom = np.zeros(len(obj_1.fom_name) + 1)

        for obj in objects:
            setting.refresh_design_region_3d(obj, region, index_opt)

        for num_batch in range(batch_size):
            data_index_factor = (i - 1) % data_batch_num
            key = keys[num_batch + batch_size * data_index_factor]
            sample = data[key]

            grad_eps = np.zeros((region.x_points, region.y_points))


            for obj in objects:
                obj.forphase[0] = sample[0] * 360
                obj.forphase[1] = sample[1] * 360
                obj.forphase[2] = sample[2] * 360
                obj.forphase[3] = sample[3] * 360
                obj.target_fom[0] = sample[-3]
                obj.target_fom[1] = sample[-2]
                obj.target_fom[2] = sample[-1]

                sim.make_forward_sim_3d(obj, V_cell, region)
                every_fom[num_batch, :] = np.real(obj.fom)
                ##############################
                sim.make_adjoint_sim_3d(obj, region)
                obj_grad = opt.calculate_gradient_3d(obj, region)
                grad_eps += obj_grad

            grad_all += grad_eps
            grad_all = grad_all / batch_size
            ###########################
            if (num_batch+1) % 4 == 0:
                for obj in objects:
                    # obj.model.eval('exit(1);')
                    obj.model.close()
                    setting.initialize_model_addrect(obj, sub_path, region, hide_fdtd=True, gpu=1)
                    setting.refresh_design_region_3d(obj, region, index_opt)

        ###############
        all_fom[:-1] = np.sum(every_fom, axis=0) / batch_size
        all_fom[-1] = np.sum(every_fom) / batch_size
        all_error.append(all_fom.copy())
        all_beta.append(beta)

        with open(save_path + '//fom.txt', 'a') as file:
            file.write(f"Iteration {i}" + str(all_fom.copy()) + '\n')
        print(f"fom={all_fom}")
        print(f"beta={beta}")
        if beta < 500:
            if ((i > 59 or beta > 1) and (i+iteration) % 10==0):
                if beta > 10:
                    beta = beta * 2
                else:
                    beta = beta * 1.2

            # beta, convergence_flag = quant.convergence_judgment(all_error, beta, objects, save_path,
            #                                                               threshold)

            ##########################
            deps_dparams = quant.d_quant_1bit(params, eps_min, eps_max, eta,
                                                  beta)

            grad = grad_all * deps_dparams
            #########################
            #根据Adam优化算法进行梯度下降更新
            m_new, v_new, params_opt = opt.adam_opt(i+iteration, m, v, params_min=params_min, params_max=params_max,
                                                            params=params, grad=grad)
            m = m_new
            v = v_new
            params = params_opt

            eps_opt = quant.quant_1bit(params, eps_min, eps_max, eta, beta)
            index_opt = np.sqrt(eps_opt)

            #################################################
            if i % 10 == 0:
                setting.refresh_design_region_3d(objects[0], region, index_opt)
                epoch_num = i // 10
                save_eps_path = os.path.join(save_path, f"epoch_{epoch_num}.npz")
                sim_file = f"epoch_{epoch_num}"
                objects[0].model.save(os.path.join(save_path, sim_file))
                np.savez(save_eps_path, eps_save=eps_opt, params=params,
                         beta=beta, m=m, v=v, all_iteration=i + iteration)
            if i % 2 == 0:
                save_eps_path = os.path.join(save_path, "interation{}.npz".format(i))
                sim_file = "interation{}".format(i)
                objects[0].model.save(os.path.join(save_path, sim_file))
                np.savez(save_eps_path, eps_save=eps_opt, params=params,
                         beta=beta, m=m, v=v, all_iteration=i + iteration)

        else:
            save_eps_path = os.path.join(sub_path, "interation{}.npz".format(i))
            np.savez(save_eps_path, eps_save=eps_opt, params=params,
                     beta=beta, m=m, v=v, all_iteration=i + iteration)
            break

        ###############
        plot.plot_opt(i, objects, all_error, all_beta, save_path, num_fom)
        #############