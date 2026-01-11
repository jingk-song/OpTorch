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
from concurrent.futures import ProcessPoolExecutor
import time
import multiprocessing
import concurrent.futures
import os
import pickle
import tempfile
import torch
import torch.nn.functional as F


class GradientCalculator(torch.nn.Module):
    def __init__(self, obj, region, device):
        super().__init__()
        self.E_for_base_list = torch.tensor(np.array(obj.E_for_base_list),
                                            dtype=torch.complex64, device=device)
        self.E_fom_base_list = torch.tensor(np.array([np.array(x) for x in obj.E_fom_base_list]),
                                            dtype=torch.complex64, device=device)
        self.E_adj_base_list = torch.tensor(np.array(obj.E_adj_base_list),
                                            dtype=torch.complex64, device=device)
        self.E_fom_desire_mode = torch.tensor(obj.E_fom_desire_mode,
                                              dtype=torch.complex64, device=device)

        self.sim_x_pos = torch.tensor(region.sim_x_pos, dtype=torch.float32, device=device)
        self.sim_y_pos = torch.tensor(region.sim_y_pos, dtype=torch.float32, device=device)
        self.x_pos = torch.tensor(region.x_pos, dtype=torch.float32, device=device)
        self.y_pos = torch.tensor(region.y_pos, dtype=torch.float32, device=device)

        self.register_buffer('target_fom', torch.zeros(len(obj.target_fom), dtype=torch.float32))
        self.register_buffer('forphase', torch.zeros(len(obj.forphase), dtype=torch.complex64))

        self.size_x = region.size_x
        self.size_y = region.size_y
        self.pixel_size = region.pixel_size
        self.x_points = int(self.size_x / self.pixel_size)
        self.y_points = int(self.size_y / self.pixel_size)

    def get_fom(self, E_fom_list):

        T_fwd = torch.zeros(len(self.target_fom), dtype=torch.complex64, device=self.target_fom.device)
        temp_factor = torch.zeros_like(T_fwd)

        all_input_power = torch.sum(torch.abs(self.forphase))
        desire_mode = self.E_fom_desire_mode * all_input_power

        for i in range(len(self.target_fom)):
            input_mode = E_fom_list[i]
            overlap_mode = torch.sum(torch.conj(desire_mode) * input_mode)
            trans_coeff = overlap_mode / torch.sum(torch.conj(desire_mode) * desire_mode)
            T_fwd[i] = trans_coeff * torch.conj(trans_coeff)
            temp_factor[i] = torch.conj(trans_coeff) * 1j

        T_sum = torch.sum(T_fwd)

        fom = -self.target_fom * torch.real(torch.log(T_fwd / T_sum))
        factor = self.target_fom * temp_factor * T_sum / T_fwd

        return fom, factor

    def bilinear_interpolation_2d(self, original_data):
        if original_data.dtype != torch.float32:
            original_data = original_data.float()

        H, W = original_data.shape

        y_new_norm = (self.y_pos - self.sim_y_pos.min()) / (self.sim_y_pos.max() - self.sim_y_pos.min()) * 2 - 1
        x_new_norm = (self.x_pos - self.sim_x_pos.min()) / (self.sim_x_pos.max() - self.sim_x_pos.min()) * 2 - 1

        grid_y, grid_x = torch.meshgrid(y_new_norm, x_new_norm, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).float()  # [1, H_new, W_new, 2]

        original_data = original_data.view(1, 1, H, W)

        interpolated = F.grid_sample(
            original_data,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return interpolated.squeeze()

    def calculate_gradient_3d(self, E_for, E_adj):
        grad_eps_sim = torch.mean(-2 * torch.real(torch.sum(E_for * E_adj, dim=3)), dim=2)

        # if grad_eps_sim.dtype != torch.float32:
        #     grad_eps_sim = grad_eps_sim.float()

        grad_eps = self.bilinear_interpolation_2d(grad_eps_sim)

        return torch.flip(grad_eps.t(), dims=[0])

    def forward(self, sample):
        if sample.dtype != torch.float32:
            sample = sample.float()

        # phases = sample[:len(self.forphase)] * 2 * np.pi
        phases = sample[:len(self.forphase)] * np.pi
        self.forphase.copy_(torch.cos(phases) + 1j * torch.sin(phases))

        ########################
        self.target_fom.copy_(sample[-len(self.target_fom):])
        ##############################

        E_for = torch.tensordot(self.forphase, self.E_for_base_list, dims=([0], [0]))
        E_fom_list = torch.tensordot(self.forphase, self.E_fom_base_list, dims=([0], [0]))
        fom, factor = self.get_fom(E_fom_list)
        E_adj = torch.tensordot(factor, self.E_adj_base_list, dims=([0], [0]))
        grad = self.calculate_gradient_3d(E_for, E_adj)

        return fom, grad


class Object:
    def __init__(self, filename, fom_name, mode_exp_monitor_name,
                 forward_source_name, adjoint_source_name,
                 wavelength, forphase, target_fom):

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
        self.wavelength = wavelength
        self.E_for_base_list = []
        self.E_adj_base_list = []
        self.E_fom_base_list = []
        self.E_fom_desire_mode =[]
        self.simfilepath = None

class Simproj_for:
    def __init__(self, filename, fom_name, mode_exp_monitor_name,
                 forward_source_name, wavelength):
        self.model = []
        self.filename = filename
        self.fom_name = fom_name
        self.mode_exp_monitor_name = mode_exp_monitor_name
        self.forward_source_name = forward_source_name
        self.wavelength = wavelength
        self.E_for_base_list = []
        self.E_fom_base_list = []
        self.simfilepath = None

class Simproj_adj:
    def __init__(self, filename,  adjoint_source_name, wavelength):
        self.model = []
        self.filename = filename
        self.adjoint_source_name = adjoint_source_name
        self.wavelength = wavelength
        self.E_adj_base_list = []
        self.simfilepath = None


class Region:
    def __init__(self):
        self.size_x = 20e-6
        self.size_y = 20e-6
        self.size_z = 220e-9
        self.pixel_size = 100e-9

        self.x_points = int(self.size_x / self.pixel_size)
        self.y_points = int(self.size_y / self.pixel_size)
        self.z_points = int(self.size_z / 110e-9) + 1

        self.dx = self.pixel_size
        self.dy = self.pixel_size


        self.x_sim_pos = np.linspace(-self.size_x / 2, self.size_x / 2, self.x_points + 1)
        self.y_sim_pos = np.linspace(-self.size_y / 2, self.size_y / 2, self.y_points + 1)

        self.x_pos = np.linspace(-self.size_x / 2  + self.dx/2, self.size_x / 2  - self.dx/2, self.x_points)
        self.y_pos = np.linspace(-self.size_y / 2  + self.dy/2, self.size_y / 2  - self.dy/2, self.y_points)


        self.z_pos = np.linspace(-self.size_z / 2, self.size_z / 2, self.z_points)
        self.dz = self.z_pos[1] - self.z_pos[0]

        #sim field params
        self.sim_x_pos = None
        self.sim_y_pos = None


def run_task(func, simmodel, gpu_num, result_dir, task_id):
    result = func(simmodel, gpu_num)
    temp_file = os.path.join(result_dir, f"{task_id}.pkl")
    with open(temp_file, 'wb') as f:
        pickle.dump(result, f)

def build_simproj(simmodel, sub_path, region, index_opt, hide_fdtd=True):
    setting.initialize_simproj(simmodel, sub_path, hide_fdtd)
    setting.refresh_design_region_3d(simmodel, region, index_opt)

def rebuild_simproj(simmodel, sub_path, region, index_opt, hide_fdtd=True):
    simmodel.model.close()
    setting.initialize_simproj(simmodel, sub_path, hide_fdtd)
    setting.refresh_design_region_3d(simmodel, region, index_opt)



if __name__ == "__main__":

    # sim params
    max_epoch = 1000
    all_error = []
    threshold = 0.1
    # quant parameters
    eta = 0.5
    beta = 1.0
    all_beta = []

    region = Region()

    # V_cell = region.dx * region.dy * region.dz

    eps_min = 1.44 ** 2
    eps_max = 3.47 ** 2
    # opt parameters
    params_min = 0
    params_max = 1

    # #filename, path, number, wavelength, target_fom, weight
    ################################
    #create path
    save_path = setting.create_new_sim_directory(cur_path, "phase_100nm_nn_pi")
    sub_path = os.path.join(save_path, 'sim_file')
    os.makedirs(sub_path)
    ##################################

    obj_1 = Object("obj_1_mnist.lsf",
                   ["fom_1", "fom_2", "fom_3", "fom_4", "fom_5",
                            "fom_6", "fom_7", "fom_8", "fom_9", "fom_10"],
                   ["fom_exp_1", "fom_exp_2", "fom_exp_3", "fom_exp_4", "fom_exp_5",
                    "fom_exp_6", "fom_exp_7", "fom_exp_8", "fom_exp_9", "fom_exp_10"],
                   ["forward_source_1", "forward_source_2", "forward_source_3", "forward_source_4", "forward_source_5",
                    "forward_source_6", "forward_source_7", "forward_source_8", "forward_source_9", "forward_source_10"],
                   ["adjoint_source_1","adjoint_source_2", "adjoint_source_3", "adjoint_source_4","adjoint_source_5",
                       "adjoint_source_6","adjoint_source_7", "adjoint_source_8", "adjoint_source_9","adjoint_source_10"],
                    1550e-9, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    num_fom = 0
    num_fom += len(obj_1.fom_name)

    # objects = [obj_1]
    # (self, filename, fom_name, mode_exp_monitor_name,
    #  forward_source_name, wavelength)
    sim_for_1 = Simproj_for("simproj_for_1.lsf", ["fom_1", "fom_2", "fom_3", "fom_4", "fom_5",
                            "fom_6", "fom_7", "fom_8", "fom_9", "fom_10"],
                ["fom_exp_1", "fom_exp_2", "fom_exp_3", "fom_exp_4", "fom_exp_5",
                 "fom_exp_6", "fom_exp_7", "fom_exp_8", "fom_exp_9", "fom_exp_10"],
    ["forward_source_1",  "forward_source_2",  "forward_source_3", "forward_source_4",  "forward_source_5"],
                1550e-9)
    sim_for_2 = Simproj_for("simproj_for_2.lsf", ["fom_1", "fom_2", "fom_3", "fom_4", "fom_5",
                            "fom_6", "fom_7", "fom_8", "fom_9", "fom_10"],
                ["fom_exp_1", "fom_exp_2", "fom_exp_3", "fom_exp_4", "fom_exp_5",
                 "fom_exp_6", "fom_exp_7", "fom_exp_8", "fom_exp_9", "fom_exp_10"],
    ["forward_source_6",  "forward_source_7",  "forward_source_8", "forward_source_9",  "forward_source_10"],
                1550e-9)

    # (self, filename, adjoint_source_name, wavelength)
    sim_adj_1 = Simproj_adj("simproj_adj_1.lsf",
                            ["adjoint_source_1", "adjoint_source_2", "adjoint_source_3","adjoint_source_4",  "adjoint_source_5"],
                            1550e-9)
    sim_adj_2 = Simproj_adj("simproj_adj_2.lsf",
                            ["adjoint_source_6", "adjoint_source_7", "adjoint_source_8","adjoint_source_9",  "adjoint_source_10"],
                            1550e-9)
    # sim_projects = [sim_for_1, sim_for_2, sim_adj_1, sim_adj_2]
    ############################################################################


    params = 1 * np.ones((region.x_points, region.y_points))
    eps_opt = params * (eps_max - eps_min) + eps_min
    index_opt = np.real(np.sqrt(eps_opt))


    # guess = np.load("iris_100nm.npz")
    # eps_opt = guess['eps_save']
    # params = (eps_opt - eps_min) / (eps_max - eps_min)
    # index_opt = np.real(np.sqrt(eps_opt))

    m = np.zeros_like(eps_opt)
    v = np.zeros_like(eps_opt)
    epoch = 0

    data = np.load('train_dataset.npz')
    keys = data.files
    data_length = len(keys)

    setting.initialize_optmodel(obj_1, sub_path, region, index_opt, hide_fdtd=True, gpu=1)

    #####################################
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future1 = executor.submit(build_simproj, sim_for_1, sub_path, region, index_opt, hide_fdtd=True)
        future2 = executor.submit(build_simproj, sim_for_2, sub_path, region, index_opt, hide_fdtd=True)
        future3 = executor.submit(build_simproj, sim_adj_1, sub_path, region, index_opt, hide_fdtd=True)
        future4 = executor.submit(build_simproj, sim_adj_2, sub_path, region, index_opt, hide_fdtd=True)

        concurrent.futures.wait([future1, future2, future3, future4])


    for i in range(1, max_epoch+1):
        print(f"Epoch {i}")

        result_dir = tempfile.mkdtemp()

        tasks = [
            {"func": sim.make_forward_base_sim, "simmodel": sim_for_1, "gpu_num": 4, "task_id": "for1"},
            {"func": sim.make_forward_base_sim, "simmodel": sim_for_2, "gpu_num": 5, "task_id": "for2"},
            {"func": sim.make_adjoint_base_sim, "simmodel": sim_adj_1, "gpu_num": 6, "task_id": "adj1"},
            {"func": sim.make_adjoint_base_sim, "simmodel": sim_adj_2, "gpu_num": 7, "task_id": "adj2"}
        ]

        processes = []
        for task in tasks:
            p = multiprocessing.Process(
                target=run_task,
                args=(task["func"], task["simmodel"], task["gpu_num"], result_dir, task["task_id"])
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for task in tasks:
            task_id = task["task_id"]
            simmodel = task["simmodel"]
            temp_file = os.path.join(result_dir, f"{task_id}.pkl")

            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    result = pickle.load(f)

                    if 'E_for_base_list' in result:
                        simmodel.E_for_base_list = result['E_for_base_list']
                        simmodel.E_fom_base_list = result['E_fom_base_list']
                    elif 'E_adj_base_list' in result:
                        simmodel.E_adj_base_list = result['E_adj_base_list']

                os.remove(temp_file)

        os.rmdir(result_dir)

        obj_1.E_for_base_list = sim_for_1.E_for_base_list + sim_for_2.E_for_base_list
        obj_1.E_fom_base_list = sim_for_1.E_fom_base_list + sim_for_2.E_fom_base_list
        obj_1.E_adj_base_list = sim_adj_1.E_adj_base_list + sim_adj_2.E_adj_base_list

        ######################
        save_sim_data_path = os.path.join(save_path, "sim_data_epoch{}.npz".format(i))
        sim_data = {
            # "E_for_base_list": obj_1.E_for_base_list,
            "E_fom_base_list": obj_1.E_fom_base_list,
            # "E_adj_base_list": obj_1.E_adj_base_list,
            "E_fom_desire_mode": obj_1.E_fom_desire_mode,
            "sim_x_pos": region.sim_x_pos,
            "sim_y_pos": region.sim_y_pos
        }
        np.savez(save_sim_data_path, **sim_data)
        ################################

        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)


        torch.backends.cudnn.benchmark = True

        start_time = time.time()
        model = GradientCalculator(obj_1, region, device).to(device)

        grad_all = torch.zeros(model.x_points, model.y_points, dtype=torch.float32, device=device)
        every_fom = torch.zeros(data_length, len(obj_1.target_fom), dtype=torch.float32, device=device)

        for idx in range(data_length):
            sample = torch.tensor(data[keys[idx]], dtype=torch.float32, device=device)
            fom, grad = model(sample)
            every_fom[idx] = fom
            grad_all += grad

            if (idx + 1) % data_length == 0:
                elapsed = time.time() - start_time
                print(f"Processed {idx + 1}/{data_length} samples，time: {elapsed:.2f}秒")

        grad_all /= data_length

        ################
        every_fom = every_fom.detach().cpu().numpy()
        grad_all = grad_all.detach().cpu().numpy()

        all_fom = np.zeros(len(obj_1.fom_name) + 1)
        all_fom[:-1] = np.sum(every_fom, axis=0) / data_length
        all_fom[-1] = np.sum(every_fom) / data_length

        all_error.append(all_fom.copy())
        all_beta.append(beta)

        with open(save_path + '//fom.txt', 'a') as file:
            file.write(f"Epoch {i}" + str(all_fom.copy()) + '\n')
        print(f"fom={all_fom}")

        if beta < 500:
            beta, convergence_flag = quant.convergence_judgment(all_error, beta, sim_for_1, save_path,
                                                                threshold)

            ##########################
            deps_dparams = quant.d_quant_1bit(params, eps_min, eps_max, eta,
                                              beta)

            grad = grad_all * deps_dparams
            #########################
            m_new, v_new, params_opt = opt.adam_opt(i + epoch, m, v, params_min=params_min, params_max=params_max,
                                                    params=params, grad=grad)
            m = m_new
            v = v_new
            params = params_opt

            eps_opt = quant.quant_1bit(params, eps_min, eps_max, eta, beta)
            index_opt = np.sqrt(eps_opt)

            ################################################
            save_eps_path = os.path.join(save_path, "epoch{}.npz".format(i))
            sim_file = "epoch{}".format(i)
            sim_for_1.model.save(os.path.join(save_path, sim_file))
            np.savez(save_eps_path, eps_save=eps_opt, params=params,
                     beta=beta, m=m, v=v, all_iteration=i + epoch)

        else:
            save_eps_path = os.path.join(sub_path, "epoch{}.npz".format(i))
            np.savez(save_eps_path, eps_save=eps_opt, params=params,
                     beta=beta, m=m, v=v, all_iteration=i + epoch)
            break

        print(f"beta={beta}")
        ###############
        plot.plot_opt(i, all_error, all_beta, save_path, num_fom)
        #############

        ######################################
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

            future1 = executor.submit(rebuild_simproj, sim_for_1, sub_path, region, index_opt, hide_fdtd=True)
            future2 = executor.submit(rebuild_simproj, sim_for_2, sub_path, region, index_opt, hide_fdtd=True)
            future3 = executor.submit(rebuild_simproj, sim_adj_1, sub_path, region, index_opt, hide_fdtd=True)
            future4 = executor.submit(rebuild_simproj, sim_adj_2, sub_path, region, index_opt, hide_fdtd=True)

            concurrent.futures.wait([future1, future2, future3, future4])
        #######################################
