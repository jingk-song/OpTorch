
import scipy.constants as const
from scipy.constants import epsilon_0

import numpy as np

def get_source_power(obj, wavelengths):
    frequency = const.c / wavelengths
    source_power = obj.model.sourcepower(frequency)
    return np.asarray(source_power).flatten()


def cross_entropy_loss(y_true, y_pred):

    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / N

    return loss


def get_fom(obj, V_cell):

    T_fwd = np.zeros(len(obj.fom_name), dtype=np.complex128)
    temp_factor = np.zeros(len(obj.fom_name), dtype=np.complex128)
    for i in range(len(obj.fom_name)):
        mode_exp_result_name = 'expansion for ' + obj.mode_exp_monitor_name[i]
        mode_exp_data_set = obj.model.getresult(obj.mode_exp_monitor_name[i], mode_exp_result_name)
        wavelengths = mode_exp_data_set['lambda'].flatten()
        trans_coeff = (mode_exp_data_set['a'] * np.sqrt(mode_exp_data_set['N'].real)).flatten()

        omega = 2.0 * np.pi * const.c / wavelengths
        adjoint_source_power = get_source_power(obj, wavelengths)
        source_power = get_source_power(obj, wavelengths)
        phase_prefactors = trans_coeff / 4.0 / source_power
        T_fwd[i] = np.real(trans_coeff * trans_coeff.conj() / source_power)
        temp_factor[i] = np.conj(phase_prefactors) * omega * 1j / np.sqrt(adjoint_source_power)
    T_sum = np.sum(T_fwd)

    for i in range(len(obj.fom_name)):
        obj.fom[i] = -obj.target_fom[i] * np.real(np.log((T_fwd[i]) / T_sum))
        obj.factor[i] = (obj.target_fom[i] * temp_factor[i] * T_sum / T_fwd[i]) * V_cell * epsilon_0

def make_forward_sim_3d(obj, V_cell, region):
    #sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.adjoint_source_name)):
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", False)
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", True)
        obj.model.set("phase", np.abs(obj.forphase[i]))

    obj.model.save(obj.filename)
    obj.model.run()

    #get fom
    get_fom(obj, V_cell)

    #get forward fields
    forward_Ex = obj.model.getresult("opt_fields", "Ex")
    forward_Ey = obj.model.getresult("opt_fields", "Ey")
    forward_Ez = obj.model.getresult("opt_fields", "Ez")

    obj.E_for = np.stack([np.squeeze(forward_Ex), np.squeeze(forward_Ey), np.squeeze(forward_Ez)], axis=3)



def make_adjoint_sim_3d(obj, region):
    ###sim setting
    obj.model.switchtolayout()
    for i in range(len(obj.forward_source_name)):
        obj.model.select(obj.forward_source_name[i])
        obj.model.set("Enabled", False)
    for i in range(len(obj.adjoint_source_name)):
        phase_radians = np.angle(obj.factor[i] * obj.weight[i])
        adj_phase = np.degrees(phase_radians)
        adj_amp = np.abs(obj.factor[i] * obj.weight[i])
        obj.model.select(obj.adjoint_source_name[i])
        obj.model.set("Enabled", True)
        obj.model.set("phase", adj_phase)
        obj.model.set("amplitude", adj_amp)
        obj.model.set('center wavelength', obj.wavelength)
        obj.model.set('wavelength span', 0)

    obj.model.run()

    #get adjoint fields
    adjoint_Ex = obj.model.getresult("opt_fields", "Ex")
    adjoint_Ey = obj.model.getresult("opt_fields", "Ey")
    adjoint_Ez = obj.model.getresult("opt_fields", "Ez")


    obj.E_adj = np.stack([np.squeeze(adjoint_Ex), np.squeeze(adjoint_Ey), np.squeeze(adjoint_Ez)], axis=3)

def fomdata_process(objects, all_error, all_beta, beta, save_path, i):
    ###############
    every_fom = [item for obj in objects for item in obj.fom]
    every_fom.append(np.sum(every_fom))
    every_fom = np.array(every_fom)
    all_error.append(every_fom.copy())
    all_beta.append(beta)
    with open(save_path + '//fom.txt', 'a') as file:
        every_fom_str = ','.join(map(str, every_fom))
        file.write(f"Iteration {i}\t{every_fom_str}\t{beta}\n")
    print(f"fom= {every_fom}")

    return every_fom
