import numpy as np
import matplotlib.pyplot as plt

def make_forward_base_sim(simmodel, gpu_num):

    E_for_base_list = []

    rows = len(simmodel.forward_source_name)
    cols = len(simmodel.fom_name)
    E_fom_base_list = [[0 for _ in range(cols)] for _ in range(rows)]


    for i in range(len(simmodel.forward_source_name)):
        simmodel.model.switchtolayout()
        for j in range(len(simmodel.forward_source_name)):
            simmodel.model.select(simmodel.forward_source_name[j])
            simmodel.model.set("Enabled", False)

        simmodel.model.select(simmodel.forward_source_name[i])
        simmodel.model.set("Enabled", True)

        simmodel.model.save(simmodel.filename)
        simmodel.model.run("FDTD", "GPU", gpu_num)


        Ex = simmodel.model.getresult("opt_fields", "Ex")
        Ey = simmodel.model.getresult("opt_fields", "Ey")
        Ez = simmodel.model.getresult("opt_fields", "Ez")

        print(f"sim_proj: {gpu_num-3} sim_num: {i} complete")

        E_stack = np.stack([np.squeeze(Ex), np.squeeze(Ey), np.squeeze(Ez)], axis=3)

        E_for_base_list.append(E_stack)

        #E_for_base_list = np.array(E_for_base_list)

        # data_E_set = simmodel.model.getresult("opt_fields", "E")
        # region.sim_x_pos = np.squeeze(data_E_set['x'])
        # region.sim_y_pos = np.squeeze(data_E_set['y'])

        for k in range(len(simmodel.fom_name)):
            E_dict = simmodel.model.getresult(simmodel.fom_name[k], "E")
            E_fom_base_list[i][k] = np.squeeze(E_dict['E'])

    return {
        'E_for_base_list': E_for_base_list,
        'E_fom_base_list': E_fom_base_list
    }


def make_adjoint_base_sim(simmodel, gpu_num):

    E_adj_base_list = []

    simmodel.model.switchtolayout()
    for i in range(len(simmodel.adjoint_source_name)):
        simmodel.model.switchtolayout()
        for j in range(len(simmodel.adjoint_source_name)):
            simmodel.model.select(simmodel.adjoint_source_name[j])
            simmodel.model.set("Enabled", False)

        simmodel.model.select(simmodel.adjoint_source_name[i])
        simmodel.model.set("Enabled", True)

        simmodel.model.save(simmodel.filename)
        simmodel.model.run("FDTD", "GPU", gpu_num)

        Ex = simmodel.model.getresult("opt_fields", "Ex")
        Ey = simmodel.model.getresult("opt_fields", "Ey")
        Ez = simmodel.model.getresult("opt_fields", "Ez")

        print(f"sim_proj: {gpu_num - 3} sim_num: {i} complete")

        E_stack = np.stack([np.squeeze(Ex), np.squeeze(Ey), np.squeeze(Ez)], axis=3)

        E_adj_base_list.append(E_stack)

        # simmodel.E_adj_base_list = np.array(simmodel.E_adj_base_list)
    return {
        'E_adj_base_list': E_adj_base_list
    }



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
