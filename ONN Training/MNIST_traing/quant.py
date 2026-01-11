import numpy as np
import os

def convergence_judgment(all_fom, beta, simmodel, path, threshold):
    test_length = 10
    #convgance test
    #test length of all_fom
    if len(all_fom) < test_length:
        print("The array does not have enough elements; "
              "it is impossible to make a judgment on convergence.")
        convergence_flag = False
    else:
        last_arrays = all_fom[-test_length:]
        last_elements = [array[-1] for array in last_arrays]
        # if (max(last_elements) - min(last_elements) ) / min(last_elements) < threshold:
        if (max(last_elements) - min(last_elements)) < threshold:
            print("The current optimization has converged.")
            beta *= 1.2
            sim_file = "grey_complete"
            simmodel.model.save(os.path.join(path, sim_file))
            convergence_flag = True
        else:
            print("The current optimization has not converged.")
            convergence_flag = False
    # print(f"beta = {beta}")

    return beta, convergence_flag

def quant_1bit(rho_i, epsilon_min, epsilon_max, eta, beta):

    if beta>500:
        res = np.where((rho_i >= 0.5),epsilon_max,epsilon_min)
    else:
        res = epsilon_min + (np.tanh(beta * eta) + np.tanh(beta * (rho_i - eta))) / (
                np.tanh(beta * eta) + np.tanh(beta * (1 - eta))) * (epsilon_max - epsilon_min)

    return res

def dq_drho(rho_i, epsilon_min, epsilon_max, eta, beta):
    const = 1e-12
    return (beta * (epsilon_max - epsilon_min)) / (
        (np.tanh(beta * eta) + np.tanh(beta * (1 - eta))) * np.cosh(beta * (rho_i - eta)) ** 2 + const
    )

def d_quant_1bit(rho_i, epsilon_min, epsilon_max, eta, beta):

    if beta > 500:
        res = np.where((rho_i >= 0.5),1,0)
    else:
        res = dq_drho(rho_i, epsilon_min, epsilon_max, eta, beta)

    return res