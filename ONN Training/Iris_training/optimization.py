import numpy as np
from scipy.constants import mu_0, epsilon_0


def bilinear_interpolation_2d(original_data, region):

    x_old = region.sim_x_pos
    y_old = region.sim_y_pos
    x_new = region.x_pos
    y_new = region.y_pos

    interpolated_data = np.zeros((len(y_new), len(x_new)))

    for i_y, y in enumerate(y_new):
        y_idx = np.searchsorted(y_old, y) - 1
        y_idx = np.clip(y_idx, 0, len(y_old) - 2)
        ty = (y - y_old[y_idx]) / (y_old[y_idx + 1] - y_old[y_idx])

        for i_x, x in enumerate(x_new):
            x_idx = np.searchsorted(x_old, x) - 1
            x_idx = np.clip(x_idx, 0, len(x_old) - 2)
            tx = (x - x_old[x_idx]) / (x_old[x_idx + 1] - x_old[x_idx])

            interpolated_data[i_y, i_x] = (
                (1 - tx) * (1 - ty) * original_data[y_idx, x_idx] +
                tx * (1 - ty) * original_data[y_idx, x_idx + 1] +
                (1 - tx) * ty * original_data[y_idx + 1, x_idx] +
                tx * ty * original_data[y_idx + 1, x_idx + 1]
            )

    return interpolated_data


def calculate_gradient_2d(obj, V_cell):

    grad_eps = -2 * np.real(np.sum(obj.E_for * obj.E_adj * epsilon_0 * V_cell, axis=2))

    return grad_eps

def calculate_gradient_3d(obj, region):

    grad_eps_sim = np.mean(-2 * np.real(np.sum(obj.E_for * obj.E_adj, axis=3)), axis=2)
    # grad_eps = np.sum(-2 * np.real(np.sum(obj.E_for * obj.E_adj * epsilon_0 * V_cell, axis=3)), axis=2)

    grad_eps = bilinear_interpolation_2d(grad_eps_sim, region)


    grad_eps = np.flip(grad_eps.T, axis=0)

    return grad_eps

def adam_opt(i, m, v, params_min, params_max, params, grad, beta1 = 0.9, beta2 = 0.999, learning_rate = 0.1):

    # adam params
    epsilon = 1e-8

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m_new / (1 - beta1 ** i)
    v_hat = v_new / (1 - beta2 ** i)
    params_opt = np.real(params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))

    params_opt = np.clip(params_opt, params_min, params_max)

    return m_new, v_new, params_opt



