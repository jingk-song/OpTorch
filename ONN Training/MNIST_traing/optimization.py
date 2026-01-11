import numpy as np

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




