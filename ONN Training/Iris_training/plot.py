import os
import matplotlib.pyplot as plt

def plot_opt(i, objects, all_fom, old_beta, path, num_fom):
    if i % 5 == 0:
        # sim_file = "interation{}".format(i)
        # objects[0].model.save(os.path.join(path, sim_file))

        iterations = range(1, len(all_fom) + 1)

        fig, ax1 = plt.subplots()
        ax1.set_yscale('log')
        for j in range(num_fom):
            ax1.plot(iterations, [sublist[j] for sublist in all_fom], label=f'fom {j}')
        ax1.plot(iterations, [sublist[num_fom] for sublist in all_fom], label='All fom', color='red')

        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.plot(iterations, old_beta, label='Beta', color='blue')

        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Fom', color='red')
        ax2.set_ylabel('Beta', color='blue')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{path}/figure_iteration_{i}.png')
        plt.close(fig)



def digital_plot_opt(i, objects, all_fom, path):

    if i % 20 == 0:
        sim_file = "interation{}".format(i)
        objects[0].model.save(os.path.join(path, sim_file))

    iterations = range(1, len(all_fom) + 1)

    fig, ax1 = plt.subplots()
    ax1.set_yscale('log')
    for j in range(len(objects)):
        ax1.plot(iterations, [sublist[j] for sublist in all_fom], label=f'fom {j}')
    ax1.plot(iterations, [sublist[len(objects)] for sublist in all_fom], label='All fom', color='red')

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Fom', color='red')
    ax1.grid(True)
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{path}/figure_iteration_{i}.png')
    plt.close(fig)
