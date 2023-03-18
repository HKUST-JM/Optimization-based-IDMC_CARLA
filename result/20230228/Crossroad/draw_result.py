import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import os

plot_mean_error = False

# def plot_result(data):
#     fig = plt.figure(figsize=(30, 6))
#     ax1 = fig.add_subplot(1, 3, 1)
#     ax2 = fig.add_subplot(1, 3, 2)
#     ax3 = fig.add_subplot(1, 3, 3)

#     ax1.plot(data[:, 0], data[:, 1], label='dist_error')
#     ax1.plot(data[:, 0], data[:, 2], label='yaw_error')
#     ax1.legend(loc='upper left')

#     ax2.plot(data[:, 0], data[:, 3], label='acc')
#     ax2.plot(data[:, 0], data[:, 4], label='steer')
#     ax2.plot(data[:, 0], data[:, 5], label='vel')
#     ax2.set_ylim(-4, 7)
#     ax2.legend(loc='upper left')

#     ax3.plot(data[:, 0], data[:, 6], label='MPC time')
#     ax3.legend(loc='upper left')

#     plt.show()


def plot_result(data, scenario='grid_1', path=''):
    
    colors = ['r', 'g', 'b', 'orange', 'mediumblue', 'purple', 'slateblue']

    fig1 = plt.figure(figsize=(7, 3))
    ax1 = fig1.subplots()

    ax1.grid(linestyle=":")
    ax1.plot(data[:, 0], data[:, 1], label='$\Delta p$ (m)')
    ax1.plot(data[:, 0], data[:, 2], label='$\phi$ (rad)')

    if plot_mean_error:
        dist_mean = np.mean(data[:, 1])
        ax1.hlines(dist_mean, data[0, 0], data[-1, 0],
                color=colors[-1], label="dist_mean($m$)")
        yaw_mean = np.mean(data[:, 2])
        ax1.hlines(yaw_mean, data[0, 0], data[-1, 0],
                color='y', label="yaw_mean[$rad$]")
    ax1.set_ylim(-0.05, 0.8)
    ax1.set_xlabel('Simulation time (s)')
    ax1.set_ylabel('Error')
    # ax1.legend(loc='upper left')
    ax1.legend(loc='upper right', ncol=2)

    fig2 = plt.figure(figsize=(7, 3))
    ax2 = fig2.subplots()

    ax2.grid(linestyle=":")
    ax2.plot(data[:, 0], data[:, 3], label='$a$ (m/s$^2$)')
    ax2.plot(data[:, 0], data[:, 4], label='$\delta$ (rad)')
    ax2.plot(data[:, 0], data[:, 5], label='$v$ (m/s)')
    ax2.set_ylim(-8, 15)
    ax2.set_xlabel('Simulation time (s)')
    ax2.set_ylabel('Control Input and Velocity')
    # ax2.legend(loc='upper left')
    ax2.legend(loc='upper right', ncol=3)

    fig3 = plt.figure(figsize=(7, 5))
    ax3 = fig3.subplots()
    ax3.grid(linestyle=":")
    ax3.plot(data[:, 0], data[:, 6], label='')
    if plot_mean_error:
        mpc_mean = np.mean(data[:, 6])
        ax3.hlines(mpc_mean, data[0, 0], data[-1, 0],
                color=colors[2], label="mpc_mean(m)")
    ax3.set_xlabel('Simulation time (s)')
    ax3.set_ylabel('MPC time (s)')
    # ax3.legend(loc='upper left')
    # ax3.legend(loc='upper right')
    plt.grid()
    fig1.savefig(os.path.join(path, scenario+'-1.pdf'),
                 bbox_inches='tight', pad_inches=0)
    fig2.savefig(os.path.join(path, scenario+'-2.pdf'),
                 bbox_inches='tight', pad_inches=0)
    fig3.savefig(os.path.join(path, scenario+'-3.pdf'),
                 bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    config = {
        "font.family":'Times New Roman',
        "font.size": 14,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    plt.rcParams.update(config)
    # plt.rcParams["font.family"] = "Times New Roman"
    # matplotlib.rcParams['font.size'] = 12
    # matplotlib.rcParams['font.family'] = 'Times New Roman'
    data = np.loadtxt('crossroad.txt')
    plot_result(data)
