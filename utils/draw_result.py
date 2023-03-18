import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

plot_mean_error = False
colors = ['r', 'g', 'b', 'orange', 'mediumblue', 'purple', 'slateblue']
legend_prop = {'size': 13}

def plot_result(scenario='test', path=''):
    config = {
        "font.family":'Times New Roman',
        "font.size": 18,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
    plt.rcParams.update(config)
    data = np.loadtxt(scenario+'.txt')
    fig1 = plt.figure(figsize=(6, 3.5))
    ax1 = fig1.subplots()
    ax1.grid(linestyle=":")
    ax1.plot(data[:, 0], data[:, 1], label='$\Delta p$ (m)')
    ax1.plot(data[:, 0], data[:, 2], label='$\Delta \phi$ (rad)')
    ax1.plot(data[:, 0], data[:, 3], label='$\Delta v$ (m/s)')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.set_ylim(-0.05, 11)
    # ax1.set_xlabel('Simulation time (s)')
    ax1.set_ylabel('Error')
    ax1.legend(loc='upper right', ncol=3, prop = legend_prop)

    fig2 = plt.figure(figsize=(6, 3.5))
    ax2 = fig2.subplots()
    ax2.grid(linestyle=":")
    ax2.plot(data[:, 0], data[:, 4], label='$a$ (m/s$^2$)')
    ax2.plot(data[:, 0], data[:, 5], label='$\delta$ (rad)')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_ylim(-4, 4.5)
    ax2.set_xlabel('Simulation time (s)')
    ax2.set_ylabel('Control Input')
    ax2.legend(loc='upper right', ncol=2, prop = legend_prop)

    fig3 = plt.figure(figsize=(6, 3.5))
    ax3 = fig3.subplots()
    ax3.grid(linestyle=":")
    V_norm = np.max(data[:, 6])
    if V_norm > 0:
        V = data[:, 6] / V_norm
    else:
        V = data[:, 6]
    NCR_norm = np.max(data[:, 7])
    if NCR_norm > 0:
        NCR = data[:, 7] / NCR_norm
    else:
        NCR = data[:, 7]
    CR_norm = np.max(data[:, 8])
    if CR_norm > 0:
        CR = data[:, 8] / CR_norm
    else:
        CR = data[:, 8]
    T_norm = np.max(data[:, 9])
    if T_norm > 0:
        T = data[:, 9] / T_norm
    else:
        T = data[:, 9]
    
    ax3.plot(data[:, 0], V, label='$F_{V}$')
    ax3.plot(data[:, 0], NCR, label='$F_{NR}$')
    ax3.plot(data[:, 0], CR, label='$F_{CR}$')
    ax3.plot(data[:, 0], T, label='$F_{TL}$')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax3.set_ylim(-0.1, 1.1)
    # ax3.set_xlabel('Simulation time (s)')
    ax3.set_ylabel('Potential Function Value')
    ax3.legend(loc='upper right', ncol=4, prop = legend_prop)
    
    fig1.savefig(os.path.join(path, scenario+'_error.pdf'),
                 bbox_inches='tight', pad_inches=0)
    fig2.savefig(os.path.join(path, scenario+'_input.pdf'),
                 bbox_inches='tight', pad_inches=0)
    fig3.savefig(os.path.join(path, scenario+'_apf.pdf'),
                 bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    senarios = 'roundabout_2'
    plot_result(senarios)
