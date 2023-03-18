import copy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D


def nc(var, roads=[-4.5, 4.5], coef=10000, slop_coef=0.1, max_n=1000):
    cost = None
    for road in roads:
        if road >= 0:
            if cost is None:
                cost = np.clip(coef**((var-road)/slop_coef), None, max_n)
            else:
                cost += np.clip(coef**((var-road)/slop_coef), None, max_n)
        else:
            if cost is None:
                cost = np.clip(coef**((road-var)/slop_coef), None, max_n)
            else:
                cost += np.clip(coef**((road-var)/slop_coef), None, max_n)
    
    return cost

def c(var, roads=[-4.5, -1.5, 1.5, 4.5], coef=80):
    cost = None
    for road in roads:
        if cost is None:
            cost = coef*np.exp(-(var-road)**2/2)/np.sqrt(2*np.pi)
        else:
            cost += coef*np.exp(-(var-road)**2/2)/np.sqrt(2*np.pi)
    
    return cost

# valid
def nc2(var, roads=[5.25, -5.25], coef=2):
    cost = np.zeros(var.shape)

    for road in roads:
        index = np.where(np.abs(var-road) <= 1.0)
        i_cost = coef / np.abs(var[index]-road)**2 - coef
        cost[index] += i_cost
        if road > 0:
            exclude_index = np.where(var > road-0.1)
            cost[exclude_index] = 200
        else:
            exclude_index = np.where(var < road+0.1)
            cost[exclude_index] = 200
    return cost

# valid
def c2(var, roads=[-1.75, 1.75], coef=20):
    cost = 0
    for road in roads:
        index = np.where(np.abs(var-road) > 1.0)
        i_cost = coef*(np.abs(var-road)-1)**2
        i_cost[index] = 0
        cost += i_cost

    return cost

def road_APF():
    xx = np.arange(0, 7.9, 0.2)
    yy = np.arange(-6.0, 6.0+0.005, 0.005)

    X, Y = np.meshgrid(xx, yy)
    Z_nc = nc2(Y)
    Z_c = c2(Y)
    Z_nc = Z_nc
    cost =  Z_c + Z_nc
    np.clip(cost, 0, 200, out=cost)
    Z = cost
    ax = plt.axes(projection='3d')  
    ax.set_xlabel('Lateral Distance (m)') #fontdict={'family' : 'Times New Roman', 'size'   : 12})
    ax.set_ylabel('Longitudinal Distance (m)') # fontdict={'family' : 'Times New Roman', 'size'   : 12})
    # ax.set_zlabel('Cost', fontdict={'family' : 'Times New Roman', 'size'   : 12})
    ax.invert_xaxis()
    # ax.set_zlim(0, 100)
    # ax.zaxis.set_major_locator(z_major_locator)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.7, 1]))
    ax.plot_surface(Y, X, Z, rstride=1,cstride=1, cmap='jet')
    ax.view_init(elev=15, azim=-70)
    plt.tight_layout()
    # plt.tick_params(labelsize=10)
    plt.savefig('APF_road.pdf', bbox_inches='tight')
    plt.show()


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def road_APF_heat():
    '''
    Get the heat map of the road APF
    '''
    xx = np.arange(0, 5, 0.25)
    yy = np.arange(-7, 7+0.005, 0.005)

    ###################
    ax = plt.subplot(1, 1, 1)

    cmap0 = 'jet'
    X, Y = np.meshgrid(xx, yy)
    Z_nc = nc2(Y, coef=1)
    
    Z_c = c2(Y, coef=1)
    Z = Z_c + Z_nc
    np.clip(Z, 0, 50, out=Z)
    # Z = normalization(Z)
    print(np.max(Z), np.min(Z))
    
    plt.pcolormesh(X, Y, Z, cmap=cmap0, shading='gouraud')
    plt.tight_layout()
    plt.show()

def vehicle_APF():
    xx = np.arange(0, 7.9, 0.05)
    yy = np.arange(-4, 4+0.02, 0.02)
    X, Y = np.meshgrid(xx, yy)
    cost = np.zeros(X.shape)
    coef = 5

    center_points = [[1.7, 0], [4.2, 0]]
    for point in center_points:
        cost += coef/((point[0]-X)**2 + (point[1]-Y)**2)
    np.clip(cost, None, 5, out=cost)

    Z = cost
    ax = plt.axes(projection='3d')  
    ax.set_xlabel('Lateral Distance (m)') #fontdict={'family' : 'Times New Roman', 'size'   : 12})
    ax.set_ylabel('Longitudinal Distance (m)') # fontdict={'family' : 'Times New Roman', 'size'   : 12})
    # ax.set_zlabel('Cost', fontdict={'family' : 'Times New Roman', 'size'   : 12})
    ax.invert_xaxis()
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.7, 1]))
    ax.plot_surface(Y, X, Z, rstride=1,cstride=1, cmap='jet')
    ax.view_init(elev=15, azim=-70)
    # plt.tick_params(labelsize=10)
    plt.tight_layout()

    plt.savefig('APF_vehicle.pdf', bbox_inches='tight')
    plt.show()

def traffic_APF():
    xx = np.arange(0, 7.9, 0.01)
    yy = np.arange(-1.5, 1.5+0.02, 0.02)
    X, Y = np.meshgrid(xx, yy)
    cost = np.zeros(X.shape)
    light_pos = 7.9

    cost_f = 20*(1/-(X - light_pos))
    cost_l = 20*(1/(- Y + 1.8))**2
    cost_r = 20*(1/(Y + 1.8))**2
    cost = cost_l+ cost_f+cost_r
    np.clip(cost, 0, 200, out=cost)
    Z = cost
    ax = plt.axes(projection='3d')  
    ax.set_xlabel('Lateral Distance (m)') #fontdict={'family' : 'Times New Roman', 'size'   : 12})
    ax.set_ylabel('Longitudinal Distance (m)') # fontdict={'family' : 'Times New Roman', 'size'   : 12})
    # ax.set_zlabel('Cost', fontdict={'family' : 'Times New Roman', 'size'   : 12})
    ax.invert_xaxis()
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.7, 1]))
    ax.plot_surface(Y, X, Z, rstride=1,cstride=1, cmap='jet')
    ax.view_init(elev=15, azim=-70)
    plt.tight_layout()
    plt.savefig('APF_traffic.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()


config = {
        "font.family":'Times New Roman',
        "font.size": 10,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
    }
plt.rcParams.update(config)
vehicle_APF()
