import numpy as np
import matplotlib.pyplot as plt

def exp_func(x, epsilon):
    return np.sign(x) * (np.abs(x) ** epsilon)

def gen_circle_points(p, r):
    sample_th = np.linspace(0, np.pi*2, 200)
    x = p[0] + r*np.cos(sample_th)
    y = p[1] + r*np.sin(sample_th)
    return np.vstack([x, y])

def gen_elliptic_points(p, r=[30., 30.], eps=0.5):
    sample_th = np.linspace(0, np.pi*2, 200)
    print(r[0]*exp_func(np.cos(sample_th), eps))
    x = r[0]*exp_func(np.cos(sample_th), eps)+p[0]
    y = r[1]*exp_func(np.sin(sample_th), eps)+p[1]
    return np.vstack([x, y])

def gen_round_points(p, line_length, r, d_dist=0.2):
    round_num = int(2*np.pi*r / d_dist)
    half_r_n = int(round_num/2)
    line_num  = int(line_length / d_dist)
    sample_th = np.linspace(-np.pi/2, 3/2*np.pi, round_num)
    r_round_x = p[0] + r*np.cos(sample_th[:half_r_n]) + line_length/2
    cross_line_x = np.linspace(p[0]-line_length/2, p[0]+line_length/2, line_num)
    l_round_x = p[0] + r*np.cos(sample_th[half_r_n:]) - line_length/2
    
    r_round_y = p[1] + r*np.sin(sample_th[:half_r_n])
    upper_cross_line_y = [p[1]+r] * line_num
    l_round_y = p[1] + r*np.sin(sample_th[half_r_n:])
    lower_cross_line_y = [p[1]-r] * line_num

    x = np.hstack([r_round_x, cross_line_x, l_round_x, cross_line_x])
    y = np.hstack([r_round_y, upper_cross_line_y, l_round_y, lower_cross_line_y])

    return np.array([x, y])

p = [0., 0.]

# circles = np.array([gen_circle_points(p, radius) for radius in np.arange(30, 40.5+1.75, 1.75)])
# for i,c in enumerate(circles):
#     if i%2==0:
#         plt.plot(c[0], c[1], c='k', linestyle='-')
#     else:
#         plt.plot(c[0], c[1], c='b', linestyle='--')

# circles = np.array([gen_elliptic_points(p, [radius, radius]) for radius in np.arange(30, 40.5+1.75, 1.75)])
# for c in circles:
#     plt.plot(c[0], c[1])

points = []
for radius in np.arange(30, 40.5+1.75, 1.75):
    points.append(gen_round_points(p, 30, radius))
    # points = np.append(points, gen_round_points(p, 30, radius))


for i,c in enumerate(points):
    if i==0 or i==len(points)-1:
        plt.plot(c[0], c[1], c='k', linestyle='-')
    elif i%2==0:
        plt.plot(c[0], c[1], c='k', linestyle='--')
    else:
        pass
        # plt.plot(c[0], c[1], c='b', linestyle='--')

plt.axis('equal')
plt.show()
