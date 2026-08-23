import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('.')

try:
    from user_data import PROFILES
    import analyze.analyze_mid_chest as chest_analyzer
except ImportError:
    exit()

USER_PROFILE = PROFILES.get("Brat (Z_Zdjecia)")
LEVERS = USER_PROFILE.get("levers")
EXERCISE_DATABASE = chest_analyzer.get_exercises_data(USER_PROFILE)

L_HUM = LEVERS.get("L_humerus", 0.326)
L_FOR = LEVERS.get("L_forearm", 0.285)
C_W = LEVERS.get("biacromial_width", 0.41)
C_D = LEVERS.get("chest_block", 0.24)


def solve_ik(S, W_target, E_hint):
    max_reach = L_HUM + L_FOR
    dist_SW = np.linalg.norm(W_target - S)

    if dist_SW > max_reach * 0.99:
        W_target = S + (W_target - S) / dist_SW * (max_reach * 0.99)
        dist_SW = np.linalg.norm(W_target - S)

    cos_theta = (L_HUM ** 2 + dist_SW ** 2 - L_FOR ** 2) / (2 * L_HUM * dist_SW)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    n1 = (W_target - S) / dist_SW
    v_hint = E_hint - S
    v_ortho = v_hint - np.dot(v_hint, n1) * n1

    if np.linalg.norm(v_ortho) < 1e-5:
        v_ortho = np.array([0, 1.0, 0])

    n2 = v_ortho / np.linalg.norm(v_ortho)

    E_dir = n1 * np.cos(theta) + n2 * np.sin(theta)
    E = S + L_HUM * E_dir
    W = E + L_FOR * ((W_target - E) / np.linalg.norm(W_target - E))

    return E, W


fig = plt.figure(figsize=(12, 10))
fig.canvas.manager.set_window_title('ING Pitch - Dumbbell Press Kinematics')
fig.patch.set_facecolor('#ffffff')

ax = fig.add_subplot(111, projection='3d')
ax.set_title("Odwrotna Kinematyka AI - Wyciskanie Hantli", fontweight='bold', pad=20, fontsize=16)

ax.set_xlim([-0.8, 0.8])
ax.set_ylim([-0.8, 0.8])
ax.set_zlim([-0.5, 1.0])
ax.set_axis_off()
ax.view_init(elev=25, azim=-65)

traj_R_x, traj_R_y, traj_R_z = [], [], []
traj_L_x, traj_L_y, traj_L_z = [], [], []

traj_func = EXERCISE_DATABASE["Flat_Dumbbell_Press"]["trajectory_func"]

for t_step in np.linspace(1, 0, 40):
    pt_R = traj_func(t_step, LEVERS, phase="eccentric")
    pt_L = np.array([-pt_R[0], pt_R[1], pt_R[2]])
    traj_R_x.append(pt_R[0]);
    traj_R_y.append(pt_R[1]);
    traj_R_z.append(pt_R[2])
    traj_L_x.append(pt_L[0]);
    traj_L_y.append(pt_L[1]);
    traj_L_z.append(pt_L[2])

for t_step in np.linspace(0, 1, 40):
    pt_R = traj_func(t_step, LEVERS, phase="concentric")
    pt_L = np.array([-pt_R[0], pt_R[1], pt_R[2]])
    traj_R_x.append(pt_R[0]);
    traj_R_y.append(pt_R[1]);
    traj_R_z.append(pt_R[2])
    traj_L_x.append(pt_L[0]);
    traj_L_y.append(pt_L[1]);
    traj_L_z.append(pt_L[2])

traj_R_x.append(traj_R_x[0]);
traj_R_y.append(traj_R_y[0]);
traj_R_z.append(traj_R_z[0])
traj_L_x.append(traj_L_x[0]);
traj_L_y.append(traj_L_y[0]);
traj_L_z.append(traj_L_z[0])

ax.plot(traj_R_x, traj_R_y, traj_R_z, color='#FF9900', linestyle='--', lw=3, alpha=0.8)
ax.plot(traj_L_x, traj_L_y, traj_L_z, color='#FF9900', linestyle='--', lw=3, alpha=0.8)

lines = {
    'bench': ax.plot([-0.2, 0.2, 0.2, -0.2, -0.2], [-0.5, -0.5, 0.3, 0.3, -0.5], [0.05, 0.05, 0.05, 0.05, 0.05],
                     color='#cccccc', lw=15, alpha=0.5)[0],
    'spine': ax.plot([0, 0], [0, 0], [0, 0], 'k-', lw=6)[0],
    'shoulders': ax.plot([0, 0], [0, 0], [0, 0], 'k-', lw=6)[0],
    'legs': ax.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], 'k-', lw=5)[0],
    'head': ax.plot([0], [0], [0], 'ko', markersize=16)[0],
    'chest': ax.plot([0], [0], [0], color='#00aaff', marker='o', markersize=35, alpha=0.2)[0],

    'arm_R': ax.plot([0, 0, 0], [0, 0, 0], [0, 0, 0], color='#2222ff', marker='o', lw=5, markersize=8)[0],
    'arm_L': ax.plot([0, 0, 0], [0, 0, 0], [0, 0, 0], color='#2222ff', marker='o', lw=5, markersize=8)[0],

    'eq_1': ax.plot([0, 0], [0, 0], [0, 0], color='#333333', lw=12)[0],
    'eq_2': ax.plot([0, 0], [0, 0], [0, 0], color='#333333', lw=12)[0],
}


def update(frame):
    current_phase = "eccentric" if frame < 100 else "concentric"
    t = (np.cos(frame * np.pi / 100) + 1) / 2

    S_R, S_L = np.array([C_W / 2, 0, 0.15]), np.array([-C_W / 2, 0, 0.15])
    Hips, Head = np.array([0, -0.4, 0.15]), np.array([0, 0.25, 0.15])
    Knee_R, Foot_R = np.array([0.15, -0.6, -0.1]), np.array([0.2, -0.5, -0.4])
    Knee_L, Foot_L = np.array([-0.15, -0.6, -0.1]), np.array([-0.2, -0.5, -0.4])

    lines['spine'].set_data(np.array([Head[0], Hips[0]]), np.array([Head[1], Hips[1]]))
    lines['spine'].set_3d_properties(np.array([Head[2], Hips[2]]))

    lines['shoulders'].set_data(np.array([S_L[0], S_R[0]]), np.array([S_L[1], S_R[1]]))
    lines['shoulders'].set_3d_properties(np.array([S_L[2], S_R[2]]))

    lines['legs'].set_data(np.array([Foot_L[0], Knee_L[0], Hips[0], Knee_R[0], Foot_R[0]]),
                           np.array([Foot_L[1], Knee_L[1], Hips[1], Knee_R[1], Foot_R[1]]))
    lines['legs'].set_3d_properties(np.array([Foot_L[2], Knee_L[2], Hips[2], Knee_R[2], Foot_R[2]]))

    lines['head'].set_data(np.array([Head[0]]), np.array([Head[1]]))
    lines['head'].set_3d_properties(np.array([Head[2]]))

    chest_pos = (S_R + S_L) / 2 + np.array([0, 0, C_D / 2])
    lines['chest'].set_data(np.array([chest_pos[0]]), np.array([chest_pos[1]]))
    lines['chest'].set_3d_properties(np.array([chest_pos[2]]))

    W_target_R = traj_func(t, LEVERS, phase=current_phase)
    E_hint_R = S_R + np.array([0.2, -0.2, -0.5])

    E_R, W_R = solve_ik(S_R, W_target_R, E_hint_R)
    W_L = np.array([-W_R[0], W_R[1], W_R[2]])
    E_L = np.array([-E_R[0], E_R[1], E_R[2]])

    lines['arm_R'].set_data(np.array([S_R[0], E_R[0], W_R[0]]), np.array([S_R[1], E_R[1], W_R[1]]))
    lines['arm_R'].set_3d_properties(np.array([S_R[2], E_R[2], W_R[2]]))

    lines['arm_L'].set_data(np.array([S_L[0], E_L[0], W_L[0]]), np.array([S_L[1], E_L[1], W_L[1]]))
    lines['arm_L'].set_3d_properties(np.array([S_L[2], E_L[2], W_L[2]]))

    db_R_st, db_R_en = W_R + np.array([0.08, 0, 0]), W_R - np.array([0.08, 0, 0])
    db_L_st, db_L_en = W_L + np.array([0.08, 0, 0]), W_L - np.array([0.08, 0, 0])

    lines['eq_1'].set_data(np.array([db_R_st[0], db_R_en[0]]), np.array([db_R_st[1], db_R_en[1]]))
    lines['eq_1'].set_3d_properties(np.array([db_R_st[2], db_R_en[2]]))

    lines['eq_2'].set_data(np.array([db_L_st[0], db_L_en[0]]), np.array([db_L_st[1], db_L_en[1]]))
    lines['eq_2'].set_3d_properties(np.array([db_L_st[2], db_L_en[2]]))


ani = animation.FuncAnimation(fig, update, frames=200, interval=25, blit=False)
plt.tight_layout()
plt.show()