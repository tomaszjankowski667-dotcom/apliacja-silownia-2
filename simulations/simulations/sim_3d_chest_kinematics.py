"""
3D ANIMATED KINEMATICS SIMULATOR (sim_3d_animated_kinematics.py)
----------------------------------------------------------------
Siatka 2x4 (8 ćwiczeń na klatkę piersiową - środek).
Prawidłowy kierunek łuków dla bramy i butterfly, chwyt neutralny przy rozpiętkach.
"""

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
    print("BŁĄD: Uruchom skrypt z głównego folderu 'aplikacja siłownia', aby wykryć moduły!")
    exit()

USER_PROFILE = PROFILES.get("Brat (Z_Zdjecia)")
LEVERS = USER_PROFILE.get("levers")
EXERCISE_DATABASE = chest_analyzer.get_exercises_data(USER_PROFILE)

L_HUM = LEVERS.get("L_humerus", 0.326)
L_FOR = LEVERS.get("L_forearm", 0.285)
C_W = LEVERS.get("biacromial_width", 0.41)
C_D = LEVERS.get("chest_block", 0.24)


# --- SILNIK ODWROTNEJ KINEMATYKI (IK) ---
def solve_ik(S, W_target, E_hint):
    max_reach = L_HUM + L_FOR
    dist_SW = np.linalg.norm(W_target - S)

    if dist_SW > max_reach * 0.99:
        W_target = S + (W_target - S) / dist_SW * (max_reach * 0.99)
        dist_SW = np.linalg.norm(W_target - S)

    cos_theta = (L_HUM**2 + dist_SW**2 - L_FOR**2) / (2 * L_HUM * dist_SW)
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


# --- INICJALIZACJA WIZUALNA (SIATKA 2x4 = 8 BOJÓW) ---
fig = plt.figure(figsize=(18, 9))
fig.canvas.manager.set_window_title('Ultimate Biomechanics AI - Wzorce 3D Klatka Piersiowa (8 Bojów)')
fig.patch.set_facecolor('#0f1115')

exercises = [
    {"name": "1. Wyciskanie Sztangi", "db_key": "Flat_Barbell_Press", "type": "bench_barbell"},
    {"name": "2. Wyciskanie Hantli", "db_key": "Flat_Dumbbell_Press", "type": "bench_dumbbell_pronation"},
    {"name": "3. Maszyna Pozioma", "db_key": "Machine_Chest_Press", "type": "seated_machine"},
    {"name": "4. Maszyna Smitha", "db_key": "Flat_Smith_Press", "type": "bench_barbell"},
    {"name": "5. Brama Poziomo", "db_key": "Mid_Cable_Crossover", "type": "standing_cable"},
    {"name": "6. Butterfly / Pec-Deck", "db_key": "Butterfly_Pec_Deck", "type": "seated_butterfly"},
    {"name": "7. Rozpiętki Hantle", "db_key": "Flat_Dumbbell_Flyes", "type": "bench_dumbbell_neutral"},
    {"name": "8. Hex / Squeeze Press", "db_key": "Hex_Squeeze_Press", "type": "bench_dumbbell_neutral"},
]

axes = []
plot_refs = []

for i, ex in enumerate(exercises):
    ax = fig.add_subplot(2, 4, i + 1, projection='3d')
    ax.set_facecolor('#0f1115')
    ax.set_title(ex["name"], color='#f97316', fontweight='bold', fontsize=10, pad=5)

    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_zlim([-0.8, 0.8])
    ax.set_axis_off()

    extype = ex["type"]
    db_key = ex["db_key"]

    if "seated" in extype or "standing" in extype:
        ax.view_init(elev=18, azim=-60)
    else:
        ax.view_init(elev=35, azim=-55)

    traj_R_x, traj_R_y, traj_R_z = [], [], []
    traj_L_x, traj_L_y, traj_L_z = [], [], []
    traj_func = EXERCISE_DATABASE[db_key]["trajectory_func"]

    for t_step in np.linspace(1, 0, 20):
        pt_R = traj_func(t_step, LEVERS, phase="eccentric")
        pt_L = np.array([-pt_R[0], pt_R[1], pt_R[2]])
        traj_R_x.append(pt_R[0]); traj_R_y.append(pt_R[1]); traj_R_z.append(pt_R[2])
        traj_L_x.append(pt_L[0]); traj_L_y.append(pt_L[1]); traj_L_z.append(pt_L[2])

    for t_step in np.linspace(0, 1, 20):
        pt_R = traj_func(t_step, LEVERS, phase="concentric")
        pt_L = np.array([-pt_R[0], pt_R[1], pt_R[2]])
        traj_R_x.append(pt_R[0]); traj_R_y.append(pt_R[1]); traj_R_z.append(pt_R[2])
        traj_L_x.append(pt_L[0]); traj_L_y.append(pt_L[1]); traj_L_z.append(pt_L[2])

    traj_R_x.append(traj_R_x[0]); traj_R_y.append(traj_R_y[0]); traj_R_z.append(traj_R_z[0])
    traj_L_x.append(traj_L_x[0]); traj_L_y.append(traj_L_y[0]); traj_L_z.append(traj_L_z[0])

    ax.plot(traj_R_x, traj_R_y, traj_R_z, color='#f97316', linestyle='--', lw=1.5, alpha=0.85)
    ax.plot(traj_L_x, traj_L_y, traj_L_z, color='#f97316', linestyle='--', lw=1.5, alpha=0.85)

    lines = {
        'spine': ax.plot([0,0], [0,0], [0,0], color='#ffffff', lw=4)[0],
        'shoulders': ax.plot([0,0], [0,0], [0,0], color='#ffffff', lw=4)[0],
        'legs': ax.plot([0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], color='#9ca3af', lw=3)[0],
        'head': ax.plot([0], [0], [0], marker='o', color='#ffffff', markersize=7)[0],
        'chest': ax.plot([0], [0], [0], marker='o', color='#22c55e', markersize=13, alpha=0.4)[0],

        'arm_R': ax.plot([0,0,0], [0,0,0], [0,0,0], color='#38bdf8', marker='o', lw=3, markersize=4)[0],
        'arm_L': ax.plot([0,0,0], [0,0,0], [0,0,0], color='#38bdf8', marker='o', lw=3, markersize=4)[0],

        'eq_1': ax.plot([0,0], [0,0], [0,0], color='#e2e8f0', lw=5)[0],
        'eq_2': ax.plot([0,0], [0,0], [0,0], color='#e2e8f0', lw=5)[0],
        'eq_3': ax.plot([0], [0], [0], marker='o', color='#ef4444', markersize=6)[0],
        'eq_4': ax.plot([0], [0], [0], marker='o', color='#ef4444', markersize=6)[0],
    }

    axes.append(ax)
    plot_refs.append(lines)


# --- GŁÓWNA PĘTLA ANIMACJI ---
def update(frame):
    current_phase = "eccentric" if frame < 100 else "concentric"
    t = (np.cos(frame * np.pi / 100) + 1) / 2

    for i, ex in enumerate(exercises):
        refs = plot_refs[i]
        extype = ex["type"]
        db_key = ex["db_key"]

        if "bench" in extype:
            S_R, S_L = np.array([C_W/2, 0, 0.15]), np.array([-C_W/2, 0, 0.15])
            Hips, Head = np.array([0, -0.4, 0.15]), np.array([0, 0.25, 0.15])
            Knee_R, Foot_R = np.array([0.15, -0.6, 0.2]), np.array([0.2, -0.5, -0.2])
            Knee_L, Foot_L = np.array([-0.15, -0.6, 0.2]), np.array([-0.2, -0.5, -0.2])
        elif "seated" in extype:
            S_R, S_L = np.array([C_W/2, 0, 0.15]), np.array([-C_W/2, 0, 0.15])
            Hips, Head = np.array([0, 0, -0.3]), np.array([0, 0, 0.4])
            Knee_R, Foot_R = np.array([0.15, 0.3, -0.3]), np.array([0.15, 0.3, -0.7])
            Knee_L, Foot_L = np.array([-0.15, 0.3, -0.3]), np.array([-0.15, 0.3, -0.7])
        elif "standing" in extype:
            S_R, S_L = np.array([C_W/2, 0, 0.2]), np.array([-C_W/2, 0, 0.2])
            Hips, Head = np.array([0, -0.1, -0.3]), np.array([0, 0.1, 0.5])
            Knee_R, Foot_R = np.array([0.15, -0.15, -0.5]), np.array([0.15, -0.2, -0.8])
            Knee_L, Foot_L = np.array([-0.15, -0.15, -0.5]), np.array([-0.15, -0.2, -0.8])

        refs['spine'].set_data(np.array([Head[0], Hips[0]]), np.array([Head[1], Hips[1]]))
        refs['spine'].set_3d_properties(np.array([Head[2], Hips[2]]))

        refs['shoulders'].set_data(np.array([S_L[0], S_R[0]]), np.array([S_L[1], S_R[1]]))
        refs['shoulders'].set_3d_properties(np.array([S_L[2], S_R[2]]))

        refs['legs'].set_data(np.array([Foot_L[0], Knee_L[0], Hips[0], Knee_R[0], Foot_R[0]]),
                              np.array([Foot_L[1], Knee_L[1], Hips[1], Knee_R[1], Foot_R[1]]))
        refs['legs'].set_3d_properties(np.array([Foot_L[2], Knee_L[2], Hips[2], Knee_R[2], Foot_R[2]]))

        refs['head'].set_data(np.array([Head[0]]), np.array([Head[1]]))
        refs['head'].set_3d_properties(np.array([Head[2]]))

        chest_pos = (S_R + S_L) / 2 + (np.array([0, 0, C_D/2]) if "bench" in extype else np.array([0, C_D/2, 0]))
        refs['chest'].set_data(np.array([chest_pos[0]]), np.array([chest_pos[1]]))
        refs['chest'].set_3d_properties(np.array([chest_pos[2]]))

        traj_func = EXERCISE_DATABASE[db_key]["trajectory_func"]
        W_target_R = traj_func(t, LEVERS, phase=current_phase)

        if "bench" in extype:
            elbow_y_hint = -0.25 if "Hex" not in db_key else -0.15
            E_hint_R = S_R + np.array([0.25, elbow_y_hint, -0.35])
        elif "seated" in extype:
            E_hint_R = S_R + np.array([0.3, -0.05, -0.15])
        elif "standing" in extype:
            E_hint_R = S_R + np.array([0.35, -0.05, 0.15])

        E_R, W_R = solve_ik(S_R, W_target_R, E_hint_R)
        W_L = np.array([-W_R[0], W_R[1], W_R[2]])
        E_L = np.array([-E_R[0], E_R[1], E_R[2]])

        refs['arm_R'].set_data(np.array([S_R[0], E_R[0], W_R[0]]), np.array([S_R[1], E_R[1], W_R[1]]))
        refs['arm_R'].set_3d_properties(np.array([S_R[2], E_R[2], W_R[2]]))

        refs['arm_L'].set_data(np.array([S_L[0], E_L[0], W_L[0]]), np.array([S_L[1], E_L[1], W_L[1]]))
        refs['arm_L'].set_3d_properties(np.array([S_L[2], E_L[2], W_L[2]]))

        # SPRZĘT
        if "barbell" in extype:
            bar_st, bar_en = W_R + np.array([0.15, 0, 0]), np.array([-W_R[0]-0.15, W_R[1], W_R[2]])
            refs['eq_1'].set_data(np.array([bar_st[0], bar_en[0]]), np.array([bar_st[1], bar_en[1]]))
            refs['eq_1'].set_3d_properties(np.array([bar_st[2], bar_en[2]]))
        elif "bench_dumbbell_pronation" in extype:
            db_R_st, db_R_en = W_R + np.array([0.05, 0, 0]), W_R - np.array([0.05, 0, 0])
            db_L_st, db_L_en = W_L + np.array([0.05, 0, 0]), W_L - np.array([0.05, 0, 0])
            refs['eq_1'].set_data(np.array([db_R_st[0], db_R_en[0]]), np.array([db_R_st[1], db_R_en[1]]))
            refs['eq_1'].set_3d_properties(np.array([db_R_st[2], db_R_en[2]]))
            refs['eq_2'].set_data(np.array([db_L_st[0], db_L_en[0]]), np.array([db_L_st[1], db_L_en[1]]))
            refs['eq_2'].set_3d_properties(np.array([db_L_st[2], db_L_en[2]]))
        elif "bench_dumbbell_neutral" in extype:
            db_R_st, db_R_en = W_R + np.array([0, 0.05, 0]), W_R - np.array([0, 0.05, 0])
            db_L_st, db_L_en = W_L + np.array([0, 0.05, 0]), W_L - np.array([0, 0.05, 0])
            refs['eq_1'].set_data(np.array([db_R_st[0], db_R_en[0]]), np.array([db_R_st[1], db_R_en[1]]))
            refs['eq_1'].set_3d_properties(np.array([db_R_st[2], db_R_en[2]]))
            refs['eq_2'].set_data(np.array([db_L_st[0], db_L_en[0]]), np.array([db_L_st[1], db_L_en[1]]))
            refs['eq_2'].set_3d_properties(np.array([db_L_st[2], db_L_en[2]]))
        elif "seated_butterfly" in extype:
            pivot = np.array([0.25, -0.1, 0.15])
            refs['eq_1'].set_data(np.array([pivot[0], W_R[0]]), np.array([pivot[1], W_R[1]]))
            refs['eq_1'].set_3d_properties(np.array([pivot[2], W_R[2]]))
            refs['eq_2'].set_data(np.array([-pivot[0], W_L[0]]), np.array([pivot[1], W_L[1]]))
            refs['eq_2'].set_3d_properties(np.array([pivot[2], W_L[2]]))
            refs['eq_1'].set_linewidth(3); refs['eq_2'].set_linewidth(3)
        elif "standing_cable" in extype:
            origin_R = np.array([0.7, -0.2, 0.3])
            refs['eq_1'].set_data(np.array([origin_R[0], W_R[0]]), np.array([origin_R[1], W_R[1]]))
            refs['eq_1'].set_3d_properties(np.array([origin_R[2], W_R[2]]))
            refs['eq_2'].set_data(np.array([-origin_R[0], W_L[0]]), np.array([origin_R[1], W_L[1]]))
            refs['eq_2'].set_3d_properties(np.array([origin_R[2], W_L[2]]))
            refs['eq_1'].set_linewidth(1.5); refs['eq_2'].set_linewidth(1.5)

ani = animation.FuncAnimation(fig, update, frames=200, interval=25, blit=False)
plt.tight_layout()
plt.show()