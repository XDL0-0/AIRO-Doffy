import numpy as np
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

dt = 0.002
traj_omega = 40
traj_zeta = 1.05
ref_pos = 0.0
ref_vel = 0.0

time_steps = 100
t = np.arange(0, time_steps * dt, dt)

# Staircase target (updates every 5 ticks = 10ms)
target_stair = np.zeros(time_steps)
target_stair[10:15] = 1.0
target_stair[15:20] = 2.0
target_stair[20:25] = 3.0
target_stair[25:] = 4.0

def sim_interp_and_ff(targets):
    ref_pos = 0.0
    ref_vel = 0.0
    ref_acc_log = []
    ref_pos_log = []
    
    prev_raw_target = 0.0
    current_raw_target = 0.0
    target_interp_step = 5
    target = 0.0
    
    for raw_target in targets:
        if raw_target != current_raw_target:
            prev_raw_target = target
            current_raw_target = raw_target
            target_interp_step = 0
            
        target_interp_step += 1
        fraction = min(target_interp_step / 5.0, 1.0)
        target = prev_raw_target + (current_raw_target - prev_raw_target) * fraction
        
        if target_interp_step <= 5:
            target_vel = (current_raw_target - prev_raw_target) / (5.0 * dt)
        else:
            target_vel = 0.0
            
        pos_err_traj = target - ref_pos
        ref_acc = (traj_omega ** 2) * pos_err_traj + 2.0 * traj_zeta * traj_omega * (target_vel - ref_vel)
        ref_vel += ref_acc * dt
        ref_pos += ref_vel * dt
        
        ref_acc_log.append(ref_acc)
        ref_pos_log.append(ref_pos)
        
    return np.array(ref_pos_log), np.array(ref_acc_log)

pos_f, acc_f = sim_interp_and_ff(target_stair)
logger.info(f"Max acc (Feedforward): {np.max(np.abs(acc_f))}")
