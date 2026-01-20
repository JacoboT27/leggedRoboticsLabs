import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from footstep_planner import FootstepPlanner

class LIPCOMPlanner:
    """
    Linear Inverted Pendulum (LIP) model for COM trajectory generation.
    """
    
    def __init__(self, param):
        self.z_com = param['h']
        self.g = param['g']
        self.dt = param['world_time_step']
        
        # ---------------------------------------------------------
        # TODO: Calculate the natural frequency (omega) of the LIP model
        # Hint: omega = sqrt(g / z_com)
        self.omega = np.sqrt(self.g/self.z_com)
        # ---------------------------------------------------------

    def interpolate_zmp_trajectory(self, footstep_plan, total_time):
        """
        Creates ZMP trajectory by interpolating through footstep positions.
        """
        t_array = np.arange(0, total_time, self.dt)
        zmp_traj = np.zeros((2, len(t_array)))
        
        t = 0
        for step_idx, step in enumerate(footstep_plan):
            ss_duration = step['ss_duration']
            ds_duration = step['ds_duration']
            
            # Single support phase: ZMP at support foot
            ss_end = t + ss_duration
            ss_indices = np.where((t_array >= t) & (t_array < ss_end))[0]
            if len(ss_indices) > 0:
                zmp_traj[:, ss_indices] = step['pos'][:2, np.newaxis]
            t = ss_end
            
            # Double support phase: Interpolate
            ds_end = t + ds_duration
            if step_idx + 1 < len(footstep_plan):
                next_pos = footstep_plan[step_idx + 1]['pos'][:2]
                curr_pos = step['pos'][:2]
                ds_indices = np.where((t_array >= t) & (t_array < ds_end))[0]
                if len(ds_indices) > 0:
                    alpha = np.linspace(0, 1, len(ds_indices))
                    zmp_traj[:, ds_indices] = (curr_pos[:, None] * (1 - alpha) + next_pos[:, None] * alpha)
            else:
                ds_indices = np.where((t_array >= t) & (t_array < ds_end))[0]
                if len(ds_indices) > 0:
                    zmp_traj[:, ds_indices] = step['pos'][:2, np.newaxis]
            t = ds_end
        
        return t_array, zmp_traj

    def solve_lip_dynamics(self, zmp_traj, t_array, x0_com, v0_com, dcm_ref):
        n_timesteps = zmp_traj.shape[1]
        com_traj = np.zeros((2, n_timesteps))
        com_vel = np.zeros((2, n_timesteps))
        
        com_traj[:, 0] = x0_com
        com_vel[:, 0] = v0_com

        c = np.cosh(self.omega * self.dt)
        s = np.sinh(self.omega * self.dt)
        
        # MPC Gain (K > 1.0 for stability)
        k_gain = 10.0 

        for i in range(n_timesteps - 1):
            x = com_traj[:, i]
            v = com_vel[:, i]
            p_ref = zmp_traj[:, i]
            
            # 1. Current DCM state
            xi_actual = x + v / self.omega
            
            # 2. MPC Law: Adjust ZMP to track DCM reference
            # p_optimal = xi_actual - (1 + k_gain/omega) * (xi_actual - xi_ref)
            # A simpler version for intuitive tuning:
            p_mpc = p_ref + k_gain * (xi_actual - dcm_ref[:, i])
            
            # 3. Foot Constraint (Saturation)
            # Ensure ZMP stays within +/- 5cm of the planned foot center
            #p_final = np.clip(p_mpc, p_ref - 0.05, p_ref + 0.05)
            p_final = p_mpc
            # 4. Analytical Update using the MPC-adjusted ZMP
            com_traj[:, i + 1] = (x - p_final) * c + (v / self.omega) * s + p_final
            com_vel[:, i + 1] = (x - p_final) * self.omega * s + v * c
            
        return com_traj, com_vel, None
    

    def plan_com_trajectory(self, footstep_plan, initial_com_pos):
        # 1. Calculate the stable initial velocity (keep this)
        initial_com_vel = self.compute_initial_velocity(footstep_plan, initial_com_pos)
        
        # 2. Get ZMP trajectory
        total_time = sum(step['ss_duration'] + step['ds_duration'] for step in footstep_plan)
        t_array, zmp_traj = self.interpolate_zmp_trajectory(footstep_plan, total_time)
        
        # 3. Generate the full DCM Reference Trajectory
        dcm_ref = self.generate_dcm_reference(footstep_plan, t_array)
        
        # 4. Solve dynamics (Pass dcm_ref to use in the MPC loop)
        com_pos, com_vel, com_acc = self.solve_lip_dynamics(
            zmp_traj, t_array, initial_com_pos, initial_com_vel, dcm_ref
        )
        
        return {'t': t_array, 'com_pos': com_pos, 'zmp': zmp_traj, 'dcm_ref': dcm_ref}
    
    def generate_dcm_reference(self, footstep_plan, t_array):
        dcm_ref = np.zeros((2, len(t_array)))
        
        # First, find the 'terminal' DCM for each step (backward pass)
        xi_eos = [footstep_plan[-1]['pos'][:2]] # End of Step DCMs
        for step in reversed(footstep_plan[:-1]):
            p = step['pos'][:2]
            T = step['ss_duration'] + step['ds_duration']
            xi_start = p + (xi_eos[0] - p) * np.exp(-self.omega * T)
            xi_eos.insert(0, xi_start)

        # Second, interpolate the DCM forward for each time step
        t_now = 0
        for i, step in enumerate(footstep_plan):
            p = step['pos'][:2]
            T = step['ss_duration'] + step['ds_duration']
            xi_start = xi_eos[i] # This step's required start DCM
            
            indices = np.where((t_array >= t_now) & (t_array < t_now + T))[0]
            for idx in indices:
                relative_t = t_array[idx] - t_now
                # DCM forward dynamics: xi(t) = (xi_0 - p) * exp(omega * t) + p
                dcm_ref[:, idx] = (xi_start - p) * np.exp(self.omega * relative_t) + p
            t_now += T
            
        return dcm_ref
    
    def plan_with_dcm(self, footstep_plan):
        # 1. Reverse pass to find DCM at start of each step
        # Start from the last footstep position
        xi_end = footstep_plan[-1]['pos'][:2]
        dcm_setpoints = [xi_end]

        for step in reversed(footstep_plan[:-1]):
            p = step['pos'][:2]
            T = step['ss_duration'] + step['ds_duration']
            # Backward integration: find xi at start of step
            xi_start = p + (xi_end - p) * np.exp(-self.omega * T)
            dcm_setpoints.insert(0, xi_start)
            xi_end = xi_start

        # 2. Compute Initial CoM Velocity from the first DCM point
        # xi = x + v/w  =>  v = w(xi - x)
        x0 = footstep_plan[0]['pos'][:2]
        v0 = self.omega * (dcm_setpoints[0] - x0)
        
        return v0
    
    def compute_initial_velocity(self, footstep_plan, initial_com_pos):
        """
        Computes the required initial velocity using the DCM backward pass.
        """
        # Start from the last footstep position (where we want to stop)
        xi_end = footstep_plan[-1]['pos'][:2]
        
        # Iterate backwards through steps to find the required DCM at the start
        for step in reversed(footstep_plan[:-1]):
            p = step['pos'][:2]
            T = step['ss_duration'] + step['ds_duration']
            # Backward DCM integration: xi_start = p + (xi_end - p) * exp(-omega * T)
            xi_start = p + (xi_end - p) * np.exp(-self.omega * T)
            xi_end = xi_start

        # Calculate V0 from the first required DCM: v = omega * (xi - x)
        v0 = self.omega * (xi_end - initial_com_pos)
        return v0

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Parameters
    params = {
        'g': 9.81,
        'h': 0.72,
        'world_time_step': 0.01,
        'ss_duration': 0.8,
        'ds_duration': 0.2,
        'first_swing': 'rfoot',
        'foot_size': 0.05,
        # Visualization params
        'vis_foot_width': 0.1, 
        'vis_foot_length': 0.2 
    }
    
    # Initial foot positions [ang_x, ang_y, ang_z, x, y, z]
    initial_lfoot = np.array([0., 0.1, 0., 0., 0.1, 0.]) 
    initial_rfoot = np.array([0., -0.1, 0., 0., -0.1, 0.])
    
    # Velocity reference sequence
    vref = np.array([
        [0.0, 0.0, 0.0],   # Step 1: Start
        [0.0, 0.0, 0.0],   # Step 2
        [0.2, 0.0, 0.0],   # Step 3: Move forward
        [0.2, 0.0, 0.0],   # Step 4
        [0.0, 0.0, 0.0],   # Step 5: Stop
    ])
    
    print("Generating footsteps...")
    planner = FootstepPlanner(vref, initial_lfoot, initial_rfoot, params)
    
    print("\nGenerating COM trajectory using LIP...")
    lip_planner = LIPCOMPlanner(params)
    
    # Start COM between the feet
    initial_com = (initial_lfoot[3:5] + initial_rfoot[3:5]) / 2.0

    traj = lip_planner.plan_com_trajectory(             # added for bonus
        planner.plan, 
        initial_com
    )

    # --- VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plot Footsteps (rectangles)
    for step in planner.plan:
        x, y = step['pos'][0], step['pos'][1]
        # Use simple visualization dimensions
        w = params['vis_foot_length']
        h = params['vis_foot_width']
        
        rect = patches.Rectangle(
            (x - w/2, y - h/2), w, h,
            linewidth=1, edgecolor='r', facecolor='none', label='Footstep'
        )
        ax.add_patch(rect)
        
    # 2. Plot ZMP
    ax.plot(traj['zmp'][0, :], traj['zmp'][1, :], 'g--', label='ZMP Ref', linewidth=2)
    
    # 3. Plot COM
    ax.plot(traj['com_pos'][0, :], traj['com_pos'][1, :], 'b-', label='CoM Trajectory', linewidth=2)
    
    # Fix legend duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_title("LIP Model: Forward Integration Results")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis('equal')
    ax.grid(True)
    plt.show()