import numpy as np
import dartpy as dart
import copy
from utils import *
import os
import ismpc
import footstep_planner
import inverse_dynamics as id
import filter
import foot_trajectory_generator as ftg
from logger import Logger




def plot_controlled_results(node, label="run", save_dir=None):
    """
    FIGURE 1:
        XY plane: Footsteps (red rectangles for EXECUTED steps only),
        ZMP reference (green dashed),
        CoM trajectory (blue)

    FIGURE 2:
        CoM, DCM, Contact vs time (x-direction)

    Windows stay open until you press Enter.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    log = node.logger.log

    # -----------------------------
    # Extract data
    # -----------------------------
    com = np.array(log[('current', 'com', 'pos')])
    com_vel = np.array(log[('current', 'com', 'vel')])

    have_zmp_ref = (('desired', 'zmp', 'pos') in log)
    zmp_ref = np.array(log[('desired', 'zmp', 'pos')]) if have_zmp_ref else None

    lfoot = np.array(log[('current', 'lfoot', 'pos')])   # (T,6): [rotvec(3), pos(3)]
    rfoot = np.array(log[('current', 'rfoot', 'pos')])

    T = com.shape[0]
    dt = float(node.params["world_time_step"])
    t = np.arange(T) * dt

    foot_size = float(node.params["foot_size"])
    half = foot_size / 2.0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # =====================================================
    # FIGURE 1 — XY CONTROLLED RESULTS (EXECUTED FOOTSTEPS)
    # =====================================================
    fig1, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    # CoM XY trajectory
    ax.plot(com[:, 0], com[:, 1], linewidth=2, label="CoM Trajectory")

    # ZMP reference XY
    if zmp_ref is not None:
        ax.plot(zmp_ref[:, 0], zmp_ref[:, 1], linestyle="--", linewidth=2, label="ZMP Ref")

    # --- Determine how many steps were actually reached during the run ---
    executed_steps = []
    if hasattr(node, "footstep_planner") and hasattr(node.footstep_planner, "get_step_index_at_time"):
        step_idxs = []
        for k in range(T):
            idx = node.footstep_planner.get_step_index_at_time(k)
            if idx is not None:
                step_idxs.append(int(idx))
        if len(step_idxs) > 0:
            max_step_idx = max(step_idxs)
            # Only plot plan entries up to what was reached
            if hasattr(node.footstep_planner, "plan"):
                executed_steps = node.footstep_planner.plan[:max_step_idx + 1]

    # --- Always draw the initial actual feet (from logs) so plot matches reality ---
    # These are the feet positions at time 0 (what the robot actually starts with).
    lx0, ly0 = float(lfoot[0, 3]), float(lfoot[0, 4])
    rx0, ry0 = float(rfoot[0, 3]), float(rfoot[0, 4])

    # One labeled patch for legend, one unlabeled to avoid duplicate legend spam
    ax.add_patch(Rectangle((lx0 - half, ly0 - half), foot_size, foot_size,
                           fill=False, edgecolor="red", linewidth=2, label="Footstep"))
    ax.add_patch(Rectangle((rx0 - half, ry0 - half), foot_size, foot_size,
                           fill=False, edgecolor="red", linewidth=2))

    # --- Draw only EXECUTED planned footsteps (if any) ---
    # If the robot collapses before stepping, this will be empty and you'll only see the initial feet.
    if len(executed_steps) > 0:
        for step in executed_steps:
            px, py = float(step["pos"][0]), float(step["pos"][1])
            ax.add_patch(Rectangle((px - half, py - half), foot_size, foot_size,
                                   fill=False, edgecolor="red", linewidth=2))

    ax.set_title("LIP Model: Controlled Results (XY Plane)", fontsize=18)
    ax.set_xlabel("X (m)", fontsize=15)
    ax.set_ylabel("Y (m)", fontsize=15)
    ax.grid(True)
    ax.axis("equal")
    ax.tick_params(labelsize=13)
    ax.legend(loc="best", fontsize=12)

    if save_dir is not None:
        fig1.savefig(os.path.join(save_dir, "controlled_results_xy.png"), dpi=200)

    # =====================================================
    # FIGURE 2 — CoM, DCM, Contact vs Time (X direction)
    # =====================================================
    eta = float(node.params["eta"])   # sqrt(g/h)
    com_x = com[:, 0]
    dcm_x = com_x + com_vel[:, 0] / eta

    lfoot_x = lfoot[:, 3]
    rfoot_x = rfoot[:, 3]
    contact_x = np.zeros(T)

    if hasattr(node, "footstep_planner"):
        for k in range(T):
            step_idx = node.footstep_planner.get_step_index_at_time(k)
            if step_idx is not None:
                phase = node.footstep_planner.get_phase_at_time(k)
                if phase == "ss" and hasattr(node.footstep_planner, "plan"):
                    step_idx = int(step_idx)
                    # clamp index just in case
                    step_idx = max(0, min(step_idx, len(node.footstep_planner.plan) - 1))
                    support = node.footstep_planner.plan[step_idx]["foot_id"]
                    contact_x[k] = lfoot_x[k] if support == "lfoot" else rfoot_x[k]
                else:
                    contact_x[k] = 0.5 * (lfoot_x[k] + rfoot_x[k])
            else:
                contact_x[k] = 0.5 * (lfoot_x[k] + rfoot_x[k])
    else:
        contact_x = 0.5 * (lfoot_x + rfoot_x)

    fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax2.plot(t, com_x, linewidth=2, label="CoM")
    ax2.plot(t, dcm_x, linewidth=2, label="DCM")
    ax2.plot(t, contact_x, linewidth=2, label="Contact")

    ax2.set_title("CoM, DCM, and Contact Point vs. Time", fontsize=18)
    ax2.set_xlabel("Time (s)", fontsize=15)
    ax2.set_ylabel("X Position (m)", fontsize=15)
    ax2.grid(True)
    ax2.tick_params(labelsize=13)
    ax2.legend(loc="best", fontsize=12)

    if save_dir is not None:
        fig2.savefig(os.path.join(save_dir, "dcm_contact_vs_time.png"), dpi=200)

    # -----------------------------
    # Keep plots open
    # -----------------------------
    plt.show(block=True)




class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.time = 0
        self.params = {
            'g': 9.81,
            'h': 0.26,            # This will be dynamically overwritten below
            'foot_size': 0.06,    # 0.10 gives the MPC sliding box the 1cm grease it needs
            'step_height': 0.02,  # 2cm clearance for flat ground [cite: 23]
            'ss_duration': 40,
            'ds_duration': 60,
            'world_time_step': world.getTimeStep(),
            'first_swing': 'rfoot',
            'µ': 0.5,
            'N': 220,
            'dof': self.hrp4.getNumDofs(),
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])

        # Updated link names for Nao
        self.lsole = hrp4.getBodyNode('l_sole') 
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('base_link') 

        # Updated redundant DOFs (arms and head)
        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # set floating base to passive, everything else to torque
            if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
            elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # YOU MUST REVERT TO THIS DEEP CROUCH
        initial_configuration = {
            'HeadYaw': 0., 'HeadPitch': 0.,
            'LHipYawPitch': 0., 'LHipRoll': 3., 'LHipPitch': -25., 'LKneePitch': 50., 'LAnklePitch': -25., 'LAnkleRoll': -3.,
            'RHipYawPitch': 0., 'RHipRoll': -3., 'RHipPitch': -25., 'RKneePitch': 50., 'RAnklePitch': -25., 'RAnkleRoll': 3.,
            # Arms down by the side is perfectly fine to keep!
            'LShoulderPitch': 80., 'LShoulderRoll': 8., 'LElbowYaw': 0., 'LElbowRoll': -25.,
            'RShoulderPitch': 80., 'RShoulderRoll': -8., 'RElbowYaw': 0., 'RElbowRoll': -25.
        }
        for joint_name, value in initial_configuration.items():
            self.hrp4.setPosition(self.hrp4.getDof(joint_name).getIndexInSkeleton(), value * np.pi / 180.)
        
        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(3, - (lsole_pos[0] + rsole_pos[0]) / 2.)
        self.hrp4.setPosition(4, - (lsole_pos[1] + rsole_pos[1]) / 2.)
        self.hrp4.setPosition(5, - (lsole_pos[2] + rsole_pos[2]) / 2.)

        # initialize state
        self.initial = self.retrieve_state()
        
        # --- THE DYNAMIC HEIGHT FIX ---
        self.params['h'] = self.initial['com']['pos'][2]
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
        print(f"TRUE LOCKED HEIGHT: {self.params['h']}")
        # ------------------------------

        self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [
            "HeadYaw", "HeadPitch", # Indices 6 and 7
            "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw",
            "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw",
            "LFinger11", "LFinger12", "LFinger13", "LFinger21", "LFinger22", "LFinger23",
            "RFinger11", "RFinger12", "RFinger13", "RFinger21", "RFinger22", "RFinger23"
        ]

        # initialize inverse AND PASS THE FOOT SIZE
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs, foot_size=self.params['foot_size'])

        # initialize footstep planner
        # Step forward 4cm at a time, keeping feet apart laterally
        reference = [(0.04, 0.0, 0.0)] * 30

        self.footstep_planner = footstep_planner.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )

        # initialize MPC controller
        self.mpc = ismpc.Ismpc(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize kalman filter
        A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
        B = self.params['world_time_step'] * self.mpc.B_lip
        d = np.zeros(9)
        d[7] = - self.params['world_time_step'] * self.params['g']
        H = np.identity(3)
        Q = block_diag(1., 1., 1.)
        R = block_diag(1e1, 1e2, 1e4)
        P = np.identity(3)
        x = np.array([self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0], \
                      self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1], \
                      self.initial['com']['pos'][2], self.initial['com']['vel'][2], self.initial['zmp']['pos'][2]])
        self.kf = filter.KalmanFilter(block_diag(A, A, A), \
                                      block_diag(B, B, B), \
                                      d, \
                                      block_diag(H, H, H), \
                                      block_diag(Q, Q, Q), \
                                      block_diag(R, R, R), \
                                      block_diag(P, P, P), \
                                      x)

        # initialize logger and plots
        self.logger = Logger(self.initial)
        #self.logger.initialize_plot(frequency=10)
        
    def customPreStep(self):
        # create current and desired states
        self.current = self.retrieve_state()

        # update kalman filter
        u = np.array([self.desired['zmp']['vel'][0], self.desired['zmp']['vel'][1], self.desired['zmp']['vel'][2]])
        self.kf.predict(u)
        x_flt, _ = self.kf.update(np.array([self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0], \
                                            self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1], \
                                            self.current['com']['pos'][2], self.current['com']['vel'][2], self.current['zmp']['pos'][2]]))
        
        # update current state using kalman filter output
        self.current['com']['pos'][0] = x_flt[0]
        self.current['com']['vel'][0] = x_flt[1]
        self.current['zmp']['pos'][0] = x_flt[2]
        self.current['com']['pos'][1] = x_flt[3]
        self.current['com']['vel'][1] = x_flt[4]
        self.current['zmp']['pos'][1] = x_flt[5]
        self.current['com']['pos'][2] = x_flt[6]
        self.current['com']['vel'][2] = x_flt[7]
        self.current['zmp']['pos'][2] = x_flt[8]

        # get references using mpc
        try:
            lip_state, contact = self.mpc.solve(self.current, self.time)
            
            # --- THE TRAP FIX ---
            self.contact = contact 
            # --------------------
            
        except Exception as e:
            print("\n" + "="*60)
            print(f"!!! FATAL MPC CRASH CAUGHT AT TIME STEP: {self.time} !!!")
            print("="*60)
            print(f"Phase       : {self.contact}")
            print(f"CURRENT CoM Vel: {np.round(self.current['com']['vel'], 4)}")
            print(f"DESIRED CoM Vel: {np.round(self.desired['com']['vel'], 4)}")
            print("="*60 + "\n")
            raise e
        
        self.desired['com']['pos'] = lip_state['com']['pos']
        self.desired['com']['vel'] = lip_state['com']['vel']
        self.desired['com']['acc'] = lip_state['com']['acc']
        self.desired['zmp']['pos'] = lip_state['zmp']['pos']
        self.desired['zmp']['vel'] = lip_state['zmp']['vel']

        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]

        # set torso and base references to the average of the feet
        for link in ['torso', 'base']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[link][key] = (self.desired['lfoot'][key][:3] + self.desired['rfoot'][key][:3]) / 2.

        # get torque commands using inverse dynamics
        commands = self.id.get_joint_torques(self.desired, self.current, contact) 
        
        # set acceleration commands
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        # log and plot
        self.logger.log_data(self.current, self.desired)
        #self.logger.update_plot(self.time)

        self.time += 1

        # log and plot
        self.logger.log_data(self.desired, self.current)
        #self.logger.update_plot(self.time)

    def retrieve_state(self):
        # com and torso pose (orientation and position)
        com_position = self.hrp4.getCOM()
        torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())
        base_orientation  = get_rotvec(self.hrp4.getBodyNode('base_link' ).getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())

        # feet poses (orientation and position)
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))
        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # velocities
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = self.hrp4.getBodyNode('base_link').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        # compute total contact force
        force = np.zeros(3)
        for contact in world.getLastCollisionResult().getContacts():
            force += contact.force

        # compute zmp
        zmp = np.zeros(3)
        zmp[2] = com_position[2] - force[2] / (self.hrp4.getMass() * self.params['g'] / self.params['h'])
        for contact in world.getLastCollisionResult().getContacts():
            if contact.force[2] <= 0.1: continue
            zmp[0] += (contact.point[0] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[0] / force[2])
            zmp[1] += (contact.point[1] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[1] / force[2])

        if force[2] <= 0.1: 
            zmp = np.array([0., 0., 0.]) 
        else:
            midpoint = (l_foot_position + r_foot_position) / 2.
            zmp[0] = np.clip(zmp[0], midpoint[0] - 0.08, midpoint[0] + 0.08)
            zmp[1] = np.clip(zmp[1], midpoint[1] - 0.08, midpoint[1] + 0.08)
            zmp[2] = midpoint[2]
        # create state dict
        return {
            'lfoot': {'pos': left_foot_pose,
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose,
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_orientation,
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_orientation,
                      'vel': base_angular_velocity,
                      'acc': np.zeros(3)},
            'joint': {'pos': self.hrp4.getPositions(),
                      'vel': self.hrp4.getVelocities(),
                      'acc': np.zeros(self.params['dof'])},
            'zmp'  : {'pos': zmp,
                      'vel': np.zeros(3),
                      'acc': np.zeros(3)}
        }

if __name__ == "__main__":
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "nao.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(0.005) # Decreased time step for NAO's fast frequency [cite: 18]

    print("\n===== DART DOF / ROOT CHECK =====")
    print("Skeleton name:", hrp4.getName())
    print("Num DOFs:", hrp4.getNumDofs())
    print("Root joint type:", hrp4.getRootJoint().getType())
    print("Root joint DOFs:", hrp4.getRootJoint().getNumDofs())

    print("\nFirst 25 DOFs (index : name):")
    for i in range(min(25, hrp4.getNumDofs())):
        print(f"{i:2d}: {hrp4.getDof(i).getName()}")
    print("===== END CHECK =====\n")


    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 1.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)

    node = Hrp4Controller(world, hrp4)
   
    viewer = dart.gui.osg.Viewer()
    node.setTargetRealTimeFactor(10) 
    viewer.addWorldNode(node)

    viewer.setUpViewInWindow(0, 0, 1280, 720)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    try:
        viewer.run()
    finally:
        plot_controlled_results(node, label="nao_debug", save_dir="plots")