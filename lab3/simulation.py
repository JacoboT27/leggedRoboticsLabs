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

class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.time = 0
        self.params = {
            'g': 9.81,
            'h': 0.3,                                       #changed from 0.72 0.3
            'foot_size': 0.08,                              #changed from 0.1 to 0.08
            'step_height': 0.02,
            'ss_duration': 70,
            'ds_duration': 30,
            'world_time_step': world.getTimeStep(),
            'first_swing': 'rfoot',
            'µ': 0.5,
            'N': 100,
            'dof': self.hrp4.getNumDofs(),
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
        print("PARAMETERS VERIFICATION")                    #added verification prints
        print("="*60)
        print(f"h (COM height): {self.params['h']:.3f} m")
        print(f"eta (frequency): {self.params['eta']:.3f} rad/s")
        print(f"foot_size: {self.params['foot_size']:.3f} m")
        print(f"Time step: {self.params['world_time_step']:.4f} s")
        print(f"Horizon N: {self.params['N']}")
        print("="*60)

        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('base_link')          #changed from body to base_link

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # set floating base to passive, everything else to torque
            if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
            elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # set initial configuration
        initial_configuration = {       #replaced for nao names
                        'HeadYaw': 0., 'HeadPitch': 0., 'LHipYawPitch': 0., 'LHipRoll': 0., 'LHipPitch': -25., 'LKneePitch': 50.,
                        'LAnklePitch': -25., 'LAnkleRoll': 0., 'RHipYawPitch': 0., 'RHipRoll': 0.,'RHipPitch': -25., 'RKneePitch': 50., 
                        'RAnklePitch': -25., 'RAnkleRoll': 0., 'LShoulderPitch': 80., 'LShoulderRoll': 10., 'LElbowYaw': -80., 
                        'LElbowRoll': -60., 'RShoulderPitch': 80., 'RShoulderRoll': -10.,'RElbowYaw': 80., 'RElbowRoll': 60.,
        }   
        print("JOINT CONFIGURATION VERIFICATION")   #added verification prints
        print("="*60)
        set_count = 0
        failed_joints = []
        for joint_name, value in initial_configuration.items():
            dof = self.hrp4.getDof(joint_name)
            if dof is None:
                print(f"  ✗ Joint NOT FOUND: {joint_name}")
                failed_joints.append(joint_name)
            else:
                self.hrp4.setPosition(dof.getIndexInSkeleton(), value * np.pi / 180.)
                set_count += 1
        
        print(f"Successfully set {set_count}/{len(initial_configuration)} joints")
        if failed_joints:
            print(f"Failed joints: {failed_joints}")
            exit(1)
        print("="*60)

        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(3, - (lsole_pos[0] + rsole_pos[0]) / 2.)
        self.hrp4.setPosition(4, - (lsole_pos[1] + rsole_pos[1]) / 2.)
        min_sole_z = min(lsole_pos[2], rsole_pos[2])                        #added minimum foot Z
        self.hrp4.setPosition(5, -min_sole_z + 0.001)                       #added 1mm clearance in Z
        print("GROUND POSITIONING VERIFICATION")                            #added verification prints
        print("="*60)
        # Re-read foot positions after placement
        lsole_pos_after = self.lsole.getTransform(dart.dynamics.Frame.World(), dart.dynamics.Frame.World()).translation()
        rsole_pos_after = self.rsole.getTransform(dart.dynamics.Frame.World(), dart.dynamics.Frame.World()).translation()
        print(f"Left foot Z:  {lsole_pos_after[2]:.4f} m")
        print(f"Right foot Z: {rsole_pos_after[2]:.4f} m")
        print(f"COM height:   {self.hrp4.getCOM()[2]:.4f} m")
        if min(lsole_pos_after[2], rsole_pos_after[2]) < -0.01:
            print("  ✗ ERROR: Feet are below ground!")
            exit(1)
        elif min(lsole_pos_after[2], rsole_pos_after[2]) > 0.05:
            print("  ⚠ WARNING: Feet are too high above ground")
        else:
            print("  ✓ Feet are on ground")
        print("="*60)
        # initialize state
        self.initial = self.retrieve_state()
        self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [                                                          #changed namaes for nao names
                        "HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
                        "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "LWristYaw", "RWristYaw",
                        "LFinger11", "LFinger12", "LFinger13", "LFinger21", "LFinger22", "LFinger23","LHand", 
                        "LThumb1", "LThumb2", "RFinger11", "RFinger12", "RFinger13", "RFinger21", "RFinger22", 
                        "RFinger23", "RHand", "RThumb1", "RThumb2",]        
        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs, foot_size=self.params['foot_size'])

        # initialize footstep planner
        #reference = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10
        reference = [(0.02, 0.0, 0.0)] * 25         #changed reference velocity
        self.footstep_planner = footstep_planner.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )
        print("FOOTSTEP PLANNER VERIFICATION")      #added verification prints
        print("="*60)
        print(f"Number of steps: {len(self.footstep_planner.plan)}")
        print("First 3 steps:")
        for i in range(min(3, len(self.footstep_planner.plan))):
            step = self.footstep_planner.plan[i]
            print(f"  Step {i}: {step['foot_id']}")
            print(f"    Position: {step['pos']}")
            print(f"    SS duration: {step['ss_duration']}, DS duration: {step['ds_duration']}")
        print("="*60)
        print("INITIALIZATION VERIFICATION")
        print("="*60)
        # initialize MPC controller
        self.mpc = ismpc.Ismpc(
            self.initial, 
            self.footstep_planner, 
            self.params
            )
        print("[DEBUG] MPC initialized successfully ✓")
        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial, 
            self.footstep_planner, 
            self.params
            )
        print("[DEBUG] Foot trajectory generator initialized ✓")
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
        print("[DEBUG] Kalman filter initialized ✓")
        # initialize logger and plots
        #self.logger = Logger(self.initial)
        #self.logger.initialize_plot(frequency=10)
        self.settling = True                            #added
        self.settling_counter = 0                       #added
        self.settling_steps = 200                       #added
        print("[DEBUG] Initialization complete! ✓")
        print("="*60)
        print("SETTLING PHASE VERIFICATION")
        print("="*60)

    def customPreStep(self):
        if self.settling:
            print(f"[DEBUG] In settling phase, counter={self.settling_counter}")
            self.settling_counter += 1
            com = self.hrp4.getCOM()
            if np.any(np.isnan(com)) or np.any(np.isinf(com)):
                print(f"[ERROR] COM is invalid: {com}")
                exit(1)
            print(f"[DEBUG] COM: [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}]")

            for i in range(6, self.params['dof']):
                dof = self.hrp4.getDof(i)
                target_pos = self.initial['joint']['pos'][i]
                current_pos = dof.getPosition()
                current_vel = dof.getVelocity()
                if np.isnan(current_pos) or np.isnan(current_vel):
                    print(f"[ERROR] Joint {i} ({dof.getName()}) has NaN: pos={current_pos}, vel={current_vel}")
                    exit(1)
                error = target_pos - current_pos
                torque = 20.0 * error - 2.0 * current_vel
                if abs(torque) > 500.0:
                    print(f"[WARNING] Joint {i} ({dof.getName()}) has large torque: {torque:.2f} N⋅m")
                self.hrp4.setCommand(i, torque)
            print("[DEBUG] Torques applied successfully")
        
            if self.settling_counter % 50 == 0:
                com = self.hrp4.getCOM()
                print(f"Settling {self.settling_counter}/{self.settling_steps}: COM Z = {com[2]:.4f} m")
            
            if self.settling_counter >= self.settling_steps:
                self.settling = False
                print("\n" + "="*60)
                print("SETTLING COMPLETE - Re-initializing")
                print("="*60)
                
                self.initial = self.retrieve_state()
                print("[DEBUG] retrieve_state() completed")
                self.params['h'] = float(self.hrp4.getCOM()[2])
                self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
                self.desired = copy.deepcopy(self.initial)
                
                self.mpc = ismpc.Ismpc(self.initial, self.footstep_planner, self.params)
                
                A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
                B = self.params['world_time_step'] * self.mpc.B_lip
                d = np.zeros(9)
                d[7] = -self.params['world_time_step'] * self.params['g']
                H = np.identity(3)
                Q = block_diag(1., 1., 1.)
                R = block_diag(1e1, 1e2, 1e4)
                P = np.identity(3)
                x = np.array([self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0],
                            self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1],
                            self.initial['com']['pos'][2], self.initial['com']['vel'][2], self.initial['zmp']['pos'][2]])
                self.kf = filter.KalmanFilter(block_diag(A,A,A), block_diag(B,B,B), d,
                                            block_diag(H,H,H), block_diag(Q,Q,Q),
                                            block_diag(R,R,R), block_diag(P,P,P), x)
                
                print(f"Settled h: {self.params['h']:.4f} m")
                print(f"Settled eta: {self.params['eta']:.4f}")
                print("Starting main control loop...")
                print("="*60)
            print("[DEBUG] Returning from settling")
            return
        print("[DEBUG] Past settling, in main control loop")
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
        lip_state, contact = self.mpc.solve(self.current, self.time)

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
        #self.logger.log_data(self.current, self.desired)
        #self.logger.update_plot(self.time)

        self.time += 1

    def retrieve_state(self):
        # com and torso pose (orientation and position)
        com_position = self.hrp4.getCOM()
        torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation())
        base_orientation  = get_rotvec(self.hrp4.getBodyNode('base_link' ).getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).rotation()) #changed from body to base_link

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
        base_angular_velocity = self.hrp4.getBodyNode('base_link').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())  #changed from body to base_link
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        # compute total contact force
        force = np.zeros(3)
        for contact in self.world.getLastCollisionResult().getContacts():
            force += contact.force
        force_z_magnitude = abs(force[2])                                   #added
        # compute zmp                                                       #replaced
        zmp = np.zeros(3)
        midpoint = (l_foot_position + r_foot_position) / 2.
        
        if force_z_magnitude > 1.0:  # Sufficient contact
            for contact in self.world.getLastCollisionResult().getContacts():
                contact_force_z = abs(contact.force[2])
                if contact_force_z > 0.1:
                    zmp[0] += contact.point[0] * contact_force_z / force_z_magnitude
                    zmp[1] += contact.point[1] * contact_force_z / force_z_magnitude
            zmp[2] = 0.0                                                    #always on the ground
        else:
            zmp = midpoint.copy()                                           # No contact - use midpoint
            zmp[2] = 0.0                                                    #always on the ground
        zmp[0] = np.clip(zmp[0], midpoint[0] - 0.15, midpoint[0] + 0.15)    #clip
        zmp[1] = np.clip(zmp[1], midpoint[1] - 0.15, midpoint[1] + 0.15)    #clip
        zmp[2] = np.clip(zmp[2], -0.05, 0.05)                               #clip
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
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "nao.urdf"))            #changed hrp4.urdf to nao.urdf
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(0.005)

    print("="*60)
    print("ROBOT LOADING VERIFICATION")                                                         #added verification prints
    print("="*60)
    print(f"Robot name: {hrp4.getName()}")
    print(f"Total DOFs: {hrp4.getNumDofs()}")
    print(f"Total mass: {hrp4.getMass():.2f} kg")
    print(f"Root joint: {hrp4.getRootJoint().getType()}")
    print(f"Number of body nodes: {hrp4.getNumBodyNodes()}")
    critical_nodes = ['l_sole', 'r_sole', 'torso', 'base_link']
    print("\nCritical body nodes:")
    for node_name in critical_nodes:
        node = hrp4.getBodyNode(node_name)
        if node:
            print(f"  ✓ {node_name} found")
        else:
            print(f"  ✗ {node_name} MISSING!")
            exit(1)
    print("="*60)

    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 0.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)
    
    for i in range(hrp4.getNumDofs()):              #added
        hrp4.getDof(i).setDampingCoefficient(1.0)

    node = Hrp4Controller(world, hrp4)

    # create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    node.setTargetRealTimeFactor(10) # speed up the visualization by 10x
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    viewer.run()
