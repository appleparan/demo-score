from scripted_policy import *


#PegA is an alias for InsertionPolicy, the default that comes with the original ACT repo
class PegAPolicy(InsertionPolicy):
    pass


class PegBPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-120)

        meet_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=140)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=122 + 2*np.random.randn())

        meet_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60 + + 2*np.random.randn())

        meet_xyz = np.array([0, 0.5, 0.33])
        meet_xyz[0] += 0.05*np.random.randn()
        meet_xyz[1] += 0.08*np.random.randn()
        meet_xyz[2] += 0.05*np.random.randn()
        lift_right = -0.1

        x_offset = np.random.uniform(-0.02, 0.05)
        
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 50, "xyz": socket_xyz + np.array([0., 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 90, "xyz": socket_xyz + np.array([0., 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 110, "xyz": socket_xyz + np.array([0., 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 140, "xyz": socket_xyz + np.array([0., 0, 0.12]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 200, "xyz": meet_xyz + np.array([0.03, 0, 0.15]), "quat": meet_quat_left.elements, "gripper": 0},
            {"t": 260, "xyz": meet_xyz + np.array([0.07, 0, 0.05]), "quat": meet_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.07, 0, 0.05]), "quat": meet_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.07, 0, 0.05]), "quat": meet_quat_left.elements, "gripper": 0},  # insertion
        ]
        
        d = .1
        tan = np.tan(np.pi/6)
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 50, "xyz": peg_xyz + np.array([x_offset, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 70, "xyz": peg_xyz + np.array([x_offset, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 100, "xyz": peg_xyz + np.array([x_offset, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 130, "xyz": peg_xyz + np.array([x_offset, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 220, "xyz": np.array([meet_xyz[0], meet_xyz[1], 0.1 - x_offset/2]), "quat": meet_quat_right.elements, "gripper": 0},
            {"t": 260, "xyz": meet_xyz + np.array([x_offset/2. - 0.005, 0, lift_right]), "quat": meet_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([x_offset/2. + d*tan - 0.005, 0, lift_right + d]), "quat": meet_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([x_offset/2. + d*tan - 0.005, 0, lift_right + d]), "quat": meet_quat_right.elements, "gripper": 0},  # insertion
        ]


class PegCPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-120)

        meet_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=140)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=120)

        meet_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_xyz = np.array([0, 0.5, 0.3])
        lift_right = -0.1

        

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 50, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 90, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 110, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 150, "xyz": socket_xyz + np.array([0, 0, 0.12]), "quat": gripper_pick_quat_left.elements, "gripper": 0},
            {"t": 220, "xyz": meet_xyz + np.array([0.03, 0, 0.15]), "quat": meet_quat_left.elements, "gripper": 0},
            {"t": 260, "xyz": meet_xyz + np.array([0.07, 0, 0.05]), "quat": meet_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.07, 0, 0.05]), "quat": meet_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.07, 0, 0.05]), "quat": meet_quat_left.elements, "gripper": 0},  # insertion
        ]

        d = .1
        tan = np.tan(np.pi/6)
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 50, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 70, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 100, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 130, "xyz": peg_xyz + np.array([0, 0, 0.1]), "quat": gripper_pick_quat_right.elements, "gripper": 0},
            {"t": 260, "xyz": meet_xyz + np.array([0., 0, lift_right]), "quat": meet_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.0 + d*tan, 0, lift_right + d]), "quat": meet_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.0 + d*tan, 0, lift_right + d]), "quat": meet_quat_right.elements, "gripper": 0},  # insertion

        ]
