from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import mat2euler
from robosuite.utils.mjcf_utils import CustomMaterial

class TowerOfHanoi(ManipulationEnv):
    """
    This class corresponds to the Tower of Hanoi task for a single robot arm.
    
    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
        env_configuration (str): Specifies how to position the robots within the environment
        controller_configs (str or list of dict): Controller parameters for creating a custom controller
        gripper_types (str or list of str): Type of gripper to use
        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters
        num_disks (int): Number of disks in the Tower of Hanoi puzzle (default is 3)
        use_camera_obs (bool): If True, every observation includes rendered image(s)
        use_object_obs (bool): If True, include object information in the observation
        reward_scale (None or float): Scales the normalized reward function
        reward_shaping (bool): If True, use dense rewards
        placement_initializer (ObjectPositionSampler): Used to place objects on reset
        horizon (int): Every episode lasts for exactly @horizon timesteps
        ...
    """
    def __init__(
        self,
        robots,
        num_disks=3,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (0.8, 0.3, 0.05)
        self.table_offset = (-0.2, -0.35, 0.8)
        
        # reward configuration
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # Tower of Hanoi specific settings
        self.num_disks = num_disks
        self.pole_height = 0.15
        self.pole_radius = 0.01
        self.disk_height = 0.02
        self.min_disk_radius = 0.03
        self.max_disk_radius = 0.08
        
        # object placement initializer
        self.placement_initializer = placement_initializer

        # Track initialization state
        self.initialization_complete = False

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
        
        # Mark initialization as complete
        self.initialization_complete = True

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        self.mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )
        
        # create poles for Tower of Hanoi
        self.poles = []
        pole_positions = [(-0.2, 0), (0, 0), (0.2, 0)]  # Left, middle, right poles
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        for pos in pole_positions:
            pole = CylinderObject(
                name=f"pole_{len(self.poles)}",
                size=[self.pole_radius, self.pole_height],
                rgba=[0.3, 0.3, 0.3, 1],
                friction=0.1,
                material=redwood,
            )
            self.poles.append(pole)
        
        # create disks
        self.disks = []
        for i in range(self.num_disks):
            # Larger disks have bigger radius
            radius = self.max_disk_radius - i * (self.max_disk_radius - self.min_disk_radius) / (self.num_disks - 1)
            disk = CylinderObject(
                name=f"disk_{i}",
                size=[radius, self.disk_height/2],
                rgba=[0.1 + 0.2*i, 0.3, 0.3, 1],
                friction=0.1,
                joints=[{"type": "free"}],  # Disk can move freely
            )
            self.disks.append(disk)
        
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.poles + self.disks,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.poles + self.disks)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.pole_body_ids = dict()
        self.disk_body_ids = dict()

        # Get site ids for poles and disks
        for pole in self.poles:
            self.object_body_ids[pole.name] = self.sim.model.body_name2id(pole.root_body)
            self.pole_body_ids[pole.name] = self.sim.model.body_name2id(pole.root_body)

        for disk in self.disks:
            self.object_body_ids[disk.name] = self.sim.model.body_name2id(disk.root_body)
            self.disk_body_ids[disk.name] = self.sim.model.body_name2id(disk.root_body)

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        
        Important keys:
            robot-state: contains robot-specific information.
            object-state: requires @self.use_object_obs to be True.
            image: requires @self.use_camera_obs to be True.
        """
        di = super()._get_observation()

        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_names[0],
                width=self.camera_widths[0],
                height=self.camera_heights[0],
                depth=self.camera_depths[0],
            )
            di['image'] = camera_obs

        # low-level object information
        if self.use_object_obs:
            # Get object information from observable sensors
            di["pole_pos"] = self._poles_pos
            di["disk_pos"] = self._disks_pos
            di["disk_rot"] = self._disks_rot

        return di

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get object observations
            modality = "object"

            @sensor(modality=modality)
            def pole_pos(obs_cache):
                """
                Returns positions of poles
                """
                poles = []
                for pole_name in sorted(self.pole_body_ids.keys()):
                    poles.append(self.sim.data.body_xpos[self.pole_body_ids[pole_name]])
                return np.array(poles)

            @sensor(modality=modality)
            def disk_pos(obs_cache):
                """
                Returns positions of disks
                """
                disks = []
                for disk_name in sorted(self.disk_body_ids.keys()):
                    disks.append(self.sim.data.body_xpos[self.disk_body_ids[disk_name]])
                return np.array(disks)

            @sensor(modality=modality)
            def disk_rot(obs_cache):
                """
                Returns rotations of disks as euler angles
                """
                rots = []
                for disk_name in sorted(self.disk_body_ids.keys()):
                    body_id = self.disk_body_ids[disk_name]
                    rots.append(mat2euler(self.sim.data.body_xmat[body_id].reshape(3, 3)))
                return np.array(rots)

            sensors = [pole_pos, disk_pos, disk_rot]
            names = [s.__name__ for s in sensors]
            
            # Create observables for each sensor
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables


    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # First reset the parent class which will initialize the simulation
        super()._reset_internal()


        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in {
            "pole_0": (np.array([0.2, 0, self.disk_height/2]), np.array([0, 0, 0, 1]), self.poles[0]),
            "pole_1": (np.array([0.2, 0, self.disk_height/2 + self.disk_height]), np.array([0, 0, 0, 1]), self.poles[1]),
            "pole_2": (np.array([0.2, 0, self.disk_height/2 + 2*self.disk_height]), np.array([0, 0, 0, 1]), self.poles[2]),
        }.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                

        # Loop through all objects and reset their positions
        for obj_pos, obj_quat, obj in {
            "disk_0": (np.array([0.2, 0, self.disk_height/2]), np.array([0, 0, 0, 1]), self.disks[0]),
            "disk_1": (np.array([0.2, 0, self.disk_height/2 + self.disk_height]), np.array([0, 0, 0, 1]), self.disks[1]),
            "disk_2": (np.array([0.2, 0, self.disk_height/2 + 2*self.disk_height]), np.array([0, 0, 0, 1]), self.disks[2]),
        }.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                
        
        # Reset disks to initial configuration (stacked on leftmost pole)
        #
        # for i, disk in enumerate(self.disks):
        #     disk_body_id = self.sim.model.body_name2id(disk.root_body)
        #     disk_pos = self.table_offset + np.array([0.2, 0, self.disk_height/2 + i*self.disk_height])
        #     self.sim.data.body_xpos[disk_body_id] = disk_pos
        #     self.sim.data.body_xquat[disk_body_id] = np.array([0, 0, 0, 1])

    def reward(self, action=None):
        """
        Reward function for the task.
        
        Sparse un-normalized reward:
            - a discrete reward of 1.0 is provided if all disks are stacked correctly on the rightmost pole
            
        Dense un-normalized reward:
            - a smooth reward based on the distance of each disk to its target position
            - a penalty for collisions
            - a reward for lifting objects and keeping them lifted
            
        Args:
            action (np array): The action taken in that timestep
            
        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a dense reward if reward_shaping is enabled
        if self.reward_shaping:
            # Add reward terms for:
            # 1. Distance of disks to target positions
            # 2. Keeping disks lifted
            # 3. Penalties for collisions
            pass  # Implementation depends on specific requirements

        return reward * self.reward_scale

    def _check_success(self):
        """
        Check if all disks are correctly stacked on the rightmost pole.
        
        Returns:
            bool: True if the task has been completed
        """
        # Get positions of all disks
        disk_positions = self._disks_pos
        
        # Check if all disks are on the rightmost pole
        for i, pos in enumerate(disk_positions):
            # Check x-coordinate (should be near 0.2 for rightmost pole)
            if abs(pos[0] - 0.2) > 0.05:
                return False
            
            # Check height (should be stacked in order)
            expected_height = self.table_offset[2] + self.disk_height/2 + i*self.disk_height
            if abs(pos[2] - expected_height) > 0.02:
                return False
        
        return True


    @property
    def _poles_pos(self):
        """
        Grabs the positions of all poles.
        """
        poles = []
        for pole_name in sorted(self.pole_body_ids.keys()):
            poles.append(self.sim.data.body_xpos[self.pole_body_ids[pole_name]])
        return np.array(poles)

    @property
    def _disks_pos(self):
        """
        Grabs the positions of all disks.
        """
        disks = []
        for disk_name in sorted(self.disk_body_ids.keys()):
            disks.append(self.sim.data.body_xpos[self.disk_body_ids[disk_name]])
        return np.array(disks)

    @property
    def _disks_rot(self):
        """
        Grabs the rotations of all disks.
        """
        rots = []
        for disk_name in sorted(self.disk_body_ids.keys()):
            body_id = self.disk_body_ids[disk_name]
            rots.append(mat2euler(self.sim.data.body_xmat[body_id].reshape(3, 3)))
        return np.array(rots)
