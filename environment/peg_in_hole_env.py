from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, xml_path_completion
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.models.objects import MujocoXMLObject

"""
Offset functions for peg and hole objects. These are needed because the peg and 
hole objects are placed on the robot relative to the robot's end effector. This
means that based on the position/orientation of the end-effector, the peg and 
hole may require different offsets to be placed correctly. For example, some of
the humanoid robots are positioned such that changes in the z-direction result
in translations going in the opposite direction. Hence, we flip the sign of the
z-component of the offset for these robots.
"""
_flip_last_dim_fn = lambda offset: offset[:-1] + [-1 * offset[-1]]
_OBJECT_POS_OFFSET_FN = {
    "GR1FloatingBody": {
        "peg": _flip_last_dim_fn,
        "hole": _flip_last_dim_fn,
    },
    "GR1": {
        "peg": _flip_last_dim_fn,
        "hole": _flip_last_dim_fn,
    },
    "G1": {
        "peg": _flip_last_dim_fn,
        "hole": _flip_last_dim_fn,
    },
    "G1FloatingBody": {"peg": _flip_last_dim_fn, "hole": _flip_last_dim_fn},
}


class PlateWithHoleObject(MujocoXMLObject):
    """
    Square plate with a hole in the center (used in PegInHole)
    """

    def __init__(self, name, scale=1.0):
        super().__init__(
            xml_path_completion("objects/plate-with-hole.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
            scale=scale,
        )


class PegInHole(ManipulationEnv):
    """
    This class corresponds to the stacking task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
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
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

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

    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:

            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_insert = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_insert)
        else:
            reward = 2.0 if r_insert > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for inserting
        """
        # reaching is successful when the gripper site is close to the center of the peg
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        dist = min(
            [
                np.linalg.norm(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]] - peg_pos)
                for arm in self.robots[0].arms
            ]
        )
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # grasping reward
        grasping_peg = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.peg)
        if grasping_peg:
            r_reach += 0.25

        # lifting is successful when the peg is above the table top by a margin
        peg_height = peg_pos[2]
        table_height = self.table_offset[2]
        peg_lifted = peg_height > table_height + 0.04
        r_lift = 1.0 if peg_lifted else 0.0

        # Aligning is successful when peg is right above hole
        if peg_lifted:
            horiz_dist = np.linalg.norm(np.array(peg_pos[:2]) - np.array(hole_pos[:2]))
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # insertion is successful when the peg is lifted and the gripper is not holding the object
        r_insert = 0
        peg_touching_hole = self.check_contact(self.peg, self.hole)
        if not grasping_peg and r_lift > 0 and peg_touching_hole:
            r_insert = 2.0

        return r_reach, r_lift, r_insert

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        red_material = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        green_material = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.peg = CylinderObject(
            name="peg",
            size_min=[0.02, 0.04],  # [radius, height]
            size_max=[0.02, 0.04],
            rgba=[1, 0, 0, 1],
            material=red_material,
        )
        self.hole = PlateWithHoleObject(
            name="hole",
            scale=0.5
        )

        # Load hole object
        hole_obj = self.hole.get_obj()
        hole_obj.set("quat", "0 0 0 1")
        hole_robot = self.robots[0]
        # Get offset function for hole object based on the robot type
        hole_offset_fn = _OBJECT_POS_OFFSET_FN.get(hole_robot.robot_model.__class__.__name__, {}).get(
            "hole", lambda x: x
        )

        hole_pos = self.table_offset.copy()
        hole_pos[1] += 0.17
        hole_obj.set("pos", array_to_string(hole_pos))

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.peg)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.peg],
                x_range=[-0.1, 0.1],
                y_range=[-0.1, 0.1],
                rotation=None,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.peg, self.hole],
        )

        # Make sure to add relevant assets from peg and hole objects
        self.model.merge_assets(self.hole)
        self.model.merge_assets(self.peg)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg_body_id])

            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.peg_body_id]), to="xyzw")

            @sensor(modality=modality)
            def hole_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hole_body_id])

            @sensor(modality=modality)
            def hole_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.hole_body_id]), to="xyzw")

            @sensor(modality=modality)
            def peg_to_hole(obs_cache):
                return (
                    obs_cache["hole_pos"] - obs_cache["peg_pos"]
                    if "peg_pos" in obs_cache and "hole_pos" in obs_cache
                    else np.zeros(3)
                )

            # Get prefix from robot model to avoid naming clashes
            arm_prefixes = self.robots[0].robot_model.naming_prefix
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors = [peg_pos, peg_quat, hole_pos, hole_quat, peg_to_hole]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _check_success(self):
        """
        Check if peg is successfully inserted into the hole.

        Returns:
            bool: True if peg is correctly inserted
        """
        peg_pos = self.sim.data.body_xpos[self.peg_body_id]
        hole_pos = self.sim.data.body_xpos[self.hole_body_id]

        # Check height of peg is correct
        if abs(peg_pos[2] - hole_pos[2]) > 0.02:
            return False

        # Check horizontal alignment
        if np.linalg.norm(peg_pos[:2] - hole_pos[:2]) > 0.02:
            return False

        return True

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the peg.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        super().visualize(vis_settings)

        # Color the gripper visualization site according to its distance to the peg
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.peg)
