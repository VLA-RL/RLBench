# From https://github.com/VLA-RL/RLBench-env-templates/blob/main/task/pick_described_object.py
from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, GraspedCondition, ConditionSet
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np
from scipy.spatial.transform import Rotation as R

GROCERY_NAMES = [
    "chocolate jello",
    "soup",
    "spam",
    "mustard",
    "sugar",
]

item_names = [
    ["chocolate pudding mix", "chocolate dessert mix", "chocolate gelatin", "chocolate jello powder", "chocolate treat mix"],
    ["canned soup", "soup can", "soup tin", "soup canister", "soup container"],
    ["spam tin", "canned ham", "spam meat", "spam can", "spam container"],
    ["mustard bottle", "mustard", "mustard condiment", "mustard container", "mustard jar"],
    ["sugar box", "sugar packet", "sugar carton", "sugar container", "sugar pack"]
]

instructions = [
    "put the %s in the basket",
    "pick up the %s and place it in the basket",
    "store the %s in the basket",
    "move the %s to the basket",
    "place the %s in the basket"
]


class PickDescribedObject(Task):
    def init_task(self) -> None:
        self.groceries = [Shape(name.replace(" ", "_")) for name in GROCERY_NAMES]
        self.grasp_points = [
            Dummy("%s_grasp_point" % name.replace(" ", "_")) for name in GROCERY_NAMES
        ]

        self.item = Dummy("waypoint0")
        self.over_box = Dummy("waypoint1")
        self.dropin_box = Dummy("waypoint2")
        
        self.register_graspable_objects(self.groceries)
        # [g.set_mass(g.get_mass()*1.1) for g in self.groceries]
        self.boundary = SpawnBoundary([Shape("workspace")])
        self.spawn_boundary = SpawnBoundary([Shape("groceries_boundary")])
        self.success_detector = ProximitySensor("success")
    
    def randomize_pose(self, open:bool = True):
        np.random.seed()
        while True:
            try:
                pos_lower_bound = np.array([-0.2, -0.35, 1])
                pos_upper_bound = np.array([0.5, 0.35, 1.3])
                rot_lower_bound = np.array([-np.pi/2, 0, -np.pi/2])
                rot_upper_bound = np.array([np.pi/2, np.pi/2, np.pi/2])
                pos = np.random.uniform(pos_lower_bound, pos_upper_bound)
                euler = np.random.uniform(rot_lower_bound, rot_upper_bound)
                euler[0] = np.clip(np.random.normal(0, np.pi/6),-np.pi/2,np.pi/2)
                euler[1] = np.clip(abs(np.random.normal(0,np.pi/6)),0, np.pi/2)
                trans = lambda rx: rx - np.pi if rx > 0 else rx + np.pi 
                euler[0] = trans(euler[0])
                quat = R.from_euler('xyz', euler).as_quat()
                joint_position = self.robot.arm.solve_ik_via_sampling(position=pos, euler=euler,trials=10)
                self.robot.arm.set_joint_positions(joint_position[0], disable_dynamics=True)
                if open:
                    state = 0.039
                else:
                    state = 0
                self.robot.gripper.set_joint_positions([state, state])
                
                break
            except Exception as e:
                # print(e)
                continue
        # return joint_position, pos, euler, quat

    def init_episode(self, index: int) -> List[str]:
        self.spawn_boundary.clear()
        [self.spawn_boundary.sample(g, min_distance=0.2) for g in self.groceries]
        self.item.set_pose(self.grasp_points[index].get_pose())
        self.item_name = GROCERY_NAMES[index]
        self.target_object = self.groceries[index]
        self.stage = 0
        self.object_grasped = False
        self.randomize_pose()
        
        grasp_condition = GraspedCondition(self.robot.gripper, self.target_object)
        detect_condition = DetectedCondition(self.target_object, self.success_detector)

        condition_set = ConditionSet(
            [grasp_condition, detect_condition], order_matters=True
        )

        self.register_success_conditions([condition_set])
        
        item_phrases = np.random.choice(item_names[index])
        #fit all the item names into the instructions
        instr = [i % item_phrases for i in instructions]
        return instr

    def variation_count(self) -> int:
        return len(GROCERY_NAMES)

    def boundary_root(self) -> Object:
        return Shape("boundary_root")

    def base_rotation_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)

    # def reward(self) -> float:
    #     reward = -1

    #     if self.robot.gripper.get_grasped_objects():
    #         if not self.object_grasped:
    #             self.object_grasped = True
    #             reward = -1

    #     # If not grasped, penalize based on distance
    #     if not self.object_grasped:
    #         distance = np.linalg.norm(
    #             self.robot.arm.get_tip().get_position()
    #             - self.target_object.get_position()
    #         )
    #         reward -= distance * 10

    #     # Check if task is completed
    #     success, term = self.success()
    #     if success and term:
    #         return 500.0  # Bonus for completing the task
    #     elif not success and term:
    #         return -500.0  # Penalty for failing to put in basket after grasping

    #     return reward
    def step(self) -> None:
        """Called each time the simulation is stepped. Can usually be left."""
        self.gripper_closed = self.robot.gripper.get_joint_positions()[0] < 0.035
        try:
            self.grasped_target = self.robot.gripper.get_grasped_objects()[0] == self.target_object
        except:
            self.grasped_target = False
        if not self.grasped_target:
            if self.gripper_closed:
            #open gripper
                # self.robot.arm.set_joint_positions(self.robot.arm.get_joint_positions())
                self.robot.gripper.set_joint_positions([0.039, 0.039], disable_dynamics=True)
        else:
            self.stage = 1


    def reward(self) -> float:
        reward = 0
        if not self.grasped_target:
            if self.gripper_closed:
                reward -= 1
            distance = np.linalg.norm(
                self.robot.arm.get_tip().get_position()
                - self.target_object.get_position()
            )
            reward -= distance
        else:
            if self.object_grasped:
                distance = np.linalg.norm(
                self.robot.arm.get_tip().get_position()
                - self.get_waypoints()[1]._waypoint.get_position()
                )
                reward -= distance
            self.object_grasped = True
        success, term = self.success()

        if term:
            if success:
                return 0
            else:
                return -100
            
        return reward