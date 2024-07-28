from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np

GROCERY_NAMES = [
    "chocolate jello",
    # "strawberry jello",
    "soup",
    "spam",
    "mustard",
    "sugar",
]


class PickDescribedObject(Task):
    def init_task(self) -> None:
        self.groceries = [Shape(name.replace(" ", "_")) for name in GROCERY_NAMES]
        self.grasp_points = [
            Dummy("%s_grasp_point" % name.replace(" ", "_")) for name in GROCERY_NAMES
        ]
        # self.ready = Dummy("waypoint0")
        self.item = Dummy("waypoint0")
        # self.lift_item = Dummy("waypoint1")
        self.over_box = Dummy("waypoint1")
        self.dropin_box = Dummy("waypoint2")

        self.register_graspable_objects(self.groceries)
        self.boundary = SpawnBoundary([Shape("workspace")])
        self.spawn_boundary = SpawnBoundary([Shape("groceries_boundary")])
        
    def randomize_arm_position(self):
        current_joint_positions = self.robot.arm.get_joint_positions()
        noise = np.random.uniform(-0.5, 0.5, len(current_joint_positions))
        new_joint_positions = current_joint_positions + noise
        self.robot.arm.set_joint_positions(new_joint_positions, disable_dynamics=True)
        
        
    def init_episode(self, index: int) -> List[str]:
        self.spawn_boundary.clear()
        [self.spawn_boundary.sample(g, min_distance=0.1) for g in self.groceries]
        self.item.set_pose(self.grasp_points[index].get_pose())
        self.item_name = GROCERY_NAMES[index]
        self.register_success_conditions(
            [DetectedCondition(self.groceries[index], ProximitySensor("success"))]
        )
        
        # self.randomize_arm_position()

        return [
            "put the %s in the basket" % GROCERY_NAMES[index],
            "pick up the %s and place in the basket" % GROCERY_NAMES[index],
            "I want to put away the %s" % GROCERY_NAMES[index],
        ]

    def variation_count(self) -> int:
        return len(GROCERY_NAMES)

    def boundary_root(self) -> Object:
        return Shape("boundary_root")

    def base_rotation_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)
