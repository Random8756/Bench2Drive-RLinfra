#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario in which the ego is parked between two vehicles and has to maneuver to start the route.
"""

from __future__ import print_function

import py_trees
import carla
import random
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      ActorTransformSetter,
                                                                      WaitForever,
                                                                      ChangeAutoPilot,
                                                                      ScenarioTimeout)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.tools.background_manager import ChangeRoadBehavior, RemoveRoadLane


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict['x']),
        y=float(actor_dict['y']),
        z=float(actor_dict['z'])
    )
    return location


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default


class UnproperSpawn(BasicScenario):
    

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True,
                 timeout=180, activate_scenario=True):
        # Get parking_waypoint based on trigger_point
        self._scenario_timeout = 240
        self._map = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)
        self._direction = get_value_parameter(config, 'direction', str, 'right')
        if self._direction == "left":
            self._parking_waypoint = self._reference_waypoint.get_left_lane()
        else:
            self._parking_waypoint = self._reference_waypoint.get_right_lane()

        if self._parking_waypoint is None:
            raise Exception(
                "Couldn't find parking point on the {} side".format(self._direction))

        super().__init__("UnproperSpawn",
                         ego_vehicles,
                         config,
                         world,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        # Move the ego to its side position
        try_repeat = 0
        while try_repeat < 20:
            try:
                self._ego_transform = self._get_displaced_transform(self.ego_vehicles[0], self._parking_waypoint)
                self.ego_vehicles[0].set_transform(self._ego_transform)
                break
            except Exception:
                try_repeat += 1
                pass

    def _get_displaced_transform(self, actor, wp):
        """
        Calculates the transforming such that the actor is at the sidemost part of the lane
        """
        wp = self._map.get_waypoint(actor.get_location())
        # if wp.get_right_lane():
        #     wp = wp.get_right_lane()
        #     new_location = carla.Location(x=wp.transform.location.x,
        #                                   y=wp.transform.location.y,
        #                                   z=wp.transform.location.z)
        # elif wp.get_left_lane():
        #     wp = wp.get_left_lane()
        #     new_location = carla.Location(x=wp.transform.location.x,
        #                                   y=wp.transform.location.y,
        #                                   z=wp.transform.location.z)
        # else:
        #     pass
        
        horizontal_vector = wp.transform.get_right_vector()* random.choice([-1, 1])
        horizontal_displacement = round(random.uniform(1, 2), 1)
        new_location = wp.transform.location + carla.Location(x=horizontal_displacement*horizontal_vector.x,
                                                              y=horizontal_displacement*horizontal_vector.y,
                                                              z=0.)
        vertical_vector = wp.transform.get_forward_vector() * random.choice([-1, 1])
        vertical_displacement = round(random.uniform(0.5, 1), 1)
        new_location = new_location + carla.Location(x=vertical_displacement*vertical_vector.x,
                                                     y=vertical_displacement*vertical_vector.y,
                                                     z=0.)
        new_yaw = round(random.uniform(0, 360), 1)
        new_rotation = carla.Rotation(pitch=wp.transform.rotation.pitch,
                                       yaw=new_yaw,
                                       roll=wp.transform.rotation.roll)
        new_location.z += 0.05  # Just in case, avoid collisions with the ground
        return carla.Transform(new_location, new_rotation)

    def _create_behavior(self):
        sequence = py_trees.composites.Sequence(name="UnproperSpawn")
        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], 200))
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        sequence.add_child(end_condition)
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
