from __future__ import print_function

import numpy as np
import py_trees
import random
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      SwitchWrongDirectionTest,
                                                                      BasicAgentBehavior,
                                                                      ScenarioTimeout,
                                                                      Idle, WaitForever,
                                                                      HandBrakeVehicle,
                                                                      OppositeActorFlow,
                                                                      ActorTransformSetter,
                                                                      KeepVelocity)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToVehicle,
                                                                               WaitUntilInFront,
                                                                               WaitUntilInFrontPosition,
                                                                               InTimeToArrivalToLocation)


from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, SetMaxSpeed, ChangeOppositeBehavior, ChangeRoadBehavior

class ObstacleEmerge(BasicScenario):
    """
    This class holds everything required for a scenario in which there is an accident
    in front of the ego, forcing it to lane change. A police vehicle is located before
    two other cars that have been in an accident.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180, activate_scenario=True):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)
        self.timeout = timeout
        
        self.move_dist = random.choice(range(15,25))
        self._min_trigger_dist = random.choice(range(6, 15))
        self._reaction_time = 2.1
        self._end_time = 10
        self._wait_duration = 5
    
        self._offset = 0.0
        
        # stochastic offset  
        self._offset = random.uniform(-0.5, 0.5)

        self._scenario_timeout = 240
        self._number_of_attempts = 6

        super().__init__(
            "ObstacleEmerge", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)
        
    def _get_obstacle_transform(self, waypoint):
        right_vector = waypoint.transform.get_right_vector()
        displacement = carla.Location(self._offset * right_vector.x, self._offset * right_vector.y)
        new_location = waypoint.transform.location + displacement
        self._collision_wp = self._map.get_waypoint(new_location)
        new_location.z = new_location.z - 200
        return carla.Transform(new_location, waypoint.transform.rotation)
        
    def _initialize_actors(self, config):
        """
        """
        move_dist = self.move_dist
        waypoint = self._reference_waypoint
        
        waypoint = waypoint.next(move_dist)
        if not waypoint:
            raise ValueError("Couldn't find a proper location to spawn the obstacle")
        waypoint = waypoint[0]
        spawn_transform = self._get_obstacle_transform(waypoint)
        blueprint = random.choice(['vehicle.*', 'walker.*'])
        obstacle = CarlaDataProvider.request_new_actor(blueprint, spawn_transform)
        if obstacle is None:
            raise ValueError("Counldn't spawn the obstacle")
        self.other_actors.append(obstacle)
        
    def _create_behavior(self):
        sequence = py_trees.composites.Sequence(name="ObstacleEmerge")
        if self.route_mode:
            total_dist = self.move_dist + 15
            sequence.add_child(LeaveSpaceInFront(total_dist))
        collision_location = self._collision_wp.transform.location
        trigger_adversary = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="TriggerObtacleStart")
        trigger_adversary.add_child(InTimeToArrivalToLocation(
            self.ego_vehicles[0], self._reaction_time, collision_location))
        trigger_adversary.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], collision_location, self._min_trigger_dist))
        sequence.add_child(trigger_adversary)
        
        behavior_sequence = py_trees.composites.Sequence("EmergeObstacleSuddenly")
        behavior_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._collision_wp.transform, True))
        behavior_sequence.add_child(Idle(self._end_time))
        behavior_sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyObstacle"))
        
        sequence.add_child(behavior_sequence)
        # sequence.add_child(Idle(self._end_time))
        # sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyObstacle"))
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
        
        
        