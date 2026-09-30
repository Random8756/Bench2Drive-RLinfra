#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import glob
import logging
import os
import sys
import importlib
import inspect
import py_trees
import traceback
import random
import numpy as np

logger = logging.getLogger("Carla Server")

import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ScenarioTriggerer, Idle
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import WaitForBlackboardVariable
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorBlockedTest,
                                                                     MinimumSpeedRouteTest)

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.background_activity import BackgroundBehavior
from srunner.scenariomanager.weather_sim import RouteWeatherBehavior
from srunner.scenariomanager.lights_sim import RouteLightsBehavior
from srunner.scenariomanager.timer import RouteTimeoutBehavior

from leaderboard.utils.route_parser import RouteParser, DIST_THRESHOLD
from leaderboard.utils.route_manipulation import interpolate_trajectory

import leaderboard.utils.parked_vehicles as parked_vehicles

SIMULATOR_TICK_TIMEOUT_LIMIT = 10000 


class RouteScenarioSetupError(RuntimeError):
    """Raised when route scenario setup fails and the worker must abort."""


class RouteScenario(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    category = "RouteScenario"
    INIT_THRESHOLD = 100000 # Runtime initialization trigger distance to ego (m)
    PARKED_VEHICLES_INIT_THRESHOLD = INIT_THRESHOLD - 50 # Runtime initialization trigger distance to parked vehicles (m)

    def __init__(self, route_config, config=None ,debug_mode=0, criteria_enable=True):
        """
        Setup all relevant parameters and create scenarios along route
        """
        # Keep environment options separate from the route config used by BasicScenario.
        self.env_config = config or {}
        self.routes_config = self.env_config.get('routes', {}) if isinstance(self.env_config, dict) else {}
        # Legacy option: limit by waypoint count.
        self.route_length_limit = self.env_config.get('route_length_limit', None)
        # New option: limit by accumulated route distance in meters.
        self.route_max_length_m = self.routes_config.get('route_max_length_m', None)
        self.client = CarlaDataProvider.get_client()

        self.route = self._get_route(route_config)
        self.length = self.compute_length()
        logger.debug('Route Length: %s', self.length)
        logger.debug("Successfully Generate The Route")
        self.world = CarlaDataProvider._world
        self.map = CarlaDataProvider.get_map()
        self.timeout = 10000

        self.all_scenario_classes = None
        self.ego_data = None

        self.scenario_triggerer = None
        self.behavior_node = None # behavior node created by _create_behavior()
        self.criteria_node = None # criteria node created by _create_test_criteria()

        self.list_scenarios = []
        self.occupied_parking_locations = []
        self.available_parking_locations = []

        scenario_configurations = self._filter_scenarios(route_config.scenario_configs)
        self.scenario_configurations = scenario_configurations
        self.missing_scenario_configurations = scenario_configurations.copy()

        ego_vehicle = self._spawn_ego_vehicle()
        if ego_vehicle is None:
            raise ValueError("Shutting down, couldn't spawn the ego vehicle")

        if debug_mode>0:
            self._draw_waypoints(self.route, vertical_shift=0.1, size=0.1, downsample=10)

        self._parked_ids = []
        self._get_parking_slots()

        super(RouteScenario, self).__init__(
            route_config.name, [ego_vehicle], route_config, self.world, debug_mode > 3, False, criteria_enable
        )
        # self.attatch_sensor()
        # Do it after the 'super', as we need the behavior and criteria tree to be initialized
        self.scenario_name = 'NoScenario'
        self.build_scenarios(ego_vehicle, debug=debug_mode > 0)
        self.spawn_parked_vehicles(ego_vehicle)

        # Set runtime init mode. Do this after the first set of scenarios has been initialized!
        # CarlaDataProvider.set_runtime_init_mode(True)
    
    def disable_short_route(self):
        self.route_length_limit = None
        self.route_max_length_m = None

    @staticmethod
    def _compute_route_length_m(route):
        if not route or len(route) <= 1:
            return 0.0

        length = 0.0
        for idx in range(1, len(route)):
            length += route[idx - 1][0].location.distance(route[idx][0].location)
        return length

    @staticmethod
    def _truncate_route_by_length_m(route, gps_route, max_length_m):
        """
        Truncate route by accumulated segment length in meters.
        Keeps at least 2 points (when available) for downstream code safety.
        """
        if max_length_m is None:
            return route, gps_route

        try:
            max_length_m = float(max_length_m)
        except (TypeError, ValueError):
            return route, gps_route

        if max_length_m <= 0 or len(route) <= 2:
            return route, gps_route

        cumulative_length = 0.0
        end_index = 0
        for idx in range(1, len(route)):
            prev_loc = route[idx - 1][0].location
            curr_loc = route[idx][0].location
            segment_length = prev_loc.distance(curr_loc)
            if cumulative_length + segment_length > max_length_m:
                break
            cumulative_length += segment_length
            end_index = idx

        # Ensure minimum route size for callers that assume route[1] exists.
        if end_index < 1:
            end_index = 1

        truncated_route = route[:end_index + 1]
        if gps_route:
            truncated_gps_route = gps_route[:min(len(gps_route), end_index + 1)]
        else:
            truncated_gps_route = gps_route

        return truncated_route, truncated_gps_route

    def _get_route(self, config):
        """
        Gets the route from the configuration, interpolating it to the desired density,
        saving it to the CarlaDataProvider and sending it to the agent

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        - debug_mode: boolean to decide whether or not the route poitns are printed
        """

        # Prepare route's trajectory (interpolate and add the GPS route)
        # the default waypoints density is 1.0m
        self.gps_route, self.route = interpolate_trajectory(config.keypoints)

        if self.route_max_length_m is not None:
            self.route, self.gps_route = self._truncate_route_by_length_m(
                self.route,
                self.gps_route,
                self.route_max_length_m
            )

        # Legacy compatibility.
        if self.route_length_limit:
            self.route = self.route[:self.route_length_limit]
            if self.gps_route:
                self.gps_route = self.gps_route[:min(len(self.gps_route), len(self.route))]

        return self.route

    def _filter_scenarios(self, scenario_configs):
        """
        Given a list of scenarios, filters out does that don't make sense to be triggered,
        as they are either too far from the route or don't fit with the route shape

        Parameters:
        - scenario_configs: list of ScenarioConfiguration
        """
        new_scenarios_config = []
        for scenario_number, scenario_config in enumerate(scenario_configs):
            trigger_point = scenario_config.trigger_points[0]
            if not RouteParser.is_scenario_at_route(trigger_point, self.route):
                print("WARNING: Ignoring scenario '{}' as it is too far from the route".format(scenario_config.name))
                continue

            scenario_config.route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            new_scenarios_config.append(scenario_config)

        return new_scenarios_config

    def _spawn_ego_vehicle(self, traffic_manager=None):
        """Spawn the ego vehicle at the first waypoint of the route"""
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_ego_actor('vehicle.lincoln.mkz_2020',
                                                          elevate_transform,
                                                          rolename='hero', route=self.route[:])
        if not ego_vehicle:
            return

        # Keep reward calculations on the base route.
        CarlaDataProvider._ego_vehicle_route_reward = None
        CarlaDataProvider._ego_vehicle_route_reward_full = None

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(elevate_transform.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))
        self.world.tick(SIMULATOR_TICK_TIMEOUT_LIMIT)
        CarlaDataProvider.on_carla_tick()

        return ego_vehicle

    def _get_parking_slots(self, max_distance=100, route_step=10):
        """Spawn parked vehicles."""

        def is_close(slot_location):
            for i in range(0, len(self.route), route_step):
                route_transform = self.route[i][0]
                if route_transform.location.distance(slot_location) < max_distance:
                    return True
            return False

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for route_transform, _ in self.route:
            min_x = min(min_x, route_transform.location.x - max_distance)
            min_y = min(min_y, route_transform.location.y - max_distance)
            max_x = max(max_x, route_transform.location.x + max_distance)
            max_y = max(max_y, route_transform.location.y + max_distance)

        # Occupied parking locations
        occupied_parking_locations = []
        for scenario in self.list_scenarios:
            occupied_parking_locations.extend(scenario.get_parking_slots())

        available_parking_locations = []
        map_name = self.map.name.split('/')[-1]
        # Keep route-local filtering/spawn bookkeeping from mutating the
        # module-level Town12/Town13 catalogs across episodes.
        available_parking_locations = list(getattr(parked_vehicles, map_name, []))

        # Exclude parking slots that are too far from the route
        filtered_parking_locations = []
        for slot in available_parking_locations:
            slot_transform = carla.Transform(
                location=carla.Location(slot["location"][0], slot["location"][1], slot["location"][2]),
                rotation=carla.Rotation(slot["rotation"][0], slot["rotation"][1], slot["rotation"][2])
            )

            in_area = (min_x < slot_transform.location.x < max_x) and (min_y < slot_transform.location.y < max_y)
            close_to_route = is_close(slot_transform.location)
            if in_area and close_to_route:
                filtered_parking_locations.append(slot)

        self.available_parking_locations = filtered_parking_locations

    def spawn_parked_vehicles(self, ego_vehicle, max_scenario_distance=10):
        """Spawn parked vehicles."""
        def is_close(slot_location, ego_location):
            return slot_location.distance(ego_location) < self.PARKED_VEHICLES_INIT_THRESHOLD
        def is_free(slot_location):
            for occupied_slot in self.occupied_parking_locations:
                if slot_location.distance(occupied_slot) < max_scenario_distance:
                    return False
            return True

        new_parked_vehicles = []

        ego_location = CarlaDataProvider.get_location(ego_vehicle)
        if ego_location is None:
            return

        for slot in self.available_parking_locations:
            slot_transform = carla.Transform(
                location=carla.Location(slot["location"][0], slot["location"][1], slot["location"][2]),
                rotation=carla.Rotation(slot["rotation"][0], slot["rotation"][1], slot["rotation"][2])
            )

            # Add all vehicles that are close to the ego and in a free space
            if is_close(slot_transform.location, ego_location) and is_free(slot_transform.location):
                mesh_bp = CarlaDataProvider.get_world().get_blueprint_library().filter("static.prop.mesh")[0]
                mesh_bp.set_attribute("mesh_path", slot["mesh"])
                mesh_bp.set_attribute("scale", "0.9")
                new_parked_vehicles.append(carla.command.SpawnActor(mesh_bp, slot_transform))
                self.available_parking_locations.remove(slot)

        # Add the actors to _parked_ids
        for response in CarlaDataProvider.get_client().apply_batch_sync(new_parked_vehicles):
            if not response.error:
                self._parked_ids.append(response.actor_id)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, waypoints, vertical_shift, size, downsample=1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for i, w in enumerate(waypoints):
            if i % downsample != 0:
                continue

            wp = w[0].location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0)  # Green

            self.world.debug.draw_point(wp, size=size, color=color, life_time=self.timeout)

        self.world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=2*size,
                                    color=carla.Color(0, 0, 128), life_time=self.timeout)
        self.world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=2*size,
                                    color=carla.Color(128, 128, 128), life_time=self.timeout)

    def get_all_scenario_classes(self):
        """
        Searches through the 'scenarios' folder for all the Python classes
        """
        # Path of all scenario at "srunner/scenarios" folder
        scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(os.getenv('SCENARIO_RUNNER_ROOT', "./")))

        all_scenario_classes = {}

        for scenario_file in scenarios_list:

            # Get their module
            module_name = os.path.basename(scenario_file).split('.')[0]
            sys.path.insert(0, os.path.dirname(scenario_file))
            scenario_module = importlib.import_module(module_name)

            # And their members of type class
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                # TODO: Filter out any class that isn't a child of BasicScenario
                all_scenario_classes[member[0]] = member[1]

        return all_scenario_classes

    def build_scenarios(self, ego_vehicle, debug=False):
        """
        Initializes the class of all the scenarios that will be present in the route.
        If a class fails to be initialized, a warning is printed but the route execution isn't stopped
        """
        new_scenarios = []

        if self.all_scenario_classes is None:
            self.all_scenario_classes = self.get_all_scenario_classes()
        if self.ego_data is None:
            self.ego_data = ActorConfigurationData(ego_vehicle.type_id, ego_vehicle.get_transform(), 'hero')

        # Part 1. Check all scenarios that haven't been initialized, starting them if close enough to the ego vehicle
        name = 'NoScenario'
        for scenario_config in self.missing_scenario_configurations:
            scenario_config.ego_vehicles = [self.ego_data]
            scenario_config.route = self.route[:]

            try:
                scenario_class = self.all_scenario_classes[scenario_config.type]
                trigger_location = scenario_config.trigger_points[0].location

                ego_location = CarlaDataProvider.get_location(ego_vehicle)
                if ego_location is None:
                    continue

                # Only init scenarios that are close to ego
                if trigger_location.distance(ego_location) < self.INIT_THRESHOLD:
                    scenario_instance = scenario_class(self.world, [ego_vehicle], scenario_config, timeout=self.timeout)

                    # Add new scenarios to list
                    self.list_scenarios.append(scenario_instance)
                    new_scenarios.append(scenario_instance)
                    if self.length < 500:
                        name = scenario_config.type
                    self.missing_scenario_configurations.remove(scenario_config)

                    self.occupied_parking_locations.extend(scenario_instance.get_parking_slots())

                    if debug:
                        scenario_loc = scenario_config.trigger_points[0].location
                        debug_loc = self.map.get_waypoint(scenario_loc).transform.location + carla.Location(z=0.2)
                        self.world.debug.draw_point(
                            debug_loc, size=0.2, color=carla.Color(128, 0, 0), life_time=self.timeout
                        )
                        self.world.debug.draw_string(
                            debug_loc, str(scenario_config.name), draw_shadow=False,
                            color=carla.Color(0, 0, 128), life_time=self.timeout, persistent_lines=True
                        )

            except Exception as e:
                message = f"Skipping scenario '{scenario_config.name}' due to setup error: {e}"
                print(f"\033[93m{message}")
                if debug:
                    print(f"\n{traceback.format_exc()}")
                print("\033[0m", end="")
                raise RouteScenarioSetupError(message) from e
        if len(self.list_scenarios) <= 1:
            # assign a name for this route if there is only one scenario to be triggered
            self.scenario_name = name
        # Part 2. Add their behavior onto the route's behavior tree
        for scenario in new_scenarios:

            # Add behavior
            if scenario.behavior_tree is not None:
                self.behavior_node.add_child(scenario.behavior_tree)
                self.scenario_triggerer.add_blackboard(
                    [scenario.config.route_var_name, scenario.config.trigger_points[0].location]
                )

            # Add the criteria criteria
            scenario_criteria = scenario.get_criteria()
            if len(scenario_criteria) == 0:
                continue

            self.criteria_node.add_child(
                self._create_criterion_tree(scenario, scenario_criteria)
            )

    # pylint: enable=no-self-use
    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Creates a parallel behavior that runs all of the scenarios part of the route.
        These subbehaviors have had a trigger condition added so that they wait until
        the agent is close to their trigger point before activating.

        It also adds the BackgroundActivity scenario, which will be active throughout the whole route.
        This behavior never ends and the end condition is given by the RouteCompletionTest criterion.
        """
        scenario_trigger_distance = DIST_THRESHOLD  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(name="Route Behavior",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        self.behavior_node = behavior
        scenario_behaviors = []
        blackboard_list = []

        # Add the behavior that manages the scenario trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0], self.route, blackboard_list, scenario_trigger_distance)
        behavior.add_child(scenario_triggerer)  # Tick the ScenarioTriggerer before the scenarios

        # register var
        self.scenario_triggerer = scenario_triggerer

        # Add the Background Activity
        behavior.add_child(BackgroundBehavior(self.ego_vehicles[0], self.route, name="BackgroundActivity"))

        behavior.add_children(scenario_behaviors)
        return behavior

    def _create_test_criteria(self):
        """
        Create the criteria tree. It starts with some route criteria (which are always active),
        and adds the scenario specific ones, which will only be active during their scenario
        """
        criteria = py_trees.composites.Parallel(name="Criteria",
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        self.criteria_node = criteria

        # End condition
        criteria.add_child(RouteCompletionTest(self.ego_vehicles[0], route=self.route))

        # 'Normal' criteria
        criteria.add_child(OutsideRouteLanesTest(self.ego_vehicles[0], route=self.route))
        criteria.add_child(CollisionTest(self.ego_vehicles[0],name="CollisionTest"))
        criteria.add_child(RunningRedLightTest(self.ego_vehicles[0]))
        criteria.add_child(RunningStopTest(self.ego_vehicles[0]))
        criteria.add_child(MinimumSpeedRouteTest(self.ego_vehicles[0], self.route, checkpoints=4, name="MinSpeedTest"))

        # These stop the route early to save computational time
        criteria.add_child(InRouteTest(
            self.ego_vehicles[0], route=self.route, offroad_max=self.env_config.get('environment', {}).get('deviation_limit', 30), terminate_on_failure=True))
        criteria.add_child(ActorBlockedTest(
            self.ego_vehicles[0], min_speed=0.1, max_time=self.env_config.get('environment', {}).get('blocked_time_out_limit', 180.0), terminate_on_failure=True, name="AgentBlockedTest")
        )


        return criteria

    def _create_weather_behavior(self):
        """
        Create the weather behavior
        """
        if len(self.config.weather) == 1:
            return  # Just set the weather at the beginning and done
        return RouteWeatherBehavior(self.ego_vehicles[0], self.route, self.config.weather)

    def _create_lights_behavior(self):
        """
        Create the street lights behavior
        """
        return RouteLightsBehavior(self.ego_vehicles[0], 100)

    def _create_timeout_behavior(self):
        """
        Create the timeout behavior
        """
        return None

    def _initialize_environment(self, world):
        """
        Set the weather
        """
        # Set the appropriate weather conditions
        world.set_weather(self.config.weather[0][1])

    def _create_criterion_tree(self, scenario, criteria):
        """
        We can make use of the blackboard variables used by the behaviors themselves,
        as we already have an atomic that handles their (de)activation.
        The criteria will wait until that variable is active (the scenario has started),
        and will automatically stop when it deactivates (as the scenario has finished)
        """
        scenario_name = scenario.name
        var_name = scenario.config.route_var_name
        check_name = "WaitForBlackboardVariable: {}".format(var_name)

        criteria_tree = py_trees.composites.Sequence(name=scenario_name)
        criteria_tree.add_child(WaitForBlackboardVariable(var_name, True, False, name=check_name))

        scenario_criteria = py_trees.composites.Parallel(name=scenario_name,
                                                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        for criterion in criteria:
            scenario_criteria.add_child(criterion)
        scenario_criteria.add_child(WaitForBlackboardVariable(var_name, False, None, name=check_name))

        criteria_tree.add_child(scenario_criteria)
        criteria_tree.add_child(Idle())  # Avoid the indiviual criteria stopping the simulation
        return criteria_tree

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self._parked_ids])
        self.remove_all_actors()
        
    def attatch_sensor(self):
        cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x",str(512))
        cam_bp.set_attribute("image_size_y",str(512))
        cam_bp.set_attribute("fov",str(105))
        cam_location = carla.Location(2,0,1)
        cam_rotation = carla.Rotation(0,180,0)
        cam_transform = carla.Transform(cam_location,cam_rotation)
        ego_cam = self.world.spawn_actor(cam_bp,cam_transform,attach_to=self.ego_vehicles[0], attachment_type=carla.AttachmentType.Rigid)
        self.world.tick(SIMULATOR_TICK_TIMEOUT_LIMIT)
        ego_cam.listen(lambda image: image.save_to_disk((self.config.img_save_dir+'/%.6d.jpg') % image.frame))
    
    def compute_length(self):
        trans_list = self.route[:]
        length = 0.
        for index, _ in enumerate(trans_list[:-1]):
            if type(trans_list[index][0]) == str or type(trans_list[index+1][0]) == str:
                continue
            length += trans_list[index][0].location.distance(trans_list[index+1][0].location)
        return length
        pass
