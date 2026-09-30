from collections import OrderedDict
from dictor import dictor

import itertools
import copy
import math
import random

from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.checkpoint_tools import fetch_dict

from agents.navigation.global_route_planner import GlobalRoutePlanner

class RouteIndexer():
    
    def __init__(self, routes_file, repetitions, routes_subset, warmup=False, training=True):
        self.training = training
        self.SHORT_LENGTH = 6000
        self.SCENARIO_TYPE = {
        # 'VehicleOpensDoorTwoWays': [],
        # 'StaticCutIn': [],
        # 'InterurbanActorFlow': [],
        # 'MergerIntoSlowTraffic': [], 
        # 'Accident': [],
        # 'AccidentTwoWays': [],
        # 'CrossingBicycleFlow': [],
        # "ParkingExit": [], 
        # "BlockedIntersection": [], 
        # 'DynamicObjectCrossing': [],
        # 'HighwayExit': [], 
        # "EnterActorFlow": [],
        # 'HighwayCutIn': [], 
        # 'InterurbanAdvancedActorFlow': [],
        # 'MergerIntoSlowTrafficV2': [], 
        # "InvadingTurn": [], 
        # 'NoScenario': [],
        # 'ConstructionObstacleTwoWays': [],
        # 'ParkingCrossingPedestrian': [],
        # 'ParkingCutIn': [], 
        # 'ConstructionObstacle': [], 
        # 'ParkedObstacle': [], 
        # 'HazardAtSideLane': [],
        # 'YieldToEmergencyVehicle': [], 
        # 'PedestrianCrossing': [],
        # 'ConstructionObstacleTwoWays': [],
        # 'HazardAtSideLaneTwoWays': [],
        # 'ParkedObstacleTwoWays': [], 
        # 'HardBreakRoute': [],
        # "VanillaSignalizedTurnEncounterGreenLight": [],
        # "VanillaSignalizedTurnEncounterRedLight": [],
        # "VanillaNonSignalizedTurn": [],
        # "VanillaNonSignalizedTurnEncounterStopsign": [],
        # "SignalizedJunctionLeftTurn": [], 
        # "SignalizedJunctionRightTurn": [], 
        # "OppositeVehicleRunningRedLight": [],  
        # "NonSignalizedJunctionLeftTurn": [],
        # "NonSignalizedJunctionRightTurn": [], 
        # "OppositeVehicleTakingPriority": [],
        # "VehicleTurningRoute": [], 
        # "VehicleTurningRoutePedestrian": [],
        # "UnproperSpawn": [],
        # "SequentialLaneChange": [],
        # "T_Junction": [],
        "NonSignalizedJunctionLeftTurnEnterFlow": [], 
        # 'SignalizedJunctionLeftTurnEnterFlow': [],
        }
        self.VANILLA_SCENS = [
        "VanillaSignalizedTurnEncounterGreenLight", 
        "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurn", 
        "VanillaNonSignalizedTurnEncounterStopsign",
        "ObstacleEmerge"
        ]
        self._configs_dict = OrderedDict()
        self._configs_list = []
        self.warmup = warmup
        self.index = 0
        self.step = 0
        self.last = 0
        self.count = 0

        route_configurations = RouteParser.parse_routes_file(routes_file, routes_subset)
        self.total = len(route_configurations) * repetitions

        # self.total = 0
        self._warmup_config_indexs = []
        self._warmup_total = 0
        for i, config in enumerate(route_configurations):
            for repetition in range(repetitions):
                # self.total += 1
                config.index = i * repetitions + repetition
                config.repetition_index = repetition
                self._configs_list.append(copy.copy(config))
            if not config.scenario_configs:
                self._warmup_config_indexs.append(i)
                self._warmup_total += 1
                # self.SCENARIO_TYPE['NoScenario'].append(i)
            else:
                if len(config.scenario_configs)==1 and config.scenario_configs[0].type in self.VANILLA_SCENS:
                    self._warmup_config_indexs.append(i)
                    self._warmup_total += 1
                if len(config.scenario_configs)==1 and config.scenario_configs[0].type in self.SCENARIO_TYPE:
                    self.SCENARIO_TYPE[config.scenario_configs[0].type].append(i)
            # if (not config.scenario_configs) or\
            #     (len(config.scenario_configs)==1 and config.scenario_configs[0].type in self.VANILLA_SCENS):
            #     self._warmup_config_indexs.append(i)
            #     self._warmup_total += 1
        if not self._warmup_config_indexs and warmup:
            raise NotImplementedError("There is no warmup configs")
        self._warmup_config_indexs = itertools.cycle(self._warmup_config_indexs)
        for key, value in self.SCENARIO_TYPE.copy().items():
            if not value:
                del self.SCENARIO_TYPE[key]
                continue
            self.SCENARIO_TYPE[key] = itertools.cycle(value)
        if not self.SCENARIO_TYPE and warmup:
            raise ValueError("There is no valid configs")
        self.scenraio_type_list = itertools.cycle(list(self.SCENARIO_TYPE))
        if len(self.SCENARIO_TYPE) == 0:
            self.current_scenario_type = "ParkingExit"
        else:
            self.current_scenario_type = next(self.scenraio_type_list)
        self.get_trained = 0
        self.short_route = True
        # self.current_scenario_index = 0
        self.route = []
        pass
    
    def disable_short_route(self):
        self.short_route = False
    
    def reset(self):
        self.index = 0
    
    def reset_warmup(self):
        self._warmup_indexs_index = 0
    
    def peek(self):
        return self.index < self.total
    
    def get_next_scenario_type(self):
        self.current_scenario_type = next(self.scenraio_type_list)
        self.get_trained = 0

    def get_next_config(self):
        config = self._configs_list[self.index]
        self.index += 1
        return config
    
    def get_index_config(self, index):
        if self.index >= self.total:
            self.reset()
        # if self.warmup:
        #     config = self._configs_list[index]
        #     if len(config.scenario_configs) < 1 or\
        #         (len(config.scenario_configs)==1 and config.scenario_configs[0].type in self.VANILLA_SCENS):
        #             return config
        #     else:
        #         return self.get_index_config(index+1)
        # else:
        config = self._configs_list[index]
        # world = client.load_world(config.town)
        # print("successfully load the world!", config.town)
        # self.route = self.generate_route(config.keypoints, world)
        # if self.short_route:
        #     count = 0
        #     while self.compute_length() > self.SHORT_LENGTH:
        #         count += 1
        #         config = self._configs_list[index+count]
        #         world = client.load_world(config.town)
        #         print("successfully load the world!", config.town)
        #         self.route = self.generate_route(config.keypoints, world)
        #         if count > 50:
        #             raise ValueError("No Short route!")
        #     return config, world
        return config
    
    def get_index(self):
        if self.index >= self.total:
            self.reset()
        if not self.training:
            self.index += 1
            return self.index-1
        if self.warmup:
            return next(self._warmup_config_indexs)
        # elif self.short_route:
        #     count = 0
        #     while True:
        #         count += 1
        #         res = next(self.SCENARIO_TYPE[self.current_scenario_type])
        #         lens, route = self.compute_length(self._configs_list[res].keypoints[:])
        #         if lens <= self.SHORT_LENGTH:
        #             break
        #         if count >= 50:
        #             raise ValueError("No short routes!")
        #     return res
        else:
            # self.index += 1
            # return self.index-1
            if self.count >= 2:
                self.count = 0
                self.current_scenario_type = next(self.scenraio_type_list)
            self.count += 1
            return next(self.SCENARIO_TYPE[self.current_scenario_type])
    
    # def get_index(self):
    #     if self.index >= self.total:
    #         self.reset()
    #     if self.warmup:
    #         return next(self._warmup_config_indexs)
    #     else:
    #         self.index += 1
    #         return (self.index-1)
    
    def get_length(self):
        return len(self._configs_list)

    def validate_and_resume(self, endpoint):
        """
        Validates the endpoint by comparing several of its values with the current running routes.
        If all checks pass, the simulation starts from the last route.
        Otherwise, the resume is canceled, and the leaderboard goes back to normal behavior
        """
        data = fetch_dict(endpoint)
        if not data:
            print('Problem reading checkpoint. Found no data')
            return False

        entry_status = dictor(data, 'entry_status')
        if not entry_status:
            print("Problem reading checkpoint. Given checkpoint is malformed")
            return False
        if entry_status == "Invalid":
            print("Problem reading checkpoint. The 'entry_status' is 'Invalid'")
            return False

        checkpoint_dict = dictor(data, '_checkpoint')
        if not checkpoint_dict or 'progress' not in checkpoint_dict:
            print("Problem reading checkpoint. Given endpoint is malformed")
            return False

        progress = checkpoint_dict['progress']
        if progress[1] != self.total:
            print("Problem reading checkpoint. Endpoint's amount of routes does not match the given one")
            return False

        route_data = dictor(checkpoint_dict, 'records')

        check_index = 0
        while check_index < progress[0]:
            route_id = self._configs_list[check_index].name
            route_id += "_rep" + str(self._configs_list[check_index].repetition_index)
            checkpoint_route_id = route_data[check_index]['route_id']

            if route_id != checkpoint_route_id:
                print("Problem reading checkpoint. Checkpoint routes don't match the current ones")
                return False

            check_index += 1

        self.index = max(0, progress[0] - 1)  # Resume means something went wrong, repeat the last route
        return True

    def compute_length(self):
        if len(self.route) < 1:
            return 10000
        length = 0.
        for index, _ in enumerate(self.route[:-1]):
            if type(self.route[index][0]) == str or type(self.route[index+1][0]) == str:
                continue
            length += self.compute_2d_distance(self.route[index][0].location, 
                                               self.route[index+1][0].location)
        return length
        pass
    
    def compute_2d_distance(self, loc1, loc2):
        return math.sqrt((loc1.x-loc2.x)**2+(loc1.y-loc2.y)**2)

    def generate_route(self, waypoints_trajectory, world, hop_resolution=1.0):

        grp = GlobalRoutePlanner(world.get_map(), hop_resolution)
        route = []
        for i in range(len(waypoints_trajectory) - 1):
            waypoint = waypoints_trajectory[i]
            waypoint_next = waypoints_trajectory[i + 1]
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)
            for wp, connection in interpolated_trace:
                route.append((wp.transform, connection))
        return route

