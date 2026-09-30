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
    
    def __init__(self, routes_file, repetitions, routes_subset, warmup=False, training=True, collect_data=None, all_data=[], longroute_file=None):
        if collect_data:
            self.collect_data = True
        else:
            self.collect_data = False
        
        self.longroute_file = longroute_file
        self.training = training
        self.short_route = True
        self.SHORT_LENGTH = 300
        self.SCENARIO_TYPE = {
                'NoScenario' :[],
                'DynamicObjectCrossing':[],
                'ParkingCrossingPedestrian':[],
                'ParkingCutIn' :[], 
                'ConstructionObstacle' :[], 
                'Accident':[],
                'ParkedObstacle':[], 
                'HazardAtSideLane':[],
                'YieldToEmergencyVehicle':[], 
                'ConstructionObstacleTwoWays':[],
                'AccidentTwoWays':[], 
                'HazardAtSideLaneTwoWays':[],
                'ParkedObstacleTwoWays':[], 
                'VehicleOpensDoorTwoWays':[], 
                'StaticCutIn':[],
                "ParkingExit":[], 
                'HardBreakRoute':[],
                "SignalizedJunctionLeftTurn":[], 
                "SignalizedJunctionRightTurn":[], 
                "OppositeVehicleRunningRedLight":[],  
                "NonSignalizedJunctionLeftTurn":[],
                "NonSignalizedJunctionRightTurn":[], 
                "OppositeVehicleTakingPriority":[],
                "VehicleTurningRoute":[], 
                "VehicleTurningRoutePedestrian":[],
                "EnterActorFlow":[],
                "BlockedIntersection":[], 
                'HighwayExit':[], 
                'InterurbanActorFlow':[],
                'MergerIntoSlowTraffic':[], 
                'HighwayCutIn':[], 
                'InterurbanAdvancedActorFlow':[],
                'MergerIntoSlowTrafficV2':[], 
                "InvadingTurn":[], 
                'CrossingBicycleFlow':[],
                'PedestrianCrossing':[],
                "VanillaSignalizedTurnEncounterGreenLight":[],
                "VanillaSignalizedTurnEncounterRedLight":[],
                "VanillaNonSignalizedTurn":[],
                "VanillaNonSignalizedTurnEncounterStopsign":[],
                "VanillaNonSignalizedTurnEncounterStopsignLong":[],
                "VanillaSignalizedTurnEncounterGreenLightLong": [], 
                "VanillaSignalizedTurnEncounterRedLightLong": [],
                # "UnproperSpawn":[],
                "T_Junction":[],
                "SequentialLaneChange":[],
                "NonSignalizedJunctionLeftTurnEnterFlow":[], 
                'SignalizedJunctionLeftTurnEnterFlow':[],
                'ControlLoss':[]
        }
        self.VANILLA_SCENS = [
        "Normal",
        "VanillaSignalizedTurnEncounterGreenLight", 
        "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurn", 
        "VanillaNonSignalizedTurnEncounterStopsign",
        # "ObstacleEmerge"
        ]
        self._configs_dict = OrderedDict()
        self._configs_list = []
        self.warmup = warmup
        self.index = 0
        self.step = 0
        self.last = 0
        self.count = 0

        route_configurations = RouteParser.parse_routes_file(routes_file, routes_subset)
        
        # self.total = 0
        self._warmup_config_indexs = []
        self._warmup_total = 0
        for i, config in enumerate(route_configurations):
            if config.name.split('_')[1] in all_data:
                continue
            for repetition in range(repetitions):
                # self.total += 1
                config.index = i * repetitions + repetition
                config.repetition_index = repetition
                self._configs_list.append(copy.copy(config))
            if not config.scenario_configs:
                self._warmup_config_indexs.append(i)
                self._warmup_total += 1
                if 'NoScenario' in self.SCENARIO_TYPE:
                    self.SCENARIO_TYPE['NoScenario'].append(i)
            else:
                if len(config.scenario_configs)==1 and config.scenario_configs[0].type in self.VANILLA_SCENS:
                    self._warmup_config_indexs.append(i)
                    self._warmup_total += 1
                    # self.SCENARIO_TYPE['NoScenario'].append(i)
                if len(config.scenario_configs)==1 and config.scenario_configs[0].type in self.SCENARIO_TYPE:
                    self.SCENARIO_TYPE[config.scenario_configs[0].type].append(i)
                # if len(config.scenario_configs)==0 and config.scenario_configs[0].type=='Normal':
                #     self.SCENARIO_TYPE['Normal'].append(i)
            # if (not config.scenario_configs) or\
            #     (len(config.scenario_configs)==1 and config.scenario_configs[0].type in self.VANILLA_SCENS):
            #     self._warmup_config_indexs.append(i)
            #     self._warmup_total += 1
        if len(self._configs_list) <= 0:
            print("********** all routes have been collected! **********")
            raise RuntimeError("Collection finished")
        self.total = len(self._configs_list)
        if not self._warmup_config_indexs and warmup:
            raise NotImplementedError("There is no warmup configs")
        self._warmup_config_indexs = itertools.cycle(self._warmup_config_indexs)
        for key, value in self.SCENARIO_TYPE.copy().items():
            if not value:
                del self.SCENARIO_TYPE[key]
                continue
            # self.SCENARIO_TYPE[key] = itertools.cycle(value)
        if not self.SCENARIO_TYPE and warmup:
            raise ValueError("There is no valid configs")
        self.scenraio_type_list = itertools.cycle(list(self.SCENARIO_TYPE))
        if len(self.SCENARIO_TYPE) == 0:
            self.current_scenario_type = "ParkingExit"
        else:
            self.current_scenario_type = next(self.scenraio_type_list)
        self.get_trained = 0
        # self.current_scenario_index = 0
        self.route = []
        # for config in self.SCENARIO_TYPE["SignalizedJunctionLeftTurnEnterFlow"]:
        #     print(config)
        self.all_indexs = list(range(self.total))
        
        self.longroute_config_list = []
        if self.longroute_file is not None:
            longroute_configurations = RouteParser.parse_routes_file(self.longroute_file, routes_subset)
            for i, config in enumerate(longroute_configurations):
                if config.name.split('_')[1] in all_data:
                    continue
                for repetition in range(repetitions):
                # self.total += 1
                    config.index = i * repetitions + repetition
                    config.repetition_index = repetition
                    self.longroute_config_list.append(copy.copy(config))
            self.longroute_total = len(self.longroute_config_list)
            self.longroute_index = 0
        pass
    
    def disable_short_route(self):
        self.short_route = False
    
    def reset(self):
        self.index = 0
    
    def reset_longroute(self):
        self.longroute_index = 0
    
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
        """
        Return the route config for the given index.
        
        This method does not update self.index, so it has no side effects on
        sequential iteration.
        
        Args:
            index: Route index in _configs_list.
            
        Returns:
            The corresponding RouteScenarioConfiguration.
        """
        if self.longroute_file is not None:
            if self.longroute_index >= self.longroute_total:
                self.reset_longroute()
            self.longroute_index += 1
            return self.longroute_config_list[self.longroute_index-1]
        return self._configs_list[index]
    
    def get_index(self):
        candidate_scenario = ['ConstructionObstacleTwoWays',
                'AccidentTwoWays', 
                'HazardAtSideLaneTwoWays',
                'ParkedObstacleTwoWays', 
                'ConstructionObstacle', 
                'Accident',
                'HazardAtSideLane',
                "SequentialLaneChange"]
        if self.collect_data:
            if len(self.all_indexs) == 0:
                # self.all_indexs = list(range(self.total))
                print("********** all routes have been collected! **********")
                raise RuntimeError("Collection finished")
            res = random.choice(self.all_indexs)
            self.all_indexs.remove(res)
            return res
        if self.index >= self.total:
            self.reset()
        if self.longroute_file is not None and self.longroute_index >= self.longroute_total:
            self.reset_longroute()
        if not self.training or (not self.short_route):
            # self.index += 1
            # return self.index-1
            bound = 2
            if self.count >= bound:
                self.count = 0
                self.current_scenario_type = next(self.scenraio_type_list)
            self.count += 1
            # return next(self.SCENARIO_TYPE[self.current_scenario_type])
            return random.choice(self.SCENARIO_TYPE[self.current_scenario_type])
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
            # if self.current_scenario_type in ["NoScenario",
            #                                   "VehicleOpensDoorTwoWays",
            #                                   "T_Junction"]:
            #     bound = 6
            if self.current_scenario_type in candidate_scenario:
                bound = 4
            elif self.current_scenario_type == 'BlockedIntersection':
                bound = 3
            else:
                bound = 2
            if self.count >= bound:
                self.count = 0
                self.current_scenario_type = next(self.scenraio_type_list)
            self.count += 1
            # return next(self.SCENARIO_TYPE[self.current_scenario_type])
            return random.choice(self.SCENARIO_TYPE[self.current_scenario_type])
    
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
