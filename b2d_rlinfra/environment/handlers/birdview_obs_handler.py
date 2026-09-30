"""BEV-mask observation handler.

Rasterises the ego-centric bird's-eye-view (lanes, vehicles, pedestrians,
traffic lights, ...) into a stacked binary-mask tensor used as the
policy's visual input.
"""
import logging

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import numpy as np
import carla
import math

logger = logging.getLogger("Interface Wrapper")
from gymnasium import spaces

import copy
import cv2 as cv
import os
from collections import deque
from pathlib import Path
import h5py

import time
__layer__ = (2, "Environment")
    
def tint(color, factor):
    r, g, b = color
    r = int(r + (255-r) * factor)
    g = int(g + (255-g) * factor)
    b = int(b + (255-b) * factor)
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return (r, g, b)

def get_global_bbx(actor, bbx):
    if actor.is_alive:
        bbx.location = actor.get_transform().transform(bbx.location)
        bbx.rotation = actor.get_transform().rotation
        return bbx
    return None

# Distance thresholds are in metres.
def close_enough(a, b):
    c_distance = abs(a.location.x - b.location.x) < 1.0 \
            and abs(a.location.y - b.location.y) < 1.0 
    return c_distance
def walker_close_enough(a, b):
    c_distance = abs(a.location.x - b.location.x) < 0.5 \
            and abs(a.location.y - b.location.y) < 0.5 
    return c_distance

Z_AXIS_MIN_DISTANCE = 8
Z_AXIS_MAX_DISTANCE = 50
class BirdViewObsManager():
    """Build configurable ego-centric BEV image or mask observations."""
    
    ELEMENT_KEYS = [
        'road', 'lanes', 'navigation', 'ego_vehicle',
        'vehicles', 'emergency_vehicles', 'pedestrians', 'obstacles',
        'red_light', 'yellow_light', 'green_light', 'stop_signs'
    ]
    
    # Fixed channel order for neural-network inputs.
    CHANNEL_ORDER = [
        'road', 'lanes', 'navigation', 'ego_vehicle',
        'vehicles', 'emergency_vehicles', 'pedestrians', 'obstacles',
        'red_light', 'yellow_light', 'green_light', 'stop_signs'
    ]
    
    DEFAULT_COLORS = {
        'background': (46, 52, 54),
        'road': (46, 52, 54),
        'lanes': (255, 0, 255),
        'navigation': (136, 138, 133),
        'ego_vehicle': (255, 255, 255),
        'vehicles': (0, 0, 255),
        'emergency_vehicles': (255, 0, 0),
        'pedestrians': (0, 255, 255),
        'obstacles': (139, 0, 0),
        'red_light': (255, 0, 0),
        'yellow_light': (255, 255, 0),
        'green_light': (0, 255, 0),
        'stop_signs': (160, 160, 0),
    }
    
    @staticmethod
    def _hex_to_rgb(hex_color):
        """Convert a hex color string to an RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def __init__(self, config):
        """Initialize from the vector-observation config."""
        self.config = config
        
        self._enabled = config.get('enable', True)
        self._output_type = config.get('type', 'bev_mask')  # 'bev_mask' or 'bev_image'
        self._map_dir = config.get('map_dir', 'resources/maps')
        self._pixels_per_meter = config.get('pixels_per_meter', 5.0)
        
        mask_width_meters = config.get('mask_width', 32)
        self._width = int(mask_width_meters * self._pixels_per_meter)
        
        ego_to_bottom_meters = config.get('ego_to_bottom', mask_width_meters / 2)
        self._pixels_ev_to_bottom = int(ego_to_bottom_meters * self._pixels_per_meter)
        
        self._lane_style = config.get('lane_style', 'unified')
        
        self._scale_bbox = config.get('scale_bbox', 1.0)
        self._scale_mask_col = config.get('scale_mask_col', 1.1)
        
        self._elements_config = config.get('elements', {})
        self._enabled_elements = set(self._elements_config.keys())
        
        self._history_elements = {
            elem for elem, cfg in self._elements_config.items()
            if cfg.get('use_history', False)
        }
        
        self._use_history = 'history_index' in config and len(self._history_elements) > 0
        if self._use_history:
            hist_idx = config['history_index']
            self._history_index = [int(x) for x in hist_idx] if hasattr(hist_idx, '__iter__') else [int(hist_idx)]
            self._history_len = abs(min(self._history_index)) + 1
            self._history_queue = deque(maxlen=self._history_len)
        else:
            self._history_index = [-1]
            self._history_queue = deque(maxlen=1)
        
        # Actor-filtering distance in meters, matching the BEV side length.
        self._distance_threshold = np.ceil(self._width / self._pixels_per_meter)
        
        self._colors = self._parse_colors(config.get('colors', {}))
        
        self._channel_info = self._compute_channel_info()
        
        self._map_loaded = False
        self._loaded_town = None
        self._road = None
        self._lane_marking_all = None
        self._lane_marking_yellow_broken = None
        self._lane_marking_yellow_solid = None
        self._world_offset = None
    
    def _parse_colors(self, colors_config):
        """Merge configured colors with defaults."""
        colors = dict(self.DEFAULT_COLORS)
        
        for key, value in colors_config.items():
            if isinstance(value, str) and value.startswith('#'):
                colors[key] = self._hex_to_rgb(value)
            elif isinstance(value, (list, tuple)) and len(value) == 3:
                colors[key] = tuple(value)
        
        return colors
    
    def _compute_channel_info(self):
        """
        Precompute channel metadata in the fixed output order.
        
        Returns:
            list of dict: {'name': str, 'index': int, 'history_index': int list or None}
        """
        channels = []
        channel_idx = 0
        
        for element in self.CHANNEL_ORDER:
            if not self.is_element_enabled(element):
                continue
            
            if self.is_history_element(element):
                for hist_idx in self._history_index:
                    channels.append({
                        'name': element,
                        'channel_idx': channel_idx,
                        'history_index': hist_idx,
                        'description': f'{element} (t{hist_idx})'
                    })
                    channel_idx += 1
            else:
                channels.append({
                    'name': element,
                    'channel_idx': channel_idx,
                    'history_index': None,
                    'description': element
                })
                channel_idx += 1
        
        return channels

    def load_prerasterized_map(self, force: bool = False):
        """Load pre-rasterized map data and reuse it for the same town."""
        world = CarlaDataProvider.get_world()
        if world is None:
            raise RuntimeError("CarlaDataProvider world not initialized!")
        
        town_name = world.get_map().name.split('/')[-1]

        if not force and self._map_loaded and self._loaded_town == town_name:
            return

        maps_h5_path = os.path.join(self._map_dir, f'{town_name}.h5')
        
        if not os.path.exists(maps_h5_path):
            raise FileNotFoundError(f"Map file not found: {maps_h5_path}")

        self._road = None
        self._lane_marking_all = None
        self._lane_marking_yellow_broken = None
        self._lane_marking_yellow_solid = None

        with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
            self._road = np.array(hf['road'], dtype=np.uint8)
            self._lane_marking_all = np.array(hf['lane_marking_all'], dtype=np.uint8)
            self._lane_marking_yellow_broken = np.array(hf['lane_marking_yellow_broken'], dtype=np.uint8)
            self._lane_marking_yellow_solid = np.array(hf['lane_marking_yellow_solid'], dtype=np.uint8)
            self._world_offset = np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)
            
            saved_ppm = float(hf.attrs['pixels_per_meter'])
            if not np.isclose(self._pixels_per_meter, saved_ppm):
                raise ValueError(
                    f"pixels_per_meter mismatch: config={self._pixels_per_meter}, map={saved_ppm}"
                )
        
        self._road = self._road.swapaxes(0, 1)
        self._lane_marking_all = self._lane_marking_all.swapaxes(0, 1)
        self._lane_marking_yellow_broken = self._lane_marking_yellow_broken.swapaxes(0, 1)
        self._lane_marking_yellow_solid = self._lane_marking_yellow_solid.swapaxes(0, 1)
        
        self._loaded_town = town_name
        self._map_loaded = True
    
    def get_observation(self):
        """Build the current BEV observation."""
        if not self._enabled:
            return {}
        
        if not self._map_loaded:
            self.load_prerasterized_map()
        
        ego_info = self._get_ego_info()
        ev_loc = ego_info['location']
        ev_rot = ego_info['rotation']
        ev_transform = ego_info['transform']
        ev_bbox = ego_info['bbox']
        
        M_warp = self._get_warp_transform(ev_loc, ev_rot)
        
        actors_data = self._collect_actors(ev_loc, ego_info['world_bbox'])
        
        self._update_history(actors_data)
        
        static_masks = self._get_static_masks(ev_loc, ev_rot, M_warp)
        
        dynamic_masks = self._get_dynamic_masks(M_warp)
        if dynamic_masks.get('emergency_vehicles') and isinstance(dynamic_masks['emergency_vehicles'], list) and len(dynamic_masks['emergency_vehicles']) > 0 and dynamic_masks['emergency_vehicles'][-1].any():
            emergency_vehicles_in_vision = True
        else:
            emergency_vehicles_in_vision = False
        
        ego_mask = self._get_ego_mask(ev_transform, ev_bbox, M_warp)
        
        if self._output_type == 'bev_mask':
            mask_output = self._build_mask_output(static_masks, dynamic_masks, ego_mask)
            image = self._render_image(static_masks, dynamic_masks, ego_mask)
            result = {
                'bev_mask': mask_output,
                'bev_image': image,
                'emergency_vehicles_in_vision': emergency_vehicles_in_vision
            }
        else:
            image = self._render_image(static_masks, dynamic_masks, ego_mask)
            result = {
                'bev_image': image,
                'emergency_vehicles_in_vision': emergency_vehicles_in_vision
            }
        
        return result  

    def reset(self):
        """Clear history and refresh map data after environment reset."""
        self._history_queue.clear()
        self.load_prerasterized_map()
    
    def is_element_enabled(self, element_name):
        """Return whether an element layer is enabled."""
        return element_name in self._enabled_elements
    
    def is_history_element(self, element_name):
        """Return whether an element layer uses history frames."""
        return element_name in self._history_elements
    
    @property
    def num_channels(self):
        """Number of output mask channels."""
        return len(self._channel_info)
    
    @property
    def output_shape(self):
        """Output mask shape as ``(C, H, W)``."""
        return (self.num_channels, self._width, self._width)
    
    def get_channel_names(self):
        """Return channel names in output order."""
        return [ch['description'] for ch in self._channel_info]
    
    def get_mask_info(self):
        """
        Return BEV mask output metadata.
        
        Returns:
            dict: output shape and per-channel details.
        """
        total_channels = len(self._channel_info)
        
        return {
            'output_type': self._output_type,
            'output_shape': (total_channels, self._width, self._width),
            'total_channels': total_channels,
            'history_index': self._history_index if self._use_history else None,
            'channels': self._channel_info.copy()
        }
    
    def print_mask_info(self):
        """Log the BEV mask channel index table."""
        info = self.get_mask_info()
        
        lines = [
            "=" * 70,
            "BEV Mask Output Info",
            "=" * 70,
            f"Output Type: {info['output_type']}",
            f"Output Shape: {info['output_shape']}  (C, H, W)",
            f"Total Channels: {info['total_channels']}",
        ]
        if info['history_index']:
            lines.append(f"History Index: {info['history_index']}")
        lines.append("-" * 70)
        lines.append(f"{'Index':<8} {'Element':<20} {'History Frame':<15} {'Description'}")
        lines.append("-" * 70)
        
        for ch in info['channels']:
            idx = ch['channel_idx']
            name = ch['name']
            hist = f"t{ch['history_index']}" if ch['history_index'] is not None else "-"
            desc = ch['description']
            lines.append(f"{idx:<8} {name:<20} {hist:<15} {desc}")
        
        lines.append("=" * 70)
        lines.append("Usage: mask[channel_idx] to access specific channel")
        lines.append("Example: mask[0] -> first channel (usually 'road')")
        logger.info("\n".join(lines))

    def _get_ego_info(self):
        """Collect ego transform and bounding-box information."""
        ego_actor = CarlaDataProvider._ego_actor
        ev_transform = ego_actor.get_transform()
        ev_loc = ev_transform.location
        ev_rot = ev_transform.rotation
        ev_bbox = ego_actor.bounding_box
        
        # Copy bbox.location before transform(), which may mutate in place.
        bbox_loc_copy = carla.Location(ev_bbox.location.x, ev_bbox.location.y, ev_bbox.location.z)
        world_bbox_loc = ev_transform.transform(bbox_loc_copy)
        world_ev_bbox = carla.BoundingBox(world_bbox_loc, ev_bbox.extent)
        world_ev_bbox.rotation = ev_transform.rotation
        
        return {
            'transform': ev_transform,
            'location': ev_loc,
            'rotation': ev_rot,
            'bbox': ev_bbox,
            'world_bbox': world_ev_bbox
        }
    
    def _is_within_distance(self, bbox):
        """Return whether a bounding box is inside the BEV range."""
      
        if bbox is None:
            return False
        ev_loc = CarlaDataProvider._ego_actor.get_location()
        c_distance = (abs(ev_loc.x - bbox.location.x) < self._distance_threshold and
                      abs(ev_loc.y - bbox.location.y) < self._distance_threshold and
                      abs(ev_loc.z - bbox.location.z) < Z_AXIS_MIN_DISTANCE)
        c_ev = abs(ev_loc.x - bbox.location.x) < 1.0 and abs(ev_loc.y - bbox.location.y) < 1.0
        return c_distance and (not c_ev)
    
    def _collect_actors(self, ev_loc, ev_world_bbox):
        """Collect nearby actors needed for dynamic BEV layers."""
        world = CarlaDataProvider._world
        
        vehicle_bbox_list = (
            world.get_level_bbs(carla.CityObjectLabel.Car) +
            world.get_level_bbs(carla.CityObjectLabel.Bicycle) +
            world.get_level_bbs(carla.CityObjectLabel.Bus) +
            world.get_level_bbs(carla.CityObjectLabel.Motorcycle) +
            world.get_level_bbs(carla.CityObjectLabel.Train) +
            world.get_level_bbs(carla.CityObjectLabel.Truck)
        )
        walker_bbox_list = world.get_level_bbs(carla.CityObjectLabel.Pedestrians)
        
        all_actors_raw = CarlaDataProvider.get_all_actors()
        
        traffic_sign_actors = []
        traffic_light_actors = []
        dirt_actors = []
        
        for actor in all_actors_raw:
            try:
                if not actor.is_active:
                    continue
            except Exception:
                if not getattr(actor, 'is_alive', False):
                    continue
            if 'traffic.stop' in actor.type_id:
                traffic_sign_actors.append(actor)
            elif 'traffic_light' in actor.type_id:
                traffic_light_actors.append(actor)
            elif 'static.prop.dirt' in actor.type_id:
                dirt_actors.append(actor)
        
        criterium = lambda bbox: self._is_within_distance(bbox)
        scale = self._scale_bbox if self._scale_bbox else None
        
        vehicles, emergency_vehicles, door_lines, obstacle_vehicles = \
            self._get_surrounding_vehicle_actors(vehicle_bbox_list, criterium, scale)
        static_obstacles = self._get_surrounding_obstacle_actors(criterium, scale)
        pedestrians = self._get_surrounding_walker_actors(
            walker_bbox_list, criterium, ev_world_bbox, ev_loc, scale
        )
        tl_green, tl_yellow, tl_red = self._get_surrounding_trafficlight_actors(
            traffic_light_actors, criterium, scale
        )
        stop_signs = self._get_stop_signs(traffic_sign_actors, criterium, scale)
        dirts = self._get_surrounding_dirt(dirt_actors, criterium, scale)
        
        obstacles = obstacle_vehicles + static_obstacles
        
        return {
            'vehicles': vehicles,
            'emergency_vehicles': emergency_vehicles,
            'pedestrians': pedestrians,
            'obstacles': obstacles,
            'door_lines': door_lines,
            'tl_green': tl_green,
            'tl_yellow': tl_yellow,
            'tl_red': tl_red,
            'stop_signs': stop_signs,
            'dirts': dirts
        }

    def _get_static_masks(self, ev_loc, ev_rot, M_warp):
        """Generate static masks: road, lanes, and route."""
        masks = {}
        
        if self.check_map_size(self._road):
            road_map, pruned_M = self._get_local_map(self._road, ev_loc, ev_rot, get_affine=True)
            lane_map = self._get_local_map(self._lane_marking_all, ev_loc, ev_rot)
            lane_yellow_broken_map = self._get_local_map(self._lane_marking_yellow_broken, ev_loc, ev_rot)
            lane_yellow_solid_map = self._get_local_map(self._lane_marking_yellow_solid, ev_loc, ev_rot)
        else:
            road_map = self._road
            lane_map = self._lane_marking_all
            lane_yellow_broken_map = self._lane_marking_yellow_broken
            lane_yellow_solid_map = self._lane_marking_yellow_solid
            pruned_M = M_warp
        
        if self.is_element_enabled('road'):
            masks['road'] = cv.warpAffine(
                road_map, pruned_M, (self._width, self._width)
            ).astype(np.bool_)
        
        if self.is_element_enabled('lanes'):
            masks['lanes'] = cv.warpAffine(
                lane_map, pruned_M, (self._width, self._width)
            ).astype(np.bool_)
            
            if self._lane_style == 'detailed':
                masks['lanes_yellow_broken'] = cv.warpAffine(
                    lane_yellow_broken_map, pruned_M, (self._width, self._width)
                ).astype(np.bool_)
                masks['lanes_yellow_solid'] = cv.warpAffine(
                    lane_yellow_solid_map, pruned_M, (self._width, self._width)
                ).astype(np.bool_)
        
        if self.is_element_enabled('navigation'):
            route_mask = np.zeros([self._width, self._width], dtype=np.uint8)
            vehicle_route = CarlaDataProvider._ego_vehicle_route
            route_len = len(vehicle_route) if vehicle_route else 0
            if vehicle_route:
                route_in_pixel = np.array([
                    [self._world_to_pixel(wp.location)] 
                    for wp, _ in vehicle_route[:80]
                ])
                route_warped = cv.transform(route_in_pixel, M_warp)
                cv.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=6)
            
            masks['navigation'] = route_mask.astype(np.bool_)
        
        return masks
    
    def _get_local_map(self, ori_map, ev_loc, ev_rot, get_affine=False):
        """Crop a local map window around the ego vehicle."""
        height, width = ori_map.shape[:2]
        center = self._world_to_pixel(ev_loc)
        
        left = int(max(0, center[0] - self._width))
        right = int(min(width, center[0] + self._width))
        bottom = int(max(0, center[1] - self._width))
        top = int(min(height, center[1] + self._width))
        
        _map = ori_map[bottom:top, left:right]
        new_center = np.array([self._width, self._width])
        
        yaw = np.deg2rad(ev_rot.yaw)
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)])
        
        bottom_left = new_center - self._pixels_ev_to_bottom * forward_vec - 0.5 * self._width * right_vec
        top_left = new_center + (self._width - self._pixels_ev_to_bottom) * forward_vec - 0.5 * self._width * right_vec
        top_right = new_center + (self._width - self._pixels_ev_to_bottom) * forward_vec + 0.5 * self._width * right_vec
        
        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width-1], [0, 0], [self._width-1, 0]], dtype=np.float32)
        M = cv.getAffineTransform(src_pts, dst_pts)
        
        if get_affine:
            return _map, M
        return _map
    
    def _update_history(self, actors_data):
        """Append current actor data to the history queue."""
        self._history_queue.append(actors_data)
    
    def _get_dynamic_masks(self, M_warp):
        """Generate dynamic masks from the history queue."""
        masks = {}
        qsize = len(self._history_queue)
        
        if qsize == 0:
            return masks
        
        element_to_key = {
            'vehicles': 'vehicles',
            'emergency_vehicles': 'emergency_vehicles',
            'pedestrians': 'pedestrians',
            'obstacles': 'obstacles',
            'green_light': 'tl_green',
            'yellow_light': 'tl_yellow',
            'red_light': 'tl_red',
            'stop_signs': 'stop_signs',
        }
        
        for element, data_key in element_to_key.items():
            if not self.is_element_enabled(element):
                continue
            
            if self.is_history_element(element):
                history_masks = []
                for idx in self._history_index:
                    actual_idx = max(idx, -qsize)
                    actors_data = self._history_queue[actual_idx]
                    actor_list = actors_data.get(data_key, [])
                    
                    if element in ['green_light', 'yellow_light', 'red_light']:
                        mask = self._get_mask_from_stopline_vtx(actor_list, M_warp)
                    else:
                        mask = self._get_mask_from_actor_list(actor_list, M_warp)
                    
                    history_masks.append(mask)
                
                masks[element] = history_masks
            else:
                actors_data = self._history_queue[-1]
                actor_list = actors_data.get(data_key, [])
                
                if element in ['green_light', 'yellow_light', 'red_light']:
                    masks[element] = self._get_mask_from_stopline_vtx(actor_list, M_warp)
                else:
                    masks[element] = self._get_mask_from_actor_list(actor_list, M_warp)
        
        # Door lines are rendered as obstacles.
        if self.is_element_enabled('obstacles') and self.is_history_element('obstacles'):
            for i, idx in enumerate(self._history_index):
                actual_idx = max(idx, -qsize)
                actors_data = self._history_queue[actual_idx]
                door_mask = self._get_mask_from_door_vtx(actors_data.get('door_lines', []), M_warp)
                masks['obstacles'][i] = masks['obstacles'][i] | door_mask
        
        return masks
    
    def _get_ego_mask(self, ev_transform, ev_bbox, M_warp):
        """Generate the ego-vehicle mask."""
        return self._get_mask_from_actor_list(
            [(ev_transform, ev_bbox.location, ev_bbox.extent)], M_warp
        )

    def _render_image(self, static_masks, dynamic_masks, ego_mask):
        """Render a color BEV image."""
        image = np.zeros([self._width, self._width, 3], dtype=np.uint8)
        image[:] = self._colors['background']
        
        if 'road' in static_masks:
            image[static_masks['road']] = self._colors['road']
        if 'navigation' in static_masks:
            image[static_masks['navigation']] = self._colors['navigation']
        if 'lanes' in static_masks:
            image[static_masks['lanes']] = self._colors['lanes']
        
        h_len = len(self._history_index) - 1
        
        def render_history_masks(masks, color):
            if isinstance(masks, list):
                for i, mask in enumerate(masks):
                    image[mask] = tint(color, (h_len - i) * 0.2)
            else:
                image[masks] = color
        
        # Later layers overwrite earlier ones.
        dynamic_render_order = [
            'stop_signs', 'green_light', 'yellow_light', 'red_light',
            'obstacles', 'emergency_vehicles', 'vehicles', 'pedestrians'
        ]
        
        for element in dynamic_render_order:
            if element in dynamic_masks:
                render_history_masks(dynamic_masks[element], self._colors.get(element, (255, 255, 255)))
        
        image[ego_mask] = self._colors['ego_vehicle']
        
        return image

    def _build_mask_output(self, static_masks, dynamic_masks, ego_mask):
        """
        Stack the masks with the given channel order
        
        Returns:
            np.ndarray: shape (C, H, W) 
        """

        all_masks = {
            **static_masks,
            'ego_vehicle': ego_mask,
            **dynamic_masks
        }
        
        channels = []
        for ch_info in self._channel_info:
            element = ch_info['name']
            hist_idx = ch_info['history_index']
            
            if element not in all_masks:
                channels.append(np.zeros((self._width, self._width), dtype=np.uint8))
                continue
            
            mask_data = all_masks[element]
            
            if hist_idx is not None:
                frame_idx = self._history_index.index(hist_idx)
                if isinstance(mask_data, list):
                    channels.append(mask_data[frame_idx].astype(np.uint8))
                else:
                    channels.append(mask_data[frame_idx].astype(np.uint8))
            else:
                channels.append(mask_data.astype(np.uint8))
        
        return np.stack(channels, axis=0)
    
    @staticmethod
    def _get_stop_signs(actor_list, criterium, scale=None):
        stops = []
        for actor in actor_list:
            bbox = get_global_bbx(actor, actor.bounding_box)
            is_within_distance = criterium(bbox)
            if is_within_distance:         
                if 'traffic.stop' in actor.type_id:
                    bb_loc = carla.Location(actor.trigger_volume.location)
                    bb_ext = carla.Vector3D(actor.trigger_volume.extent)
                    bb_ext.x = max(bb_ext.x, bb_ext.y)
                    bb_ext.y = max(bb_ext.x, bb_ext.y)
                    if scale is not None:
                        bb_ext.x = scale * bb_ext.x
                        bb_ext.y = scale * bb_ext.y  
                    trans = actor.get_transform()
                    stops.append((carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext))
        return stops

    @staticmethod
    def _get_surrounding_dirt(actor_list, criterium, scale=None):
        dirts = []
        for actor in actor_list:
            bbox = get_global_bbx(actor, actor.bounding_box)
            is_within_distance = criterium(bbox)
            if is_within_distance:
                if 'static.prop.dirt' in actor.type_id:
                    bb_loc = carla.Location(actor.bounding_box.location)
                    bb_ext = carla.Vector3D(actor.bounding_box.extent)
                    bb_ext.x = max(bb_ext.x, bb_ext.y)
                    bb_ext.y = max(bb_ext.x, bb_ext.y)
                    if scale is not None:
                        bb_ext.x = scale * bb_ext.x
                        bb_ext.y = scale * bb_ext.y  
                    trans = actor.get_transform()
                    dirts.append((carla.Transform(trans.location, trans.rotation), bb_loc, bb_ext))
        return dirts

    def _get_mask_from_stopline_vtx(self, stopline_vtx, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        if not stopline_vtx:
            return mask.astype(np.bool_)
        for sp_locs in stopline_vtx:
            stopline_in_pixel = np.array([[self._world_to_pixel(x)] for x in sp_locs])
            stopline_warped = cv.transform(stopline_in_pixel, M_warp)
            stopline_pts = np.round(stopline_warped[:, 0]).astype(np.int32)
            cv.line(mask, tuple(stopline_pts[0].tolist()), tuple(stopline_pts[1].tolist()),
                    color=1, thickness=6)
        return mask.astype(np.bool_)
    
    def _get_mask_from_door_vtx(self, door_lines, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        if not door_lines: 
            return mask.astype(np.bool_)
        lines_to_plot = []
        to_pix = lambda location_pair: cv.transform(np.array([[self._world_to_pixel(location_pair[0])],[self._world_to_pixel(location_pair[1])]]), M_warp)
        for pair in door_lines:
            lines_to_plot.append(to_pix(pair))
        for pair in lines_to_plot:
            cv.polylines(mask, [np.round(pair).astype(np.int32)], False, 1, thickness=2)
        return mask.astype(np.bool_)

    def _get_mask_from_actor_list(self, actor_list, M_warp):
        mask = np.zeros([self._width, self._width], dtype=np.uint8)
        if not actor_list: 
            return mask.astype(np.bool_)
        for actor_transform, bb_loc, bb_ext in actor_list:

            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners_in_pixel = np.array([[self._world_to_pixel(corner)] for corner in corners])
            corners_warped = cv.transform(corners_in_pixel, M_warp)

            cv.fillConvexPoly(mask, np.round(corners_warped).astype(np.int32), 1)
        return mask.astype(bool)
    
    @staticmethod
    def _get_surrounding_trafficlight_actors(actor_list, criterium, scale=None):
        tl_green_stopline_vertices = []
        tl_yellow_stopline_vertices = []
        tl_red_stopline_vertices = []
        for actor in actor_list:
            bbox = get_global_bbx(actor, actor.bounding_box)
            is_within_distance = criterium(bbox)
            if is_within_distance:
                base_transform = actor.get_transform()
                tv_loc = actor.trigger_volume.location
                tv_ext = actor.trigger_volume.extent 
                x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)
                area = []
                for x in x_values:
                    point_location = base_transform.transform(tv_loc + carla.Location(x=x)) 
                    area.append(point_location)
                ini_wps = []
                for pt in area:
                    wpx = CarlaDataProvider._map.get_waypoint(pt)
                    # As x_values are arranged in order, only the last one has to be checked
                    if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                        ini_wps.append(wpx)    
                tl_green_stopline_vertice = []
                tl_yellow_stopline_vertice = []
                tl_red_stopline_vertice = []
                for wpx in ini_wps:
                    # Below: just use trigger volume, otherwise it's on the zebra lines.

                    while not wpx.is_intersection:
                        next_wp = wpx.next(0.5)[0]
                        if next_wp and not next_wp.is_intersection:
                            wpx = next_wp
                        else:
                            break
                
                    vec_forward = wpx.transform.get_forward_vector()
                    vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

                    loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
                    loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
                    if actor.state == carla.TrafficLightState.Green:
                        tl_green_stopline_vertice.append([loc_left, loc_right])
                        continue
                    if actor.state == carla.TrafficLightState.Red:
                        tl_red_stopline_vertice.append([loc_left, loc_right])
                        continue
                    if actor.state == carla.TrafficLightState.Yellow:
                        tl_yellow_stopline_vertice.append([loc_left, loc_right])
                        continue
                if tl_green_stopline_vertice:
                    tl_green_stopline_vertices += tl_green_stopline_vertice 
                if tl_red_stopline_vertice:
                    tl_red_stopline_vertices += tl_red_stopline_vertice
                if tl_yellow_stopline_vertice:
                    tl_yellow_stopline_vertices += tl_yellow_stopline_vertice
            
        return tl_green_stopline_vertices, tl_yellow_stopline_vertices, tl_red_stopline_vertices

    @staticmethod
    def _get_surrounding_obstacle_actors(criterium, scale=None):
        actors = []
        if CarlaDataProvider._actor_obstacle_map is None:
            return actors
        for actor, bbox in CarlaDataProvider._actor_obstacle_map.items():
            if actor.is_alive:
                test = actor.get_transform()
                if np.abs(actor.bounding_box.location.x) > 1:
                    actor.bounding_box.location.x = 0.2 
                if np.abs(actor.bounding_box.location.y) > 1:
                    actor.bounding_box.location.y = 0.2 
                bbox = get_global_bbx(actor, actor.bounding_box)
                is_within_distance = criterium(bbox)
                if is_within_distance:
                    bb_loc = carla.Location()
                    bb_ext = carla.Vector3D(bbox.extent)
                    if scale is not None:
                        bb_ext = bb_ext * scale
                        bb_ext.x = max(bb_ext.x, 0.8)
                        bb_ext.y = max(bb_ext.y, 0.8)
                    actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
        return actors 
           
    @staticmethod
    def _get_surrounding_walker_actors(bbox_list, criterium, ev_bbox, ev_loc, scale=None):
        actors = []
        for bbox in bbox_list:
            is_within_distance = criterium(bbox)
            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if scale is not None:
                    bb_ext = bb_ext * scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)
                actors.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
                if CarlaDataProvider._actor_walker_blocker_map:
                    for actor in CarlaDataProvider._actor_walker_blocker_map:
                        if actor.is_alive:
                            actor_bbox = get_global_bbx(actor, actor.bounding_box)
                            if actor_bbox and walker_close_enough(actor_bbox, bbox):                                # z 
                                if np.abs(actor.get_location().z - ev_loc.z) > Z_AXIS_MAX_DISTANCE:
                                    actors.pop()              
        return actors    
    
    @staticmethod
    def _get_surrounding_vehicle_actors(bbox_list, criterium, vehicle_scale=None):
        vehicle_bbxs = []
        emergency_bbxs = []
        door_lines = []
        obstacle_vehicle_bbxs = []
        for bbox in bbox_list:
            is_within_distance = criterium(bbox)
            if is_within_distance:
                bb_loc = carla.Location()
                bb_ext = carla.Vector3D(bbox.extent)
                if vehicle_scale is not None:
                    bb_ext = bb_ext * vehicle_scale
                    bb_ext.x = max(bb_ext.x, 0.8)
                    bb_ext.y = max(bb_ext.y, 0.8)
                if CarlaDataProvider._actor_dooropen_map:              
                    for actor in CarlaDataProvider._actor_dooropen_map:
                        actor_bbox = get_global_bbx(actor, actor.bounding_box)
                        if actor.is_alive and close_enough(actor_bbox, bbox):
                            actor_transform = actor.get_transform()
                            if CarlaDataProvider._actor_dooropen_map[actor] == carla.VehicleDoor.FR:
                                head_right = carla.Transform(actor_transform.transform(carla.Location(bbox.extent.x, bbox.extent.y)),actor_transform.rotation).location
                                center_right = carla.Transform(actor_transform.transform(carla.Location(0, 2 * bbox.extent.y)),actor_transform.rotation).location
                                door_lines.append([head_right, center_right])
                            elif CarlaDataProvider._actor_dooropen_map[actor] == carla.VehicleDoor.FL:
                                head_left = carla.Transform(actor_transform.transform(carla.Location(bbox.extent.x, -bbox.extent.y)),actor_transform.rotation).location
                                center_left = carla.Transform(actor_transform.transform(carla.Location(0, -2 * bbox.extent.y)),actor_transform.rotation).location
                                door_lines.append([head_left, center_left])
                            else:
                                raise Exception("Unsupport door type: {}".format(CarlaDataProvider._actor_dooropen_map[actor]))
                vehicle_bbxs.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
                flag = 0 # flag to identify whether the vehicle has been poped. 
                if CarlaDataProvider._actor_emergency_map:
                    for actor in CarlaDataProvider._actor_emergency_map:
                        if actor.is_alive:
                            actor_bbox = get_global_bbx(actor, actor.bounding_box)
                            if close_enough(actor_bbox, bbox):
                                emergency_bbxs.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
                                vehicle_bbxs.pop()
                                flag = 1
                                break
                if CarlaDataProvider._actor_obstacle_map:
                    for actor in CarlaDataProvider._actor_obstacle_map:
                        if actor.is_alive:
                            if np.abs(actor.bounding_box.location.x) > 1:
                                actor.bounding_box.location.x = 0.2 
                            if np.abs(actor.bounding_box.location.y) > 1:
                                actor.bounding_box.location.y = 0.2 
                            actor_bbox = get_global_bbx(actor, actor.bounding_box)
                            if close_enough(actor_bbox, bbox):
                                if "vehicle" in actor.type_id: 
                                    obstacle_vehicle_bbxs.append((carla.Transform(bbox.location, bbox.rotation), bb_loc, bb_ext))
                                    if not flag:
                                        vehicle_bbxs.pop()
                                        flag = 1
                                    break
            
        return vehicle_bbxs, emergency_bbxs, door_lines, obstacle_vehicle_bbxs, 
                
    def _get_warp_transform(self, ev_loc, ev_rot):
        ev_loc_in_px = self._world_to_pixel(ev_loc)
        yaw = np.deg2rad(ev_rot.yaw)

        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])  
        right_vec = np.array([np.cos(yaw + 0.5*np.pi), np.sin(yaw + 0.5*np.pi)]) # same to np.array([-np.sin(yaw), np.cos(yaw)])

        bottom_left = ev_loc_in_px - self._pixels_ev_to_bottom * forward_vec - (0.5*self._width) * right_vec
        top_left = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec - (0.5*self._width) * right_vec
        top_right = ev_loc_in_px + (self._width-self._pixels_ev_to_bottom) * forward_vec + (0.5*self._width) * right_vec

        src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
        dst_pts = np.array([[0, self._width-1],
                            [0, 0],
                            [self._width-1, 0]], dtype=np.float32)
        return cv.getAffineTransform(src_pts, dst_pts)
    
    def check_map_size(self, _map):
        height, width = _map.shape[:2]
        if height > 3 * self._width or width > 3 * self._width:
            return True
        else:
            return False

    def _world_to_pixel(self, location, projective=False):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self._pixels_per_meter * (location.y - self._world_offset[1])

        if projective:
            p = np.array([x, y, 1], dtype=np.float32)
        else:
            p = np.array([x, y], dtype=np.float32)
        return p

    def _world_to_pixel_width(self, width):
        """Convert world units to pixels."""
        return self._pixels_per_meter * width
