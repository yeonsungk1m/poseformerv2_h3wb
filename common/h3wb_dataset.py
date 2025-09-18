# Copyright (c) 2024, PoseFormerV2 contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from collections import deque
from common.camera import normalize_screen_coordinates
from common.h36m_dataset import (
    h36m_cameras_extrinsic_params,
    h36m_cameras_intrinsic_params,
)
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton


JointSpec = Union[int, str]

DEFAULT_BODY_PARENTS = [-1, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
DEFAULT_LEFT_FOOT_PARENTS = [15, 15, 15]
DEFAULT_RIGHT_FOOT_PARENTS = [16, 16, 16]
DEFAULT_LEFT_HAND_PARENTS = [
    9,
    91,
    92,
    93,
    94,
    91,
    96,
    97,
    98,
    91,
    100,
    101,
    102,
    91,
    104,
    105,
    106,
    91,
    108,
    109,
    110,
]
DEFAULT_RIGHT_HAND_PARENTS = [
    10,
    112,
    113,
    114,
    115,
    112,
    117,
    118,
    119,
    112,
    121,
    122,
    123,
    112,
    125,
    126,
    127,
    112,
    129,
    130,
    131,
]


def _to_python_dict(data: Mapping, *keys: str) -> Optional[Mapping]:
    """Utility to safely extract nested dictionaries from a numpy npz file."""

    for key in keys:
        if key and key in data:
            raw_value = data[key]
            break
    else:
        return None

    # Values stored in numpy files are wrapped in object arrays. Convert them
    # to proper Python dictionaries for ease of use.
    if isinstance(raw_value, np.ndarray) and raw_value.dtype == object:
        return raw_value.item()
    return raw_value


def _resolve_dataset_paths(path: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(path, (list, tuple, set)):
        candidates = [str(p) for p in path]
    else:
        candidate_path = str(path)
        if os.path.isdir(candidate_path):
            candidates = sorted(glob.glob(os.path.join(candidate_path, '*.npz')))
        elif os.path.exists(candidate_path):
            candidates = [candidate_path]
        else:
            base_dir = os.path.dirname(candidate_path) or '.'
            candidates = []
            train_path = os.path.join(base_dir, 'h3wb_train.npz')
            if os.path.exists(train_path):
                candidates.append(train_path)
            test_pattern = os.path.join(base_dir, 'task*_test_3d.npz')
            candidates.extend(sorted(glob.glob(test_pattern)))
            if not candidates:
                candidates = sorted(glob.glob(os.path.join(base_dir, '*.npz')))
    resolved: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        if os.path.exists(candidate):
            resolved.append(candidate)
            seen.add(candidate)
    if not resolved:
        raise FileNotFoundError(f'Could not locate H3WB dataset files at {path}')
    return resolved


def _merge_nested_dict(target: Dict, source: Optional[Mapping]) -> None:
    if not source:
        return
    for subject, subject_data in source.items():
        subject_dict = target.setdefault(subject, {})
        for sequence, value in subject_data.items():
            if sequence in subject_dict:
                continue
            subject_dict[sequence] = value

def _to_ndarray(value) -> np.ndarray:
    array = value
    while isinstance(array, np.ndarray) and array.dtype == object and array.size == 1:
        array = array.item()
    return np.asarray(array)

def _reshape_pose(array: np.ndarray, dims: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        frames, flattened = arr.shape
        if flattened % dims != 0:
            raise ValueError(f'Pose array with shape {arr.shape} is incompatible with dimension {dims}.')
        arr = arr.reshape(frames, flattened // dims, dims)
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f'Expected pose array with 3 dimensions, got shape {arr.shape}.')

    if arr.shape[-1] < dims:
        raise ValueError(f'Pose array has insufficient components in the last dimension: {arr.shape}.')
    if arr.shape[-1] > dims:
        arr = arr[..., :dims]
    return arr.astype('float32')

def _sanitize_pose_entry(value):
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim == 0:
            return _sanitize_pose_entry(value.item())
        return [_sanitize_pose_entry(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {key: _sanitize_pose_entry(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_pose_entry(item) for item in value]
    return value


def _ingest_pose_annotations(
    container: Optional[Mapping],
    positions_world: Dict,
    positions_camera: Dict,
    positions_2d: Dict,
) -> None:
    if not isinstance(container, Mapping):
        return

    for subject, actions in container.items():
        if not isinstance(actions, Mapping):
            continue

        subject_world = positions_world.get(subject)
        if isinstance(subject_world, Mapping):
            if not isinstance(subject_world, dict):
                subject_world = dict(subject_world)
        else:
            subject_world = {}
        positions_world[subject] = subject_world

        subject_camera = positions_camera.get(subject)
        if isinstance(subject_camera, Mapping):
            if not isinstance(subject_camera, dict):
                subject_camera = dict(subject_camera)
        else:
            subject_camera = {}
        positions_camera[subject] = subject_camera

        subject_2d = positions_2d.get(subject)
        if isinstance(subject_2d, Mapping):
            if not isinstance(subject_2d, dict):
                subject_2d = dict(subject_2d)
        else:
            subject_2d = {}
        positions_2d[subject] = subject_2d

        for action, act_data in actions.items():
            if not isinstance(act_data, Mapping):
                continue

            global_pose = act_data.get('global_3d')
            if global_pose is not None and action not in subject_world:
                subject_world[action] = _reshape_pose(_sanitize_pose_entry(global_pose), dims=3)

            camera_pose = act_data.get('camera_3d')
            if camera_pose is not None and action not in subject_camera:
                subject_camera[action] = _sanitize_pose_entry(camera_pose)

            pose_2d = act_data.get('pose_2d')
            if pose_2d is not None and action not in subject_2d:
                subject_2d[action] = _sanitize_pose_entry(pose_2d)


def _prepare_position_container(container: Optional[Mapping]) -> Dict:
    if not isinstance(container, Mapping):
        return {}

    prepared: Dict = {}
    for subject, value in container.items():
        if isinstance(value, Mapping) and not isinstance(value, dict):
            prepared[subject] = dict(value)
        else:
            prepared[subject] = value
    return prepared

def _coerce_camera_sequence(
    subject: str,
    cameras: Dict[str, List[Dict]],
    value,
    dims: Optional[int],
    dtype: Optional[str] = None,
) -> List[np.ndarray]:
    if value is None:
        return []
    def _convert(item):
        array = _to_ndarray(item)
        if dims is not None:
            array = _reshape_pose(array, dims)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    if isinstance(value, Mapping):
        ordered: List[np.ndarray] = []
        used_keys = set()
        camera_list = cameras.get(subject, []) if cameras else []
        for idx, cam in enumerate(camera_list):
            candidates = [idx, str(idx)]
            cam_id = cam.get('id') if isinstance(cam, dict) else None
            if cam_id is not None:
                candidates.extend([cam_id, str(cam_id)])
            found = None
            for candidate in candidates:
                if candidate in value:
                    found = value[candidate]
                    used_keys.add(candidate)
                    break
            if found is not None:
                ordered.append(_convert(found))
        for key, item in value.items():
            if key in used_keys:
                continue
            ordered.append(_convert(item))
        return ordered

    if isinstance(value, np.ndarray):
        if dims is None:
            if value.ndim == 1:
                sequence = [value]
            elif value.ndim == 2:
                sequence = [value[i] for i in range(value.shape[0])]
            else:
                raise ValueError(f'Unexpected array shape {value.shape}.')
        else:
            if value.ndim == 3:
                sequence = [value]
            elif value.ndim == 4:
                sequence = [value[i] for i in range(value.shape[0])]
            else:
                raise ValueError(f'Unexpected pose array shape {value.shape}.')
    elif isinstance(value, (list, tuple)):
        sequence = list(value)
    else:
        sequence = [value]

    return [_convert(item) for item in sequence]
def _parents_from_legacy_metadata(metadata: Mapping) -> Optional[List[int]]:
    """Fallback for legacy H3WB archives without explicit skeleton parents."""

    face = metadata.get('face')
    left_hand = metadata.get('left_hand')
    right_hand = metadata.get('right_hand')

    if face is None or left_hand is None or right_hand is None:
        return None

    try:
        face_len = len(face)
        left_hand_len = len(left_hand)
        right_hand_len = len(right_hand)
    except TypeError:
        return None

    if left_hand_len != len(DEFAULT_LEFT_HAND_PARENTS):
        return None
    if right_hand_len != len(DEFAULT_RIGHT_HAND_PARENTS):
        return None

    offset = (
        len(DEFAULT_BODY_PARENTS)
        + len(DEFAULT_LEFT_FOOT_PARENTS)
        + len(DEFAULT_RIGHT_FOOT_PARENTS)
        + face_len
    )
    # The legacy annotations expect the left hand segment to start at index 91.
    if offset != 91:
        return None

    parents = (
        DEFAULT_BODY_PARENTS
        + DEFAULT_LEFT_FOOT_PARENTS
        + DEFAULT_RIGHT_FOOT_PARENTS
        + ([0] * face_len)
        + DEFAULT_LEFT_HAND_PARENTS
        + DEFAULT_RIGHT_HAND_PARENTS
    )

    expected_lengths: List[int] = []
    num_joints = metadata.get('num_joints')
    if isinstance(num_joints, (int, np.integer)):
        expected_lengths.append(int(num_joints))
    joint_names = metadata.get('joint_names')
    if isinstance(joint_names, (list, tuple)):
        expected_lengths.append(len(joint_names))

    if expected_lengths and any(length != len(parents) for length in expected_lengths):
        return None

    return parents
    
class Human3WBDataset(MocapDataset):
    """Loader for the Human3.6M Whole-Body (H3WB) dataset."""
    def __init__(self, path: Union[str, Sequence[str]], min_sequence_length: Optional[int] = None):
        self._min_sequence_length = int(min_sequence_length) if min_sequence_length else None
        paths = _resolve_dataset_paths(path)
        records = [self._load_archive(p) for p in paths]

        metadata = self._merge_metadata(records)
        skeleton_meta = metadata.get('skeleton', metadata)

        parents = self._build_parents(skeleton_meta)
        joints_left = skeleton_meta.get('joints_left') or skeleton_meta.get('left_side', [])
        joints_right = skeleton_meta.get('joints_right') or skeleton_meta.get('right_side', [])
        joints_left = self._resolve_indices(joints_left, skeleton_meta)
        joints_right = self._resolve_indices(joints_right, skeleton_meta)

        skeleton = Skeleton(parents=parents, joints_left=joints_left, joints_right=joints_right)

        fps = metadata.get('fps', 50)
        super().__init__(fps=fps, skeleton=skeleton)

        self._metadata = metadata
        self._joint_names = skeleton_meta.get('joint_names', [])
        self._joint_name_to_index = {name: idx for idx, name in enumerate(self._joint_names)}

        self._joint_groups = self._build_joint_groups(metadata)
        self._centering_map = self._build_centering_map(metadata)
        self._default_centering = metadata.get('default_centering', 'hip')

        self._subjects_train = self._normalize_subject_list(metadata.get('subjects_train'))
        self._subjects_test = self._normalize_subject_list(metadata.get('subjects_test'))
        self._keypoints_metadata = self._prepare_keypoints_metadata(metadata, skeleton)
        self._keypoints_normalized = bool(
            metadata.get('keypoints_normalized', metadata.get('keypoints_are_normalized', False))
        )
        self._cameras = self._build_cameras(metadata)        

        positions_world = {}
        positions_camera = {}
        positions_2d = {}
        sample_ids: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        frame_ids: Dict[str, Dict[str, np.ndarray]] = {}
        for record in records:
            _merge_nested_dict(positions_world, record['positions_world'])
            _merge_nested_dict(positions_camera, record['positions_3d'])
            _merge_nested_dict(positions_2d, record['positions_2d'])
            _merge_nested_dict(sample_ids, record.get('sample_ids'))
            _merge_nested_dict(frame_ids, record.get('frame_ids'))
       

        self._data = {}
        subjects = metadata.get('subjects')
        if not subjects:
            subjects_set = set()
            for container in (positions_world, positions_camera, positions_2d):
                if container:
                    subjects_set.update(container.keys())
            subjects = sorted(subjects_set)

        for subject in subjects:
            self._data[subject] = {}
            subject_world = positions_world[subject] if subject in positions_world else {}
            subject_camera = positions_camera[subject] if subject in positions_camera else {}
            subject_2d = positions_2d[subject] if subject in positions_2d else {}
            subject_samples = sample_ids[subject] if subject in sample_ids else {}
            subject_frames = frame_ids[subject] if subject in frame_ids else {}

            sequence_names = set(subject_world.keys()) | set(subject_camera.keys()) | set(subject_2d.keys())
            for seq in sorted(sequence_names):
                entry: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {
                    'cameras': self._cameras.get(subject, [])
                }
                if seq in subject_world:
                    entry['positions'] = np.asarray(subject_world[seq], dtype='float32')
                if seq in subject_camera:
                    entry['positions_3d'] = _coerce_camera_sequence(
                        subject, self._cameras, subject_camera[seq], dims=3, dtype='float32'
                    )
                if seq in subject_2d:
                    entry['positions_2d'] = _coerce_camera_sequence(
                        subject, self._cameras, subject_2d[seq], dims=2, dtype='float32'
                    )
                    entry['pose_2d'] = [pose.copy() for pose in entry['positions_2d']]
                if seq in subject_samples:
                    entry['sample_ids'] = _coerce_camera_sequence(
                        subject, self._cameras, subject_samples[seq], dims=None
                    )
                if seq in subject_frames:
                    entry['frame_ids'] = _to_ndarray(subject_frames[seq])
                self._data[subject][seq] = entry

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------
    def _load_archive(self, path: str) -> Dict:
        with np.load(path, allow_pickle=True) as data:
            metadata = _to_python_dict(data, 'metadata') or {}
            metadata = copy.deepcopy(metadata)

            record = {
                'path': path,
                'metadata': metadata,
                'positions_world': {},
                'positions_3d': {},
                'positions_2d': {},
                'sample_ids': {},
                'frame_ids': {},
            }
            legacy_world = _to_python_dict(data, 'positions_world', 'positions', 'pose_world')
            _merge_nested_dict(record['positions_world'], legacy_world)
            legacy_3d = _to_python_dict(data, 'positions_3d', 'pose_3d', 'poses_3d', 'pose3d')
            _merge_nested_dict(record['positions_3d'], legacy_3d)
            legacy_2d = _to_python_dict(data, 'positions_2d', 'pose_2d', 'poses_2d', 'pose2d')
            _merge_nested_dict(record['positions_2d'], legacy_2d)

            train_data = _to_python_dict(data, 'train_data')
            if train_data:
                (
                    world_train,
                    camera_train,
                    pose2d_train,
                    sample_train,
                    frames_train,
                ) = self._parse_whole_body_container(train_data)
                _merge_nested_dict(record['positions_world'], world_train)
                _merge_nested_dict(record['positions_3d'], camera_train)
                _merge_nested_dict(record['positions_2d'], pose2d_train)
                _merge_nested_dict(record['sample_ids'], sample_train)
                _merge_nested_dict(record['frame_ids'], frames_train)

                subjects_train = set(world_train.keys()) | set(camera_train.keys()) | set(pose2d_train.keys())
                if subjects_train:
                    existing = metadata.get('subjects_train')
                    if isinstance(existing, np.ndarray):
                        combined = set(existing.tolist())
                    elif isinstance(existing, (list, tuple, set)):
                        combined = set(existing)
                    else:
                        combined = set()
                    combined.update(subjects_train)
                    metadata['subjects_train'] = sorted(combined)
                if not metadata.get('split'):
                    metadata['split'] = 'train'

            test_data = _to_python_dict(data, 'test_data', 'data')
            if test_data:
                (
                    world_test,
                    camera_test,
                    pose2d_test,
                    sample_test,
                    frames_test,
                ) = self._parse_whole_body_container(test_data)
                _merge_nested_dict(record['positions_world'], world_test)
                _merge_nested_dict(record['positions_3d'], camera_test)
                _merge_nested_dict(record['positions_2d'], pose2d_test)
                _merge_nested_dict(record['sample_ids'], sample_test)
                _merge_nested_dict(record['frame_ids'], frames_test)

                subjects_test = set(world_test.keys()) | set(camera_test.keys()) | set(pose2d_test.keys())
                if subjects_test:
                    existing = metadata.get('subjects_test')
                    if isinstance(existing, np.ndarray):
                        combined = set(existing.tolist())
                    elif isinstance(existing, (list, tuple, set)):
                        combined = set(existing)
                    else:
                        combined = set()
                    combined.update(subjects_test)
                    metadata['subjects_test'] = sorted(combined)
                if not train_data and not metadata.get('split'):
                    metadata['split'] = 'test'

            return record

    def _parse_whole_body_container(
        self, container: Optional[Mapping]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, Dict[str, Dict[str, np.ndarray]]],
        Dict[str, Dict[str, Dict[str, np.ndarray]]],
        Dict[str, Dict[str, Dict[str, np.ndarray]]],
        Dict[str, Dict[str, np.ndarray]],
    ]:
        positions_world: Dict[str, Dict[str, np.ndarray]] = {}
        positions_3d: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        positions_2d: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        sample_ids: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        frame_ids: Dict[str, Dict[str, np.ndarray]] = {}

        if container is None:
            return positions_world, positions_3d, positions_2d, sample_ids, frame_ids

        for subject, actions in container.items():
            actions = self._ensure_mapping(actions)
            if not isinstance(actions, Mapping):
                continue

            subject_world: Dict[str, np.ndarray] = {}
            subject_camera: Dict[str, Dict[str, np.ndarray]] = {}
            subject_2d: Dict[str, Dict[str, np.ndarray]] = {}
            subject_samples: Dict[str, Dict[str, np.ndarray]] = {}
            subject_frames: Dict[str, np.ndarray] = {}

            for action_name, action_data in actions.items():
                action_data = self._ensure_mapping(action_data)
                if not isinstance(action_data, Mapping):
                    continue

                (
                    world_arr,
                    camera_dict,
                    pose2d_dict,
                    sample_dict,
                    frame_vector,
                ) = self._parse_whole_body_sequence(action_data)
                if world_arr is not None:
                    subject_world[action_name] = world_arr
                if camera_dict:
                    subject_camera[action_name] = camera_dict
                if pose2d_dict:
                    subject_2d[action_name] = pose2d_dict
                if sample_dict:
                    subject_samples[action_name] = sample_dict
                if frame_vector is not None:
                    subject_frames[action_name] = frame_vector

            if subject_world:
                positions_world[subject] = subject_world
            if subject_camera:
                positions_3d[subject] = subject_camera
            if subject_2d:
                positions_2d[subject] = subject_2d
            if subject_samples:
                sample_ids[subject] = subject_samples
            if subject_frames:
                frame_ids[subject] = subject_frames

        return positions_world, positions_3d, positions_2d, sample_ids, frame_ids

    def _ensure_mapping(self, value):
        while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
            value = value.item()
        return value

    def _parse_whole_body_sequence(
        self, action_data: Mapping
    ) -> Tuple[
        Optional[np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Optional[np.ndarray],
    ]:
        min_length = self._min_sequence_length if self._min_sequence_length else 0

        def _to_float_array(value) -> Optional[np.ndarray]:
            if value is None:
                return None
            array = value
            while isinstance(array, np.ndarray) and array.dtype == object and array.size == 1:
                array = array.item()
            array = np.asarray(array)
            array = array.squeeze()
            if array.dtype != np.float32:
                array = array.astype('float32', copy=False)
            return array

        def _to_array(value) -> Optional[np.ndarray]:
            if value is None:
                return None
            array = value
            while isinstance(array, np.ndarray) and array.dtype == object and array.size == 1:
                array = array.item()
            return np.asarray(array)

        world = _to_float_array(action_data.get('global_3d'))

        frame_count = world.shape[0] if isinstance(world, np.ndarray) and world.ndim >= 1 else None

        camera_dict: Dict[str, np.ndarray] = {}
        pose2d_dict: Dict[str, np.ndarray] = {}
        sample_dict: Dict[str, np.ndarray] = {}

        for cam_key, cam_value in action_data.items():
            if cam_key == 'global_3d':
                continue
            cam_value = self._ensure_mapping(cam_value)
            if not isinstance(cam_value, Mapping):
                continue

            camera = _to_float_array(cam_value.get('camera_3d') or cam_value.get('positions_3d'))
            if camera is not None:
                camera_dict[str(cam_key)] = camera
                if frame_count is None and camera.ndim >= 1:
                    frame_count = camera.shape[0]

            pose = _to_float_array(cam_value.get('pose_2d') or cam_value.get('positions_2d'))
            if pose is not None:
                pose2d_dict[str(cam_key)] = pose
                if frame_count is None and pose.ndim >= 1:
                    frame_count = pose.shape[0]

            samples = _to_array(cam_value.get('sample_id') or cam_value.get('sample_ids'))
            if samples is not None:
                sample_dict[str(cam_key)] = samples
                if frame_count is None and samples.ndim >= 1:
                    frame_count = samples.shape[0]

        if min_length and frame_count is not None and frame_count < min_length:
            return None, {}, {}, {}, None

        frame_vector = _to_array(
            action_data.get('frame_id')
            or action_data.get('frame_ids')
            or action_data.get('frames')
            or action_data.get('frame')
        )
        if frame_vector is not None and frame_count is not None and frame_vector.ndim >= 1:
            if frame_vector.shape[0] > frame_count:
                frame_vector = frame_vector[:frame_count]

        return world, camera_dict, pose2d_dict, sample_dict, frame_vector

    def _merge_metadata(self, records: List[Dict]) -> Dict:
        base: Dict = {}
        subjects_all: set = set()
        subjects_train: set = set()
        subjects_test: set = set()

        for record in records:
            meta = record.get('metadata') or {}
            if not base:
                base = copy.deepcopy(meta)
            else:
                self._combine_metadata_dict(base, meta)

            source_subjects = set()
            for key in ('positions_world', 'positions_3d', 'positions_2d'):
                container = record.get(key)
                if container:
                    source_subjects.update(container.keys())
            subjects_all.update(source_subjects)

            if meta.get('subjects_train'):
                subjects_train.update(meta['subjects_train'])
            if meta.get('subjects_test'):
                subjects_test.update(meta['subjects_test'])

            split = meta.get('split')
            if not split:
                filename = os.path.basename(record['path']).lower()
                if 'train' in filename and 'test' not in filename:
                    split = 'train'
                elif 'test' in filename:
                    split = 'test'
            if split == 'train':
                subjects_train.update(source_subjects)
            elif split == 'test':
                subjects_test.update(source_subjects)

        if subjects_all and not base.get('subjects'):
            base['subjects'] = sorted(subjects_all)
        if subjects_train and not base.get('subjects_train'):
            base['subjects_train'] = sorted(subjects_train)
        if subjects_test and not base.get('subjects_test'):
            base['subjects_test'] = sorted(subjects_test)

        return base

    def _combine_metadata_dict(self, base: Dict, incoming: Dict) -> None:
        for key, value in incoming.items():
            if value is None:
                continue
            if key in ('subjects', 'subjects_train', 'subjects_test'):
                combined = list(base.get(key, [])) + list(value)
                base[key] = list(dict.fromkeys(combined))
            elif key not in base:
                base[key] = copy.deepcopy(value)
            elif isinstance(base[key], dict) and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key not in base[key]:
                        base[key][sub_key] = copy.deepcopy(sub_value)

    def _build_parents(self, skeleton_meta: Mapping) -> List[int]:
        joint_names = list(skeleton_meta.get('joint_names', []))
        name_to_index = {name: idx for idx, name in enumerate(joint_names)}

        num_joints_meta = skeleton_meta.get('num_joints')
        joint_count = len(joint_names)
        if isinstance(num_joints_meta, (int, np.integer)):
            joint_count = max(joint_count, int(num_joints_meta))
        elif isinstance(num_joints_meta, (float, np.floating)) and not np.isnan(num_joints_meta):
            joint_count = max(joint_count, int(num_joints_meta))
        elif isinstance(num_joints_meta, str):
            try:
                joint_count = max(joint_count, int(num_joints_meta))
            except ValueError:
                pass

        def _coerce_index(value):
            if isinstance(value, np.generic):
                value = value.item()
            if value is None:
                return None
            if isinstance(value, (int, np.integer)):
                return int(value)
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    return None
                if float(value).is_integer():
                    return int(value)
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                lower = text.lower()
                if lower in {'none', 'null'}:
                    return None
                if lower == '-1':
                    return -1
                if text in name_to_index:
                    return name_to_index[text]
                try:
                    return int(text)
                except ValueError:
                    if lower == 'root' and 'root' not in name_to_index and joint_count:
                        return 0
                    raise KeyError(f'Unknown joint reference: {value!r}')
            raise KeyError(f'Unknown joint reference: {value!r}')

        def _ensure_length(parents_list: List[int]) -> List[int]:
            if joint_count and len(parents_list) < joint_count:
                parents_list = parents_list + [-1] * (joint_count - len(parents_list))
            return parents_list

        def _attempt_parent_pairs(pairs) -> Optional[List[int]]:
            resolved: Dict[int, int] = {}
            max_index = -1
            try:
                for child_spec, parent_spec in pairs:
                    child_idx = _coerce_index(child_spec)
                    if child_idx is None:
                        continue
                    parent_idx = _coerce_index(parent_spec)
                    parent_idx = -1 if parent_idx is None else parent_idx
                    resolved[child_idx] = parent_idx
                    max_index = max(max_index, child_idx)
                    if parent_idx >= 0:
                        max_index = max(max_index, parent_idx)
            except KeyError:
                return None
            if not resolved:
                return None
            count = joint_count or (max_index + 1 if max_index >= 0 else 0)
            if count <= 0:
                return None
            parents_list = [-1] * count
            for idx, parent in resolved.items():
                if idx >= len(parents_list):
                    parents_list.extend([-1] * (idx - len(parents_list) + 1))
                parents_list[idx] = parent
            return _ensure_length(parents_list)

        def _parents_from_scalar_list(values) -> Optional[List[int]]:
            if values is None:
                return None
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                return None
            if not values:
                return None
            scalars: List[int] = []
            try:
                for item in values:
                    parent_idx = _coerce_index(item)
                    scalars.append(-1 if parent_idx is None else parent_idx)
            except KeyError:
                return None
            return _ensure_length(scalars)

        def _parents_from_pairs(spec) -> Optional[List[int]]:
            if spec is None:
                return None
            if isinstance(spec, np.ndarray):
                spec = spec.tolist()
            if isinstance(spec, Mapping):
                direct = _attempt_parent_pairs(spec.items())
                if direct is not None:
                    return direct

                adjacency_pairs = []
                for key, value in spec.items():
                    if isinstance(value, Mapping):
                        lower = {str(k).lower(): v for k, v in value.items()}
                        child_candidate = lower.get('child') or lower.get('joint') or lower.get('index') or lower.get('name')
                        parent_candidate = lower.get('parent') or lower.get('attach_to') or lower.get('source') or lower.get('from')
                        if child_candidate is None and len(value) == 1:
                            child_candidate, parent_candidate = next(iter(value.items()))
                        if child_candidate is None:
                            child_candidate = key
                        adjacency_pairs.append((child_candidate, parent_candidate))
                    elif isinstance(value, (list, tuple, set)):
                        for child in value:
                            adjacency_pairs.append((child, key))
                    else:
                        adjacency_pairs.append((value, key))
                if adjacency_pairs:
                    attempted = _attempt_parent_pairs(adjacency_pairs)
                    if attempted is not None:
                        return attempted
                return None

            if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
                if not spec:
                    return None
                scalar_result = _parents_from_scalar_list(spec)
                if scalar_result is not None:
                    return scalar_result

                candidate_pairs = []
                for item in spec:
                    if isinstance(item, Mapping):
                        lower = {str(k).lower(): v for k, v in item.items()}
                        child_candidate = lower.get('child') or lower.get('joint') or lower.get('index') or lower.get('name')
                        parent_candidate = lower.get('parent') or lower.get('source') or lower.get('from') or lower.get('attach_to')
                        if child_candidate is None and len(item) == 1:
                            child_candidate, parent_candidate = next(iter(item.items()))
                        if child_candidate is None:
                            continue
                        candidate_pairs.append((child_candidate, parent_candidate))
                    else:
                        try:
                            child_spec, parent_spec = item
                        except (TypeError, ValueError):
                            return None
                        candidate_pairs.append((child_spec, parent_spec))
                if candidate_pairs:
                    return _attempt_parent_pairs(candidate_pairs)
            return None

        def _parents_from_parts(parts_spec) -> Optional[List[int]]:
            if not parts_spec:
                return None
            parents_combined: List[int] = []
            offset = 0
            for part in parts_spec:
                local_parents = part.get('parents')
                if local_parents is None:
                    return None
                local_list = list(local_parents)
                attach_to = part.get('attach_to', -1)
                try:
                    attach_to_idx = _coerce_index(attach_to) if attach_to not in (-1, None) else -1
                except KeyError:
                    return None
                for parent in local_list:
                    if parent == -1 and attach_to_idx != -1:
                        parents_combined.append(attach_to_idx)
                    elif parent == -1:
                        parents_combined.append(-1)
                    else:
                        parents_combined.append(parent + offset)
                offset += len(local_list)
            if not parents_combined:
                return None
            return _ensure_length(parents_combined)

        def _parents_from_edges(edges_spec) -> Optional[List[int]]:
            if edges_spec is None:
                return None
            if isinstance(edges_spec, np.ndarray):
                edges_spec = edges_spec.tolist()

            def _collect_edge_pairs(spec) -> List[Tuple[JointSpec, JointSpec]]:
                pairs: List[Tuple[JointSpec, JointSpec]] = []
                if isinstance(spec, Mapping):
                    for key, value in spec.items():
                        if isinstance(value, Mapping):
                            lower = {str(k).lower(): v for k, v in value.items()}
                            a = lower.get('a') or lower.get('from') or lower.get('parent') or lower.get('source')
                            b = lower.get('b') or lower.get('to') or lower.get('child') or lower.get('target')
                            if a is None and b is None and len(value) == 1:
                                a, b = next(iter(value.items()))
                            elif a is None or b is None:
                                continue
                            pairs.append((a, b))
                        elif isinstance(value, (list, tuple, set)):
                            for child in value:
                                pairs.append((key, child))
                        else:
                            pairs.append((key, value))
                elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
                    for item in spec:
                        if isinstance(item, Mapping):
                            lower = {str(k).lower(): v for k, v in item.items()}
                            a = lower.get('a') or lower.get('from') or lower.get('parent') or lower.get('source')
                            b = lower.get('b') or lower.get('to') or lower.get('child') or lower.get('target')
                            if a is None and b is None and len(item) == 1:
                                a, b = next(iter(item.items()))
                            elif a is None or b is None:
                                continue
                            pairs.append((a, b))
                        else:
                            try:
                                first, second = item
                            except (TypeError, ValueError):
                                continue
                            pairs.append((first, second))
                return pairs

            edge_pairs = _collect_edge_pairs(edges_spec)
            if not edge_pairs:
                return None
            adjacency: Dict[int, set] = {}
            nodes: set = set()
            try:
                for a_spec, b_spec in edge_pairs:
                    a_idx = _coerce_index(a_spec)
                    b_idx = _coerce_index(b_spec)
                    if a_idx is None or b_idx is None:
                        continue
                    adjacency.setdefault(a_idx, set()).add(b_idx)
                    adjacency.setdefault(b_idx, set()).add(a_idx)
                    nodes.add(a_idx)
                    nodes.add(b_idx)
            except KeyError:
                return None
            if not nodes:
                return None

            root_candidate = skeleton_meta.get('root') or skeleton_meta.get('root_joint') or skeleton_meta.get('root_index') or skeleton_meta.get('root_name')
            root_idx = None
            if root_candidate is not None:
                try:
                    root_idx = _coerce_index(root_candidate)
                except KeyError:
                    root_idx = None
            if root_idx is None or root_idx not in nodes:
                root_idx = min(nodes)

            parents_list: List[int] = [-1] * max(joint_count or 0, max(nodes) + 1)
            visited = {root_idx}
            queue = deque([root_idx])
            while queue:
                node = queue.popleft()
                for neighbour in adjacency.get(node, []):
                    if neighbour in visited:
                        continue
                    if neighbour >= len(parents_list):
                        parents_list.extend([-1] * (neighbour - len(parents_list) + 1))
                    parents_list[neighbour] = node
                    visited.add(neighbour)
                    queue.append(neighbour)

            for node in nodes:
                if node in visited:
                    continue
                visited.add(node)
                queue = deque([node])
                while queue:
                    current = queue.popleft()
                    for neighbour in adjacency.get(current, []):
                        if neighbour in visited:
                            continue
                        if neighbour >= len(parents_list):
                            parents_list.extend([-1] * (neighbour - len(parents_list) + 1))
                        parents_list[neighbour] = current
                        visited.add(neighbour)
                        queue.append(neighbour)

            return _ensure_length(parents_list)

        parents = _parents_from_pairs(skeleton_meta.get('parents'))
        if parents is not None:
            return parents

        for key in ('parent_map', 'parent_dict', 'parents_map', 'parent_ids', 'parent_indices'):
            parents = _parents_from_pairs(skeleton_meta.get(key))
            if parents is not None:
                return parents

        parents = _parents_from_parts(skeleton_meta.get('parts', []))
        if parents is not None:
            return parents
        for key in ('kinematic_tree', 'tree', 'edges', 'bones', 'links', 'connections'):
            parents = _parents_from_pairs(skeleton_meta.get(key))
            if parents is not None:
                return parents
            parents = _parents_from_edges(skeleton_meta.get(key))
            if parents is not None:
                return parents
        parents = _parents_from_legacy_metadata(skeleton_meta)
        if parents is not None:
            return parents
        
        raise KeyError('H3WB metadata must define skeleton parents.')

    def _resolve_joint(self, value: JointSpec) -> int:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise KeyError('Empty joint reference')
            if text in self._joint_name_to_index:
                return self._joint_name_to_index[text]
            try:
                return int(text)
            except ValueError as exc:
                raise KeyError(f'Unknown joint name: {value}') from exc
        return int(value)

    def _resolve_indices(self, values: Iterable[JointSpec], skeleton_meta: Mapping) -> List[int]:
        if values is None:
            return []
        joint_names = skeleton_meta.get('joint_names', [])
        name_to_index = {name: idx for idx, name in enumerate(joint_names)}
        resolved = []
        for value in values:
            if isinstance(value, str):
                text = value.strip()
                if text in name_to_index:
                    resolved.append(name_to_index[text])
                    continue
                try:
                    resolved.append(int(text))
                except ValueError as exc:
                    raise KeyError(f'Unknown joint name: {value}') from exc
            else:
                resolved.append(int(value))
        return resolved
    def _iter_joint_specs(self, spec) -> Iterable:
        if spec is None:
            return []
        if isinstance(spec, np.ndarray):
            if spec.ndim == 0:
                return self._iter_joint_specs(spec.item())
            return [item for element in spec.tolist() for item in self._iter_joint_specs(element)]
        if isinstance(spec, Mapping):
            return [item for value in spec.values() for item in self._iter_joint_specs(value)]
        if isinstance(spec, (list, tuple, set)):
            return [item for element in spec for item in self._iter_joint_specs(element)]
        if isinstance(spec, (bytes, bytearray)):
            try:
                return [spec.decode()]
            except Exception:
                return [spec]
        return [spec]

    def _resolve_joint_list(self, spec) -> List[int]:
        resolved: List[int] = []
        for candidate in self._iter_joint_specs(spec):
            if candidate is None:
                continue
            try:
                idx = self._resolve_joint(candidate)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if idx not in resolved:
                resolved.append(idx)
        return resolved

    def _find_hip_hint(self, metadata: Optional[Mapping]):
        if not isinstance(metadata, Mapping):
            return None
        preferred_keys = (
            'hip',
            'hips',
            'hip_center',
            'hip_indices',
            'hip_joints',
            'pelvis',
            'pelvis_center',
        )
        for key in preferred_keys:
            if key in metadata and metadata[key] is not None:
                return metadata[key]
        for key, value in metadata.items():
            if isinstance(key, str) and 'hip' in key.lower() and value is not None:
                return value
        return None

    def _infer_hip_from_structure(self) -> List[int]:
        skeleton = self.skeleton()
        if not skeleton:
            return []
        left = list(skeleton.joints_left())
        right = list(skeleton.joints_right())
        if not left or not right:
            return []
        parents_array = skeleton.parents()
        parents: List[int] = parents_array.tolist() if hasattr(parents_array, 'tolist') else list(parents_array)

        def depth(index: int) -> int:
            visited = set()
            current = index
            level = 0
            while current != -1 and current not in visited and current < len(parents):
                visited.add(current)
                parent = parents[current]
                if parent == current:
                    break
                current = parent
                level += 1
            return level

        best_pair: Optional[Tuple[int, int]] = None
        best_depth = float('inf')
        for l_idx, r_idx in zip(left, right):
            candidate_depth = min(depth(l_idx), depth(r_idx))
            if best_pair is None or candidate_depth < best_depth:
                best_pair = (l_idx, r_idx)
                best_depth = candidate_depth
        return list(best_pair) if best_pair is not None else []

    def _infer_hip_from_names(self) -> List[int]:
        skeleton = self.skeleton()
        if not skeleton or not self._joint_names:
            return []
        normalized_names = {
            idx: re.sub(r'[^a-z0-9]+', '', str(name).lower())
            for idx, name in enumerate(self._joint_names)
        }
        left = list(skeleton.joints_left())
        right = list(skeleton.joints_right())
        hip_tokens = ('hip', 'pelv')
        for l_idx, r_idx in zip(left, right):
            left_name = normalized_names.get(l_idx, '')
            right_name = normalized_names.get(r_idx, '')
            combined = left_name + right_name
            if any(token in combined for token in hip_tokens):
                return [l_idx, r_idx]
        hip_candidates = [
            idx
            for idx, name in normalized_names.items()
            if any(token in name for token in hip_tokens)
        ]
        if hip_candidates:
            left_set = set(left)
            right_set = set(right)
            left_choice = next((idx for idx in hip_candidates if idx in left_set), None)
            right_choice = next(
                (idx for idx in hip_candidates if idx in right_set and idx != left_choice),
                None,
            )
            if left_choice is not None and right_choice is not None:
                return [left_choice, right_choice]
            if len(hip_candidates) >= 2:
                return hip_candidates[:2]
        return []

    def _build_joint_groups(self, metadata: Mapping) -> Dict[str, List[int]]:
        groups = metadata.get('joint_groups', {})
        resolved = {}
        for name, indices in groups.items():
            resolved[name] = [self._resolve_joint(idx) for idx in indices]
        return resolved

    def _build_centering_map(self, metadata: Mapping) -> Dict[str, List[int]]:
        centering = metadata.get('centering', {})
        if isinstance(centering, np.ndarray):
            if centering.size == 1 and centering.dtype == object:
                centering = centering.item()
            else:
                centering = {}
        if not isinstance(centering, Mapping):
            centering = {}
        resolved = {}
        for mode, indices in centering.items():
            resolved[mode] = self._resolve_joint_list(indices)

        if 'root' not in resolved or not resolved['root']:
            resolved['root'] = [0]
        if 'hip' not in resolved or not resolved['hip']:
            hip_hint = self._find_hip_hint(metadata)
            if hip_hint is None:
                skeleton_meta = metadata.get('skeleton')
                if not isinstance(skeleton_meta, Mapping):
                    skeleton_meta = None
                hip_hint = self._find_hip_hint(skeleton_meta)
            hip_indices = self._resolve_joint_list(hip_hint) if hip_hint is not None else []
            if hip_indices:
                resolved['hip'] = hip_indices

        if 'hip' not in resolved or not resolved['hip']:
            inferred_from_names = self._infer_hip_from_names()
            if inferred_from_names:
                resolved['hip'] = inferred_from_names

        if 'hip' not in resolved or not resolved['hip']:
            inferred_from_structure = self._infer_hip_from_structure()
            if inferred_from_structure:
                resolved['hip'] = inferred_from_structure

        if 'hip' not in resolved or not resolved['hip']:
            num_joints = self.skeleton().num_joints() if self.skeleton() else None
            resolved['hip'] = [11, 12] if num_joints is None or num_joints > 12 else [0]
        return resolved
    def _normalize_subject_list(self, value: Optional[Union[List[str], Tuple[str, ...], set, str]]) -> Optional[List[str]]:
        if not value:
            return None
        if isinstance(value, str):
            items = [item for item in value.split(',') if item]
        else:
            items = list(value)
        return sorted(set(items))

    def _prepare_keypoints_metadata(self, metadata: Mapping, skeleton: Skeleton) -> Dict:
        raw = metadata.get('keypoints_metadata') or metadata.get('keypoints') or {}
        info = copy.deepcopy(raw)
        if 'layout_name' not in info:
            info['layout_name'] = metadata.get('layout_name', 'h3wb')
        if 'num_joints' not in info:
            if self._joint_names:
                info['num_joints'] = len(self._joint_names)
            else:
                info['num_joints'] = skeleton.num_joints()
        if 'keypoints_symmetry' not in info:
            info['keypoints_symmetry'] = (
                list(skeleton.joints_left()),
                list(skeleton.joints_right()),
            )
        return info

    def _build_cameras(self, metadata: Mapping) -> Dict[str, List[Dict]]:
        cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        overrides = metadata.get('cameras') or metadata.get('camera_parameters') or {}
        resolutions = metadata.get('camera_resolutions') or {}

        for subject, cams in cameras.items():
            subject_override = overrides.get(subject) if isinstance(overrides, Mapping) else None
            subject_res = resolutions.get(subject) if isinstance(resolutions, Mapping) else None
            for cam_idx, cam in enumerate(cams):
                intrinsic = copy.deepcopy(h36m_cameras_intrinsic_params[cam_idx % len(h36m_cameras_intrinsic_params)])
                cam.update(intrinsic)

                override = None
                if isinstance(subject_override, Mapping):
                    override = subject_override.get(cam_idx) or subject_override.get(cam.get('id'))
                elif isinstance(subject_override, (list, tuple)) and cam_idx < len(subject_override):
                    override = subject_override[cam_idx]
                if isinstance(override, Mapping):
                    for key, value in override.items():
                        cam[key] = value

                if isinstance(subject_res, Mapping):
                    res_entry = subject_res.get(cam_idx) or subject_res.get(cam.get('id'))
                elif isinstance(subject_res, (list, tuple)) and cam_idx < len(subject_res):
                    res_entry = subject_res[cam_idx]
                else:
                    res_entry = None
                if isinstance(res_entry, Mapping):
                    cam['res_w'] = res_entry.get('res_w', cam['res_w'])
                    cam['res_h'] = res_entry.get('res_h', cam['res_h'])

                for key, value in list(cam.items()):
                    if key in {'id', 'res_w', 'res_h'}:
                        continue
                    cam[key] = np.array(value, dtype='float32')

                cam['center'] = normalize_screen_coordinates(
                    cam['center'], w=cam['res_w'], h=cam['res_h']
                ).astype('float32')
                cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
                if 'translation' in cam:
                    cam['translation'] = cam['translation'] / 1000.0

                cam['intrinsic'] = np.concatenate(
                    (
                        cam['focal_length'],
                        cam['center'],
                        cam['radial_distortion'],
                        cam['tangential_distortion'],
                        [
                            1 / cam['focal_length'][0],
                            0,
                            -cam['center'][0] / cam['focal_length'][0],
                            0,
                            1 / cam['focal_length'][1],
                            -cam['center'][1] / cam['focal_length'][1],
                            0,
                            0,
                            1,
                        ],
                    )
                )
        return cameras
    # ------------------------------------------------------------------
    # Accessors for training pipeline
    # ------------------------------------------------------------------
    def metadata(self) -> Mapping:
        return self._metadata

    def subjects_train(self) -> Optional[List[str]]:
        return self._subjects_train

    def subjects_test(self) -> Optional[List[str]]:
        return self._subjects_test

    def default_centering(self) -> str:
        return self._default_centering

    def resolve_center(self, mode: str) -> Optional[List[int]]:
        return self._centering_map.get(mode)

    def has_partitions(self) -> bool:
        return bool(self._joint_groups)
    def keypoints(self) -> Dict[str, Dict[str, List[np.ndarray]]]:
        keypoints: Dict[str, Dict[str, List[np.ndarray]]] = {}
        for subject, actions in self._data.items():
            subject_dict: Dict[str, List[np.ndarray]] = {}
            for action, entry in actions.items():
                poses = None
                if 'positions_2d' in entry:
                    poses = entry['positions_2d']
                elif 'pose_2d' in entry:
                    poses = entry['pose_2d']
                if poses is not None:
                    subject_dict[action] = [np.asarray(pose, dtype='float32').copy() for pose in poses]
            if subject_dict:
                keypoints[subject] = subject_dict
        return keypoints

    def keypoints_metadata(self) -> Mapping:
        return self._keypoints_metadata

    def keypoints_are_normalized(self) -> bool:
        return self._keypoints_normalized
    
    def accumulate_part_errors(
        self, predicted: np.ndarray, target: np.ndarray
    ) -> Tuple[Dict[str, float], int]:
        if not self._joint_groups:
            return {}, 0

        num_frames = int(np.prod(predicted.shape[:2]))
        errors: Dict[str, float] = {}
        for name, indices in self._joint_groups.items():
            if not indices:
                continue
            diff = predicted[..., indices, :] - target[..., indices, :]
            per_joint = np.linalg.norm(diff, axis=-1)
            per_frame = per_joint.mean(axis=-1)
            errors[name] = per_frame.sum()
        return errors, num_frames