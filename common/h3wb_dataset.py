# Copyright (c) 2024, PoseFormerV2 contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import glob
import os
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


def _coerce_camera_sequence(subject: str, cameras: Dict[str, List[Dict]], value, dims: int) -> List[np.ndarray]:
    if value is None:
        return []

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
                ordered.append(_reshape_pose(found, dims))
        for key, item in value.items():
            if key in used_keys:
                continue
            ordered.append(_reshape_pose(item, dims))
        return ordered

    if isinstance(value, np.ndarray):
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

    return [_reshape_pose(item, dims) for item in sequence]
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
    def __init__(self, path: Union[str, Sequence[str]]):
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
        for record in records:
            _merge_nested_dict(positions_world, record['positions_world'])
            _merge_nested_dict(positions_camera, record['positions_3d'])
            _merge_nested_dict(positions_2d, record['positions_2d'])        

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

            sequence_names = set(subject_world.keys()) | set(subject_camera.keys()) | set(subject_2d.keys())
            for seq in sorted(sequence_names):
                entry: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {
                    'cameras': self._cameras.get(subject, [])
                }
                if seq in subject_world:
                    entry['positions'] = np.asarray(subject_world[seq], dtype='float32')
                if seq in subject_camera:
                    entry['positions_3d'] = _coerce_camera_sequence(subject, self._cameras, subject_camera[seq], dims=3)
                if seq in subject_2d:
                    entry['positions_2d'] = _coerce_camera_sequence(subject, self._cameras, subject_2d[seq], dims=2)
                self._data[subject][seq] = entry

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------
    def _load_archive(self, path: str) -> Dict:
        with np.load(path, allow_pickle=True) as data:
            metadata = data['metadata'].item() if 'metadata' in data else {}
            return {
                'path': path,
                'metadata': metadata,
                'positions_world': _to_python_dict(data, 'positions_world', 'positions', 'pose_world'),
                'positions_3d': _to_python_dict(
                    data,
                    'positions_3d',
                    'pose_3d',
                    'poses_3d',
                    'pose3d',
                ),
                'positions_2d': _to_python_dict(
                    data,
                    'positions_2d',
                    'pose_2d',
                    'poses_2d',
                    'pose2d',
                ),
            }

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
            if value not in self._joint_name_to_index:
                raise KeyError(f'Unknown joint name: {value}')
            return self._joint_name_to_index[value]
        return int(value)

    def _resolve_indices(self, values: Iterable[JointSpec], skeleton_meta: Mapping) -> List[int]:
        if values is None:
            return []
        joint_names = skeleton_meta.get('joint_names', [])
        name_to_index = {name: idx for idx, name in enumerate(joint_names)}
        resolved = []
        for value in values:
            if isinstance(value, str):
                if value not in name_to_index:
                    raise KeyError(f'Unknown joint name: {value}')
                resolved.append(name_to_index[value])
            else:
                resolved.append(int(value))
        return resolved

    def _build_joint_groups(self, metadata: Mapping) -> Dict[str, List[int]]:
        groups = metadata.get('joint_groups', {})
        resolved = {}
        for name, indices in groups.items():
            resolved[name] = [self._resolve_joint(idx) for idx in indices]
        return resolved

    def _build_centering_map(self, metadata: Mapping) -> Dict[str, List[int]]:
        centering = metadata.get('centering', {})
        resolved = {}
        for mode, indices in centering.items():
            if isinstance(indices, (list, tuple)):
                resolved[mode] = [self._resolve_joint(idx) for idx in indices]
            else:
                resolved[mode] = [self._resolve_joint(indices)]
        if 'root' not in resolved:
            resolved['root'] = [0]
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
                if 'positions_2d' in entry:
                    subject_dict[action] = [pose.copy() for pose in entry['positions_2d']]
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