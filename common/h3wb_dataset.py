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

from common.camera import normalize_screen_coordinates
from common.h36m_dataset import (
    h36m_cameras_extrinsic_params,
    h36m_cameras_intrinsic_params,
)
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton


JointSpec = Union[int, str]


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

    
class Human3WBDataset(MocapDataset):
    """Loader for the Human3.6M Whole-Body (H3WB) dataset."""
    def __init__(self, path: Union[str, Sequence[str]]):
        paths = _resolve_dataset_paths(path)
        records = [self._load_archive(p) for p in paths]

        metadata = self._merge_metadata(records)
        skeleton_meta = metadata.get('skeleton', metadata)

        parents = self._build_parents(skeleton_meta)
        joints_left = self._resolve_indices(skeleton_meta.get('joints_left', []), skeleton_meta)
        joints_right = self._resolve_indices(skeleton_meta.get('joints_right', []), skeleton_meta)

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
        parents = skeleton_meta.get('parents')
        if parents is not None:
            return list(parents)

        # Fallback: stitch part-wise parent lists together. Each part is
        # expected to provide local parents and an optional attachment point in
        # the already processed structure.
        parents = []
        offset = 0
        parts = skeleton_meta.get('parts', [])
        for part in parts:
            local_parents = part.get('parents', [])
            attach_to = part.get('attach_to', -1)
            local_parents = list(local_parents)
            for idx, parent in enumerate(local_parents):
                if parent == -1 and attach_to != -1:
                    parents.append(attach_to)
                else:
                    parents.append(parent + offset if parent != -1 else -1)
            offset += len(local_parents)
        if parents:
            return parents
        raise KeyError('H3WB metadata must define skeleton parents.')

    def _resolve_joint(self, value: JointSpec) -> int:
        if isinstance(value, str):
            if value not in self._joint_name_to_index:
                raise KeyError(f'Unknown joint name: {value}')
            return self._joint_name_to_index[value]
        return int(value)

    def _resolve_indices(self, values: Iterable[JointSpec], skeleton_meta: Mapping) -> List[int]:
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