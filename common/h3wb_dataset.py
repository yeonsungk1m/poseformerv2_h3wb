# Copyright (c) 2024, PoseFormerV2 contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from common.camera import normalize_screen_coordinates
from common.h36m_dataset import (
    h36m_cameras_extrinsic_params,
    h36m_cameras_intrinsic_params,
)
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton


JointSpec = Union[int, str]


def _to_python_dict(data: Mapping, key: str, fallback: Optional[str] = None):
    """Utility to safely extract nested dictionaries from a numpy npz file."""

    if key in data:
        raw_value = data[key]
    elif fallback is not None and fallback in data:
        raw_value = data[fallback]
    else:
        return None

    # Values stored in numpy files are wrapped in object arrays. Convert them
    # to proper Python dictionaries for ease of use.
    if isinstance(raw_value, np.ndarray) and raw_value.dtype == object:
        return raw_value.item()
    return raw_value


class Human3WBDataset(MocapDataset):
    """Loader for the Human3.6M Whole-Body (H3WB) dataset.

    The dataset serialisation closely mirrors the Human3.6M protocol. 3D
    poses are stored per camera in millimetres while 2D detections are in
    image coordinates. During initialisation we:

    * Build the 133-joint skeleton from metadata (left/right joint sets).
    * Deep-copy the Human3.6M camera intrinsics/extrinsics and normalise the
      camera centre coordinates while converting translations from mm to m.
    * Store, for each sequence, the world-frame 3D poses, camera-frame 3D
      poses and corresponding 2D detections.

    Additional metadata is preserved so that the training script can derive
    dataset-specific behaviour (e.g. centering strategy and per-part metrics).
    """

    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)

        metadata = data['metadata'].item()
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

        self._subjects_train = metadata.get('subjects_train')
        self._subjects_test = metadata.get('subjects_test')

        # Clone the Human3.6M cameras for compatibility with the existing
        # training pipeline. Orientation is reused but translation is scaled
        # to metres.
        self._cameras = copy.deepcopy(h36m_cameras_extrinsic_params)
        for subject, cameras in self._cameras.items():
            for cam_idx, cam in enumerate(cameras):
                cam.update(copy.deepcopy(h36m_cameras_intrinsic_params[cam_idx]))
                for key, value in cam.items():
                    if key not in ['id', 'res_w', 'res_h']:
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

        positions_world = _to_python_dict(data, 'positions_world', fallback='positions')
        positions_camera = _to_python_dict(data, 'positions_3d')
        positions_2d = _to_python_dict(data, 'positions_2d')

        self._data = {}
        subjects = metadata.get('subjects')
        if subjects is None:
            sources = [positions_world, positions_camera, positions_2d]
            subjects = set()
            for source in sources:
                if source is not None:
                    subjects.update(source.keys())
        for subject in sorted(subjects):
            self._data[subject] = {}
            subject_world = positions_world[subject] if positions_world and subject in positions_world else {}
            subject_camera = (
                positions_camera[subject] if positions_camera and subject in positions_camera else {}
            )
            subject_2d = positions_2d[subject] if positions_2d and subject in positions_2d else {}

            sequence_names = set(subject_world.keys()) | set(subject_camera.keys()) | set(subject_2d.keys())
            for seq in sorted(sequence_names):
                entry = {'cameras': self._cameras.get(subject, [])}
                if seq in subject_world:
                    entry['positions'] = subject_world[seq].astype('float32')
                if seq in subject_camera:
                    entry['positions_3d'] = [np.array(p, dtype='float32') for p in subject_camera[seq]]
                if seq in subject_2d:
                    entry['positions_2d'] = [np.array(p, dtype='float32') for p in subject_2d[seq]]
                self._data[subject][seq] = entry

    # ------------------------------------------------------------------
    # Helper builders
    # ------------------------------------------------------------------
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