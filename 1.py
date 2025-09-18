# 1.py (대체)
import numpy as np
from collections.abc import Mapping

def peek(name, obj, depth=0, max_children=6):
    indent = '  ' * depth
    if isinstance(obj, np.ndarray):
        print(f"{indent}{name}: ndarray shape={obj.shape}, dtype={obj.dtype}")
        # object 배열이면 내부 한두 개를 더 들여다보기
        if obj.dtype == object:
            flat = obj.ravel()
            for i, it in enumerate(flat[:min(len(flat), max_children)]):
                peek(f"{name}[obj_{i}]", it, depth+1)
    elif isinstance(obj, Mapping):
        print(f"{indent}{name}: dict with {len(obj)} keys")
        for i, k in enumerate(list(obj.keys())[:max_children]):
            peek(f"{name}['{k}']", obj[k], depth+1)
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}{name}: {type(obj).__name__} len={len(obj)}")
        for i, it in enumerate(obj[:min(len(obj), max_children)]):
            peek(f"{name}[{i}]", it, depth+1)
    else:
        print(f"{indent}{name}: {type(obj).__name__} -> {obj}")

path = '/home/s3/kimyeonsung/poseformerv2_h3wb/data/task1_test_3d.npz'  # 수정
data = np.load(path, allow_pickle=True)

# 최상단 키들
print("top-level keys:", list(data.keys()))

# 흔한 키 이름들 시도 (metadata/train_data/test_data)
meta = data['metadata'].item() if 'metadata' in data else {}
print("metadata keys:", list(meta.keys()))

train = data['train_data'].item() if 'train_data' in data else (
        data['train'].item() if 'train' in data else None)
print("has train_data:", isinstance(train, dict))

if train:
    # 한 샘플만 깊게 훑기
    subj = next(iter(train))
    act  = next(iter(train[subj]))
    print("sample -> subject:", subj, "action:", act)
    peek("train[subj][act]", train[subj][act])

    # 액션 레벨에서 pose_2d/positions_2d가 바로 있는지 체크
    act_block = train[subj][act]
    if isinstance(act_block, dict):
        if 'pose_2d' in act_block:
            peek("act_block['pose_2d']", act_block['pose_2d'])
        if 'positions_2d' in act_block:
            peek("act_block['positions_2d']", act_block['positions_2d'])
