import os
import json
from collections import defaultdict

import numpy as np
import open3d as o3d
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    hamming_loss,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from scipy.optimize import linear_sum_assignment
from utils.dataset import TMPPDataset
from collections import Counter
import yaml



# ==========================================
# 1. Constants & Configuration
# ==========================================

NUM2TASK = {
    1: "leave", 2: "relocate", 3: "reorient", 4: "fold",
    5: "washing-up", 6: "mop", 7: "vacuum", 8: "wipe",
    9: "close", 10: "turn-off", 11: "dispose", 12: "empty",
}

TASK2NUM = {
    "leave": 1,
    "relocate": 2,
    "reorient": 3,
    "fold": 4,
    "washing-up": 5,
    "mop": 6,
    "vacuum": 7,
    "wipe": 8,
    "close": 9,
    "turn-off": 10,
    "turn": 10,  # alias
    "dispose": 11,
    "empty": 12,
}


IOU_THRES = 0.25
ZERO_DIVISION = 0


# ==========================================
# 2. Geometry & Math Utils
# ==========================================

def compute_iou(center1, size1, center2, size2) -> float:
    center1 = np.asarray(center1, dtype=np.float32)
    size1 = np.asarray(size1, dtype=np.float32)
    center2 = np.asarray(center2, dtype=np.float32)
    size2 = np.asarray(size2, dtype=np.float32)

    min1, max1 = center1 - size1 / 2.0, center1 + size1 / 2.0
    min2, max2 = center2 - size2 / 2.0, center2 + size2 / 2.0

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(inter_max - inter_min, 0.0)

    inter_vol = float(np.prod(inter_size))
    vol1 = float(np.prod(size1))
    vol2 = float(np.prod(size2))

    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0
    return float(inter_vol / union_vol)


def read_tasmap_pp_npy(npy_path):
    data = np.load(npy_path, allow_pickle=True)
    positions = np.array([item["position"] for item in data])
    colors_rgb = np.array([item["color"] for item in data])
    keys = np.array([item["key"] for item in data])
    tasks = np.array([item["task"] for item in data])
    return positions, colors_rgb, keys, tasks


def get_bbox_from_points(positions, keys, min_size=0.01):
    key_groups = defaultdict(list)
    for idx, k in enumerate(keys):
        key_groups[int(k)].append(idx)

    bbox_data = {}
    for inst_key, indices in key_groups.items():
        points = positions[indices]
        bbox_o3d = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(points)
        )

        center = np.asarray(bbox_o3d.get_center(), dtype=np.float32)
        extent = np.asarray(bbox_o3d.get_extent(), dtype=np.float32)
        safe_extent = np.maximum(extent, min_size)

        bbox_data[int(inst_key)] = {
            "bbox_center": center.tolist(),
            "bbox_size": safe_extent.tolist(),
        }

    return bbox_data, key_groups


def match_object_to_instance_hungarian(object_bbox_data, instance_bbox_data, iou_thres):
    obj_ids = list(object_bbox_data.keys())
    inst_ids = list(instance_bbox_data.keys())

    if not obj_ids or not inst_ids:
        return {}, {}

    iou_mat = np.zeros((len(obj_ids), len(inst_ids)), dtype=np.float32)

    for i, obj_id in enumerate(obj_ids):
        o = object_bbox_data[obj_id]
        o_size = o.get("bbox_size", o.get("bbox_extend"))
        for j, inst_id in enumerate(inst_ids):
            s = instance_bbox_data[inst_id]
            iou_mat[i, j] = compute_iou(o["bbox_center"], o_size, s["bbox_center"], s["bbox_size"])

    row_ind, col_ind = linear_sum_assignment(-iou_mat)

    obj_to_inst = {}
    inst_to_obj = {}

    for r, c in zip(row_ind, col_ind):
        iou = float(iou_mat[r, c])
        if iou < iou_thres:
            continue
        obj_id = int(obj_ids[r])
        inst_id = int(inst_ids[c])
        obj_to_inst[obj_id] = {"best_idx": inst_id, "iou_score": iou}
        inst_to_obj[inst_id] = {"best_idx": obj_id, "iou_score": iou}

    return obj_to_inst, inst_to_obj


def parse_task_string(task_raw):

    if isinstance(task_raw, list):
        candidates = task_raw
    elif isinstance(task_raw, str):
        candidates = task_raw.split(",")
    elif isinstance(task_raw, (int, np.integer)):
        candidates = [str(task_raw)]
    else:
        return []

    valid_tasks = []
    for t in candidates:
        if isinstance(t, str):
            t = t.strip().lower()
            if not t:
                continue
            if t.isdigit():
                val = int(t)
                if val in NUM2TASK:
                    valid_tasks.append(val)
            elif t in TASK2NUM:
                valid_tasks.append(TASK2NUM[t])
            else:
                first_word = t.split(" ")[0]
                if first_word in TASK2NUM:
                    valid_tasks.append(TASK2NUM[first_word])
        elif isinstance(t, (int, np.integer)):
            val = int(t)
            if val in NUM2TASK:
                valid_tasks.append(val)

    # if valid_tasks == []:
    #     return None
    return sorted(set(valid_tasks))


# ==========================================
# 3. Method Loaders
# ==========================================

def load_tasmap_pp_scene(npy_path, is_gt=False):
    if not os.path.exists(npy_path):
        return None, None

    positions, _, keys, tasks = read_tasmap_pp_npy(npy_path)
    inst_bbox, key_groups = get_bbox_from_points(positions, keys)

    inst_tasks = {}
    for inst_id, indices in key_groups.items():
        if type(tasks[indices[0]]) == dict:
            inst_tasks[int(inst_id)] = tasks[indices[0]].get("task_GT")
        else:
            raw_task = tasks[indices[0]]
            inst_tasks[int(inst_id)] = parse_task_string(raw_task)

    return inst_bbox, inst_tasks







# ==========================================
# 5. Preprocessing: GT + Methods
# ==========================================

def load_gt_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gt_dict_for_scene_names(scene_names, gt_path_template):
    gt_dict = {}
    for s in scene_names:
        p = gt_path_template.format(scene_name=s, scene_id=s)
        gt_dict[s] = load_gt_json(p)
    return gt_dict


def extract_gt_task_list(obj_payload):
    if isinstance(obj_payload, dict):
        if "task" in obj_payload:
            return parse_task_string(obj_payload["task"])
        elif "action" in obj_payload:
            return parse_task_string(obj_payload["action"])
    return []




def get_gt_data(cfg, dataset):
    gt_data = {}
    scene_list = dataset.get_scene_list()
    for scene_id in tqdm(scene_list, desc="Preprocessing GT Scenes"):
        benchmark_npy_path = dataset.get_benchmark_file_path(cfg, scene_id)
        gt_bbox, gt_tasks = load_tasmap_pp_scene(benchmark_npy_path)
        if gt_bbox is None or gt_tasks is None:
            continue

        seq_name = "_".join(scene_id)
        gt_data[seq_name] = {
            "bbox": gt_bbox,
            "tasks": gt_tasks,
        }

    return gt_data


def get_method_data(cfg, dataset, methods):
    method_data = {m: {} for m in methods}
    scene_list = dataset.get_scene_list()
    for scene_id in tqdm(scene_list, desc="Preprocessing Methods"):
        for method in methods:
            result_npy_path = dataset.get_result_file_path(cfg, scene_id)
            inst_bbox, inst_tasks = load_tasmap_pp_scene(result_npy_path)

            if inst_bbox is None or inst_tasks is None:
                continue

            seq_name = "_".join(scene_id)
            method_data[method][seq_name] = {
                "bbox": inst_bbox, 
                "tasks": inst_tasks
            }

    return method_data


# ==========================================
# 6. Build full_results + matches (NOW uses gt_data)
# ==========================================

def build_scene_results(seq_name, gt_scene_data, method_data, methods):

    gt_bbox = gt_scene_data["bbox"]
    gt_tasks = gt_scene_data["tasks"]

    obj_ids = list(gt_bbox.keys())

    scene_results = defaultdict(dict)
    scene_matches = {}

    # GT tasks
    for obj_id in obj_ids:
        scene_results[str(obj_id)]["task_GT"] = gt_tasks.get(obj_id, [])

    # preds
    for method in methods:
        scene_matches[method] = {}

        if seq_name not in method_data.get(method, {}):
            for obj_id in obj_ids:
                scene_results[str(obj_id)][method] = []
            continue

        inst_bbox = method_data[method][seq_name]["bbox"]
        inst_tasks = method_data[method][seq_name]["tasks"]

        obj_to_inst, _ = match_object_to_instance_hungarian(gt_bbox, inst_bbox, IOU_THRES)
        scene_matches[method] = obj_to_inst

        for obj_id in obj_ids:
            match_info = obj_to_inst.get(int(obj_id))
            if match_info:
                inst_id = match_info["best_idx"]
                scene_results[str(obj_id)][method] = inst_tasks.get(inst_id, [])
            # else:
            #     scene_results[str(obj_id)][method] = []

    return scene_results, scene_matches


# ==========================================
# 7. Metrics
# ==========================================

def to_num_labels(lst):
    out = []
    for x in lst:
        if isinstance(x, (int, np.integer)):
            out.append(int(x))
        else:
            k = str(x).strip().lower()
            if k in TASK2NUM:
                out.append(TASK2NUM[k])
            elif k.isdigit():
                v = int(k)
                if v in NUM2TASK:
                    out.append(v)
    return sorted(set(out))


def convert_numeric_keys_to_int(obj):
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            nk = k
            if isinstance(k, str) and k.isdigit():
                nk = int(k)
            new[nk] = convert_numeric_keys_to_int(v)
        return new
    elif isinstance(obj, list):
        return [convert_numeric_keys_to_int(x) for x in obj]
    return obj




def iter_samples(full_results, methods):
    for scene_id, d in full_results.items():
        for key, v in d.items():
            if "task_GT" not in v:
                continue
            y_true = to_num_labels(v["task_GT"])
            for method in methods:
                if method in v:
                    if v[method] is None:
                        continue
                    y_pred = to_num_labels(v[method])
                # else:
                #     # 예측 없는 경우 => 전부 0으로 취급
                #     y_pred = []
                    yield y_true, y_pred, method, scene_id, key


def compute_metrics(full_results, methods, all_classes=None, verbose=False):
    if all_classes is not None:
        mlb = MultiLabelBinarizer(classes=all_classes)
    else:
        mlb = MultiLabelBinarizer()

    per_model_true = defaultdict(list)
    per_model_pred = defaultdict(list)
    per_sample_rows = []

    for y_true, y_pred, method, scene_id, key in iter_samples(full_results, methods):
        per_model_true[method].append(y_true)
        per_model_pred[method].append(y_pred)
        per_sample_rows.append({
            "scene_id": scene_id,
            "key": key,
            "model": method,
            "gt": y_true,
            "pred": y_pred,
        })

    summary = {}
    for method in methods:
        if method not in per_model_true:
            continue

        Y_true_bin = mlb.fit_transform(per_model_true[method])
        Y_pred_bin = mlb.transform(per_model_pred[method])

        hl = hamming_loss(Y_true_bin, Y_pred_bin)
        hs = 1.0 - hl
        subset_acc = accuracy_score(Y_true_bin, Y_pred_bin)

        f1_macro = f1_score(Y_true_bin, Y_pred_bin, average="macro", zero_division=ZERO_DIVISION)
        precision_macro = precision_score(
            Y_true_bin, Y_pred_bin, average="macro", zero_division=ZERO_DIVISION
        )
        recall_macro = recall_score(
            Y_true_bin, Y_pred_bin, average="macro", zero_division=ZERO_DIVISION
        )

        f1_micro = f1_score(Y_true_bin, Y_pred_bin, average="micro", zero_division=ZERO_DIVISION)
        precision_micro = precision_score(
            Y_true_bin, Y_pred_bin, average="micro", zero_division=ZERO_DIVISION
        )
        recall_micro = recall_score(
            Y_true_bin, Y_pred_bin, average="micro", zero_division=ZERO_DIVISION
        )

        cls_prec, cls_rec, cls_f1, cls_support = precision_recall_fscore_support(
            Y_true_bin, Y_pred_bin, average=None, zero_division=ZERO_DIVISION
        )

        per_class_metrics = {}
        for idx, cls_id in enumerate(mlb.classes_):
            cls_id = int(cls_id)
            cls_name = NUM2TASK.get(cls_id, str(cls_id))
            per_class_metrics[cls_id] = {
                "name": cls_name,
                "support": int(cls_support[idx]),
                "precision": float(f"{cls_prec[idx]:.3f}"),
                "recall": float(f"{cls_rec[idx]:.3f}"),
                "f1": float(f"{cls_f1[idx]:.3f}"),
            }

        summary[method] = {
            "num_samples": int(Y_true_bin.shape[0]),
            "(CP) precision_macro": f"{precision_macro:.3f}",
            "(CR) recall_macro": f"{recall_macro:.3f}",
            "(CF1) f1_macro": f"{f1_macro:.3f}",
            "(OP) precision_micro": f"{precision_micro:.3f}",
            "(OR) recall_micro": f"{recall_micro:.3f}",
            "(OF1) f1_micro": f"{f1_micro:.3f}",
            "(1-HL) hamming_score": f"{hs:.3f}",
            "(SA) subset_acc": f"{subset_acc:.3f}",
            "per_class": per_class_metrics,
        }

    return {"summary": summary, "per_sample": per_sample_rows}



def print_gt_task_counts_and_ratios(gt_data, classes=range(1, 13)):
    cnt = Counter()
    total_objs = 0
    total_labels = 0  

    for scene_id, sdata in gt_data.items():
        obj_tasks = sdata.get("tasks", {})
        for obj_id, tasks in obj_tasks.items():
            total_objs += 1
            tasks = tasks or []
            clean = []
            for t in tasks:
                try:
                    t = int(t)
                except Exception:
                    continue
                if t in classes:
                    clean.append(t)

            total_labels += len(clean)
            for t in clean:
                cnt[t] += 1

    print(f"[GT] scenes={len(gt_data)}, objects={total_objs}, total_task_labels={total_labels}")
    print("task_id (name) : count | % of labels | % of objects")
    for t in classes:
        count = cnt[t]
        pct_labels = (count / total_labels * 100.0) if total_labels > 0 else 0.0
        pct_objs = (count / total_objs * 100.0) if total_objs > 0 else 0.0
        print(f"{t:2d} ({NUM2TASK.get(t,'?'):>10s}) : {count:5d} | {pct_labels:6.2f}% | {pct_objs:6.2f}%")

# ==========================================
# 8. Evaluate
# ==========================================

def evaluate(
    cfg,
    dataset,
    output_dir,
    methods,
    save_intermediate=True,
):

    os.makedirs(output_dir, exist_ok=True)
    save_res_path = os.path.join(output_dir, "full_results.json")
    save_match_path = os.path.join(output_dir, "obj_to_inst.json")
    save_eval_path = os.path.join(output_dir, "eval_summary.json")


    # 2) Preprocess GT + Methods
    print("Preprocessing GT (get_gt_data)...")
    save_gt_data_path = os.path.join(output_dir, "gt_data.json")
    if save_intermediate and os.path.exists(save_gt_data_path):
        with open(save_gt_data_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_data = convert_numeric_keys_to_int(gt_data)
    else:
        gt_data = get_gt_data(cfg, dataset)
        with open(save_gt_data_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=4)

    print_gt_task_counts_and_ratios(gt_data, classes=range(1, 13))

    print("Preprocessing Methods (get_method_data)...")
    save_method_data_path = os.path.join(output_dir, "method_data.json")
    if save_intermediate and os.path.exists(save_method_data_path):
        with open(save_method_data_path, "r", encoding="utf-8") as f:
            method_data = json.load(f)
        method_data = convert_numeric_keys_to_int(method_data)
    else:
        method_data = get_method_data(cfg, dataset, methods)
        with open(save_method_data_path, "w", encoding="utf-8") as f:
            json.dump(method_data, f, ensure_ascii=False, indent=4)
    

    if save_intermediate and os.path.exists(save_res_path):
        with open(save_res_path, "r", encoding="utf-8") as f:
            full_results = json.load(f)
    else:
        full_results = {}

    if save_intermediate and os.path.exists(save_match_path):
        with open(save_match_path, "r", encoding="utf-8") as f:
            full_obj_to_inst = json.load(f)
    else:
        full_obj_to_inst = {}


    print("Generating matched results...")
    scene_list = dataset.get_scene_list()
    for scene_id in tqdm(scene_list, desc="Evaluating Scenes"):
        seq_name = "_".join(scene_id)
        if seq_name not in gt_data:
            continue  # GT bbox missing or GT json missing

        scene_results, scene_matches = build_scene_results(
            seq_name=seq_name,
            gt_scene_data=gt_data[seq_name],
            method_data=method_data,
            methods=methods
        )

        full_results.setdefault(seq_name, {})
        for k, v in scene_results.items():
            full_results[seq_name][k] = v

        full_obj_to_inst.setdefault(seq_name, {})
        full_obj_to_inst[seq_name].update(scene_matches)

        if save_intermediate:
            with open(save_res_path, "w", encoding="utf-8") as f:
                json.dump(full_results, f, ensure_ascii=False, indent=4)
            with open(save_match_path, "w", encoding="utf-8") as f:
                json.dump(full_obj_to_inst, f, ensure_ascii=False, indent=4)

    # 4) Metrics
    print("Computing metrics...")
    summary = compute_metrics(full_results, methods, all_classes=ALL_CLASSES)

    with open(save_eval_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(json.dumps(summary, ensure_ascii=False, indent=4))
    print(f"Saved outputs to: {output_dir}")
    return summary





if __name__ == "__main__":
    DATASET = "tasmap_pp" 
    ALL_CLASSES = list(range(1, 13))
    # ALL_CLASSES = None

    
    if DATASET == "tasmap_pp":
        # Load configuration
        config_file_path = "/workspace/tasmap_pp/configs/tasmap_pp.yaml"
        cfg = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)
        dataset = TMPPDataset(root_env_dir=cfg["data_path"]["root_env_dir"])
        benchmark_root_dir = cfg["evaluation"]["benchmark_root_dir"]


    output_dir = os.path.join("/workspace/tasmap_pp/eval_outputs", DATASET)


    evaluate(
        cfg=cfg,
        dataset=dataset,
        output_dir=output_dir,
        methods=["tasmap_pp"],
        save_intermediate=True,
    )
