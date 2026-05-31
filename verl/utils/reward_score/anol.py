import json
import re
import math
from typing import List, Dict, Union, Tuple
# pip install affiliation-metrics
from affiliation.metrics import pr_from_events
# pip install scikit-learn
from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np

from affiliation.generics import convert_vector_to_events

MAX_LEN = 230400  # upper bound on time-series length (matches TSB-UAD max)

ALLOWED_PRED_CLASSES = {
    "contextual point",
    "global point",
    "seasonal",
    "shapelet",
    "trend",
    'normal'
}

def _normalize_pred_label(label: str) -> str:

    if not isinstance(label, str):
        return ""
    s = " ".join(label.strip().lower().split())
    if s == "contextual":
        s = "contextual point"
    elif s == "global":
        s = "global point"
    return s if s in ALLOWED_PRED_CLASSES else ""

def _normalize_gt_label(label: str) -> str:
   
    if not isinstance(label, str):
        return ""
    s = " ".join(label.strip().lower().split())
    if s == "contextual":
        return "contextual point"
    if s == "global":
        return "global point"
    return s if s in {"seasonal", "shapelet", "trend", "contextual point", "global point", 'normal'} else ""


def extract_prediction_type(text: str) -> Tuple[str, bool]:


    m = re.search(r"<class>(.*?)</class>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return "", False

    raw = m.group(1).strip()
    if not raw or raw.lower() in {"none", "[]"}:
        return "", True  

    candidate = raw
   
    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data and isinstance(data[0], str):
                candidate = data[0]
            elif isinstance(data, dict):
                for key in ["prediction", "class", "label", "type"]:
                    if key in data and isinstance(data[key], str):
                        candidate = data[key]
                        break
        except Exception:
            candidate = raw

    pred = _normalize_pred_label(str(candidate))
    return pred, True



def extract_prediction_intervals(text: str):
  
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE))
    matches_think = list(re.finditer(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE))
    fmt_score = 1.0

    if not matches:
        return "wrong", 0.0
    if not matches_think:
        fmt_score = 0.0

    content = matches[-1].group(1).strip()
    if not content or content.lower() in {"none"}:
        return "wrong", 0.0
    if content.lower() in {"[]", "array([])"}:
        return [], fmt_score

 
    try:
        parsed_data = json.loads(content)
    except json.JSONDecodeError:
        try:
            allowed_globals = {"__builtins__": None}
            allowed_locals = {"np": np, "array": np.array, "list": list, "tuple": tuple, "int": int, "float": float}
            parsed_data = eval(content, allowed_globals, allowed_locals)
        except Exception as e:
            return "wrong", 0.0

    intervals = []
    for item in parsed_data if isinstance(parsed_data, list) else [parsed_data]:
        try:
            if isinstance(item, dict) and "start" in item and "end" in item:
                start, end = item["start"], item["end"]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                start, end = item
            elif hasattr(item, "tolist") and callable(item.tolist):
                start, end = item.tolist()
            else:
                continue

            if not np.isfinite(start) or not np.isfinite(end):
                continue
            start, end = int(start), int(end)
            if start < 0 or end < 0 or start >= end:
                continue
            if end > MAX_LEN:
                end = MAX_LEN

            intervals.append((start, end))
        except Exception:
            continue

    if not intervals:
        return "wrong", fmt_score

    return intervals, fmt_score


def extract_gt_intervals(gt: Union[str, List]) -> List[Tuple[int, int]]:
   
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception:
            try:
                gt = eval(gt, {"__builtins__": None}, {"array": np.array, "np": np})
            except Exception as e:
                print(f"[ERROR Ground] Failed to parse ground truth: {e}")
                assert False, f"Invalid ground truth format: {gt}"

    intervals = []
    try:
        for item in gt:
            if isinstance(item, dict) and "start" in item and "end" in item:
                intervals.append((int(item["start"]), int(item["end"])))
            elif isinstance(item, list) and len(item) == 2:
                intervals.append((int(item[0]), int(item[1])))
            elif hasattr(item, 'tolist') and len(item.tolist()) == 2:
                item = item.tolist()
                intervals.append((int(item[0]), int(item[1])))
    except Exception as e:
        print(f"[WARN] Failed to standardize GT interval: {e}")

    intervals = [pair for pair in intervals if pair[0] <= pair[1]]
    return intervals


def compute_affinity_reward_single_sample(
    predict_str: str,
    ground_truth: Union[str, List[Dict]],
    *,
    total_len: int,           
    ground_type: str = "None"  
) -> Tuple[Dict[str, float], float, List[Tuple[int, int]], List[Tuple[int, int]]]:

    gt_intervals = extract_gt_intervals(ground_truth)
    gt_type_norm = _normalize_gt_label(ground_type)
    pred_intervals, fmt_score = extract_prediction_intervals(predict_str)

 
    pred_type, fmt_has_class = extract_prediction_type(predict_str)
    if not fmt_has_class:
        fmt_score = 0.0


    class_acc = 1.0 if (pred_type != "" and pred_type == gt_type_norm) else 0.0


    if pred_intervals == "wrong":
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "affi precision": 0.0, "affi recall": 0.0, "affi f1": 0.0,
            "class_acc": round(float(class_acc), 4),
        }, fmt_score, [], gt_intervals

    if not pred_intervals and not gt_intervals:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "affi precision": 1.0, "affi recall": 1.0, "affi f1": 1.0,
            "class_acc": round(float(class_acc), 4),
        }, fmt_score, [], []

    try:
        trange = (0, int(total_len))
        length = trange[1]

        def intervals_to_vector(intervals, vec_length):
            vec = np.zeros(vec_length, dtype=int)
            for start, end in intervals:
                start_c = max(0, min(vec_length, start))
                end_c = max(0, min(vec_length, end))
                if start_c < end_c:
                    vec[start_c:end_c] = 1
            return vec

        y_pred = intervals_to_vector(pred_intervals, length)
        y_true = intervals_to_vector(gt_intervals, length)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        events_pred = convert_vector_to_events(y_pred)
        events_gt = convert_vector_to_events(y_true)


        try:
            aff = pr_from_events(events_pred, events_gt, trange)
            aff_precision = float(aff.get("precision", 0.0))
            aff_recall = float(aff.get("recall", 0.0))
        except Exception:
            aff_precision = aff_recall = 0.0

        if not np.isfinite(aff_precision):
            aff_precision = 0.0
        if not np.isfinite(aff_recall):
            aff_recall = 0.0

        aff_f1 = 2 * aff_precision * aff_recall / (aff_precision + aff_recall) if (aff_precision + aff_recall) > 0 else 0.0

        return {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "affi precision": round(aff_precision, 4),
            "affi recall": round(aff_recall, 4),
            "affi f1": round(float(aff_f1), 4),
            "class_acc": round(float(class_acc), 4),
        }, fmt_score, pred_intervals, gt_intervals

    except Exception:
     
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "affi precision": 0.0, "affi recall": 0.0, "affi f1": 0.0,
            "class_acc": 0.0,
        }, fmt_score, pred_intervals, gt_intervals


def compute_score(
    predict_str: str,
    ground_truth: Union[str, List[Dict]],
    extra_info: Dict = {},

) -> Tuple[float, float, Dict[str, float], List[Tuple[int, int]], List[Tuple[int, int]]]:

    total_len, ground_type = extra_info.get('series_length', None), extra_info.get('anomaly_type', None)

    result, fmt_score, pred_intervals, gt_intervals = compute_affinity_reward_single_sample(
        predict_str, ground_truth, total_len=total_len, ground_type=ground_type
    )

    if result:
        strict_metrics_avg = (
            result["precision"] + result["recall"] + result["f1"]
            + result["affi precision"] + result["affi recall"] + result["affi f1"]
        ) / 6.0


    else:
        strict_metrics_avg = 0.0

    final_score = strict_metrics_avg * 0.7 + result["class_acc"] * 0.2 + fmt_score * 0.1
    result = {"affi  precision":  result["affi precision"], "affi recall":  result["affi recall"],"affi f1":  result["affi f1"], "class_acc":  result["class_acc"] }


    return final_score, fmt_score, result, pred_intervals, gt_intervals

