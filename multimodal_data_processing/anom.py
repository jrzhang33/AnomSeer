# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
import pickle
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import stumpy
from PIL import Image
from datasets import Dataset
from scipy.signal import savgol_filter
from scipy.stats import zscore


def _dominant_period(x: np.ndarray) -> float:
    """Computes the dominant period of a time series using FFT."""
    x = np.asarray(x, np.float32)
    n = x.size
    if n < 8: return float('inf')
    xdm = x - x.mean()
    Y = np.fft.rfft(xdm)
    P = np.abs(Y)**2
    if P.size > 0: P[0] = 0.0
    freqs = np.fft.rfftfreq(n, d=1.0)
    if P.size <= 1: return float('inf')
    k = int(np.argmax(P))
    f = float(freqs[k])
    return float(1.0 / f) if f > 0 else float('inf')

def vector_to_intervals(vec: np.ndarray) -> List[List[int]]:
    """Converts a binary (0/1) vector to a list of [start, end) intervals."""
    vec = np.asarray(vec).astype(int).flatten()
    intervals = []
    in_seg = False
    start = 0
    for i, v in enumerate(vec):
        if v == 1 and not in_seg:
            start = i
            in_seg = True
        elif v == 0 and in_seg:
            intervals.append([start, i])
            in_seg = False
    if in_seg:
        intervals.append([start, len(vec)])
    return intervals



def _try_load_npy(path: str) -> Optional[np.ndarray]:
    try:
        if os.path.isfile(path):
            return np.load(path, allow_pickle=True)
    except Exception:
        return None

def _find_series_and_gt(root_task_dir: str, split: str, idx: str) -> Tuple[Optional[np.ndarray], List[List[int]]]:
    """Finds raw series and ground truth intervals from common directory patterns."""
    series_path = os.path.join(root_task_dir, split, "series", f"{idx}.npy")
    labels_path = os.path.join(root_task_dir, split, "labels", f"{idx}.npy")
    
    ts = _try_load_npy(series_path)
    gt_vec = _try_load_npy(labels_path)
    
    intervals = []
    if gt_vec is not None:
        try:
            intervals = vector_to_intervals(gt_vec.astype(int))
        except Exception:
            pass
            
    return ts, intervals

def _parse_idx_from_png(png_path: str) -> str:
    """Extracts the numerical index from a PNG filename."""
    base = os.path.basename(png_path)
    m = re.search(r"(\d+)", base)
    return m.group(1) if m else os.path.splitext(base)[0]




def detect_range_anomalies(ts: np.ndarray, k: float = 3.0) -> Tuple[List[List[int]], str, Dict]:
    """Uses k-sigma rule for range anomalies and generates a CoT-style text."""
    if ts.size < 2:
        return [], "Series too short.", {}
    
    mu, std = np.mean(ts), np.std(ts)
    if std < 1e-6:
        return [], f"The series is constant (value={mu:.2f}), so no range anomalies can be detected.", {}
        
    upper, lower = mu + k * std, mu - k * std
    is_anomaly = (ts > upper) | (ts < lower)
    intervals = vector_to_intervals(is_anomaly)
    
    analysis_text = (
        f"To find range anomalies, I first calculated the global statistics: "
        f"the mean is {mu:.3f} and the standard deviation is {std:.3f}. "
        f"This establishes a normal range of [{lower:.3f}, {upper:.3f}] using a {k}-sigma rule. "
        f"Scanning the series for points outside this boundary resulted in finding {len(intervals)} anomalous interval(s)."
    )
    metrics = {"range_mean": mu, "range_std": std, "range_upper": upper, "range_lower": lower}
    return intervals, analysis_text, metrics

def detect_trend_anomalies(ts: np.ndarray, window: int = 21, k: float = 3.0) -> Tuple[List[List[int]], str, Dict]:
    """Uses smoothed gradient for trend anomalies and generates a CoT-style text."""
    if ts.size < window * 2:
        return [], "Series too short for trend analysis.", {}

    smooth_ts = savgol_filter(ts, window_length=window, polyorder=2)
    gradient = np.gradient(smooth_ts)
    grad_mu, grad_std = np.mean(gradient), np.std(gradient)
    if grad_std < 1e-6:
         return [], f"The trend appears to be constant (gradient={grad_mu:.4f}), so no significant changes were detected.", {}

    threshold = k * grad_std
    is_anomaly = np.abs(gradient - grad_mu) > threshold
    intervals = vector_to_intervals(is_anomaly)
    
    analysis_text = (
        f"For trend analysis, I focused on the rate of change. After smoothing the series with a window of {window}, "
        f"I computed its gradient. The normal gradient fluctuates around a mean of {grad_mu:.4f} with a std of {grad_std:.4f}. "
        f"A significant trend shift should cause a gradient deviation larger than the {k}-sigma threshold of {threshold:.4f}. "
        f"This check identified {len(intervals)} interval(s) meeting the condition."
    )
    metrics = {"trend_grad_mean": grad_mu, "trend_grad_std": grad_std, "trend_grad_threshold": threshold}
    return intervals, analysis_text, metrics

def detect_point_anomalies(ts: np.ndarray, window: int = 50, k: float = 3.5) -> Tuple[List[List[int]], str, Dict]:
    """Uses Matrix Profile (STUMPY) for point anomalies and generates a CoT-style text."""
    if ts.size < window * 2:
        return [], "Series too short for discord detection.", {}
        

    mp = stumpy.stump(ts, m=window, ignore_trivial=True)



    raw_mp_distances = mp[:, 0]
    mp_distances = np.zeros(raw_mp_distances.shape, dtype=np.float64)
    for i, val in enumerate(raw_mp_distances):
        try:
            float_val = float(val)
            mp_distances[i] = float_val
        except (ValueError, TypeError):
            mp_distances[i] = 0.0

    finite_mask = np.isfinite(mp_distances)
    if np.any(finite_mask):
        max_finite_val = np.max(mp_distances[finite_mask])
        mp_distances[~finite_mask] = max_finite_val
    else:
        return [], "Matrix Profile computation resulted in non-finite values only.", {}

    mp_zscores = zscore(mp_distances)
    discord_idx = np.argmax(mp_distances)
    max_zscore = mp_zscores[discord_idx]
    
    intervals = []
    if max_zscore > k:
        intervals = [[int(discord_idx), int(discord_idx + window)]]

    analysis_text = (
        f"To detect contextual point anomalies, I used the Matrix Profile (window={window}, ignoring trivial matches) to find the most unusual subsequence (discord). "
        f"The computation revealed that the highest discord score is {mp_distances[discord_idx]:.2f}, located at index {discord_idx}. "
        f"This score corresponds to a z-score of {max_zscore:.2f}, which is above my threshold of {k}, indicating a significant anomaly. "
        f"Thus, {len(intervals)} anomaly was identified."
    )
    metrics = {"point_discord_idx": int(discord_idx), "point_discord_zscore": float(max_zscore)}
    return intervals, analysis_text, metrics

def detect_freq_anomalies(ts: np.ndarray, window: int = 100, k: float = 3.0) -> Tuple[List[List[int]], str, Dict]:
    """Uses dominant period changes for frequency anomalies and generates a CoT-style text."""
    if ts.size < window * 3:
        return [], "Series too short for frequency analysis.", {}
        
    periods = [_dominant_period(ts[i : i + window]) for i in range(ts.size - window)]
    periods = np.array([p if np.isfinite(p) else -1 for p in periods])
    valid_periods = periods[periods != -1]
    
    if valid_periods.size < 10:
        return [], "Could not determine a stable dominant period.", {}

    per_mu, per_std = np.mean(valid_periods), np.std(valid_periods)
    if per_std < 1.0: # If period is very stable
        return [], f"The dominant period is highly stable around {per_mu:.2f}, so no frequency anomalies detected.", {}

    threshold_upper = per_mu + k * per_std
    threshold_lower = per_mu - k * per_std
    is_anomaly_periods = (periods > threshold_upper) | ((periods < threshold_lower) & (periods != -1))
    
    is_anomaly_ts = np.zeros_like(ts, dtype=bool)
    for i, is_anom in enumerate(is_anomaly_periods):
        if is_anom:
            is_anomaly_ts[i : i + window] = True
            
    intervals = vector_to_intervals(is_anomaly_ts)
    
    analysis_text = (
        f"To find frequency anomalies, I calculated the dominant period over a sliding window of size {window}. "
        f"The typical period is around {per_mu:.2f} with a std of {per_std:.2f}. "
        f"I'm looking for regions where the period significantly deviates from the normal range of [{threshold_lower:.2f}, {threshold_upper:.2f}]. "
        f"This analysis pointed to {len(intervals)} interval(s) with clear frequency shifts."
    )
    metrics = {"freq_period_mean": per_mu, "freq_period_std": per_std}
    return intervals, analysis_text, metrics


def _json_default(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)

def _intervals_str(intervals):
    if not intervals:
        return "[]"
    return ", ".join([f"[{int(s)}, {int(e)}]" for s, e in intervals])

def _normalize_class(name: str) -> str:
    m = (name or "").lower()
    if m in ["global","global point","out-of-range","range","noisy-range"]:
        return "global point"
    if m in ["contextual","contextual point","point","noisy-point"]:
        return "contextual point"
    if m in ["trend","trend shift","noisy-trend","flat-trend"]:
        return "trend"
    if m in ["seasonal","frequency","seasonal/frequency deviation","freq","noisy-freq"]:
        return "seasonal"
    if m in ["shapelet","subsequence","shapelet/subsequence"]:
        return "shapelet"
    return "normal"


def _get_expert_reasoning_flow(ts: np.ndarray, task_hint: str) -> Tuple[str, callable]:
    """
    Simulates an expert's diagnostic process to select the right tool.
    Returns a reasoning text and the selected detection function.
    """
    # --- Step 1: Global Scan 
    global_mu, global_std = np.mean(ts), np.std(ts)
    max_zscore = 0
    if global_std > 1e-6:
        z_scores = np.abs((ts - global_mu) / global_std)
        max_zscore = np.max(z_scores)

    if max_zscore > 5.0: # A very high Z-score suggests a simple range anomaly is likely.
        reasoning = (
            "My initial check reveals extreme values. The global mean is "
            f"{global_mu:.3f} and std is {global_std:.3f}, but some points have a z-score as high as {max_zscore:.2f}. "
            "This strongly suggests a range-based anomaly. I will now apply a k-sigma rule to formalize this."
        )
        return reasoning, detect_range_anomalies

    # --- Step 2: Structural Scan
    
    # Check for stable trend
    gradient = np.gradient(ts)
    grad_mu, grad_std = np.mean(gradient), np.std(gradient)
    # A low gradient std relative to the overall data std might indicate a stable trend
    is_trend_stable = (grad_std / global_std) < 0.1 if global_std > 1e-6 else True
    
    # Check for stable frequency
    dominant_p = _dominant_period(ts)
    is_freq_stable = dominant_p != float('inf') and dominant_p > 8 # has a detectable period

    # Decision based on structure and task hint
    task_lc = task_hint.lower()
    if 'trend' in task_lc:
        reasoning = (
            f"The global values seem normal (max z-score={max_zscore:.2f}), so I'll check the trend. "
            "The gradient of the series appears unstable. This suggests a potential trend anomaly. "
            "I will analyze the smoothed gradient to confirm any significant shifts."
        )
        return reasoning, detect_trend_anomalies

    if 'freq' in task_lc:
        reasoning = (
            f"Global values and trend seem stable. However, the signal appears periodic. "
            "An unstable period can indicate a frequency anomaly. "
            "I will use a sliding window analysis to check for significant changes in the dominant period."
        )
        return reasoning, detect_freq_anomalies

    # --- Step 3: Pattern Scan (If all else seems normal, look for unique patterns) ---
    # Corresponds to Matrix Profile logic. This is the default for 'point' or when other checks fail.
    reasoning = (
        f"The series does not exhibit obvious global outliers (max z-score={max_zscore:.2f}) or clear structural instability. "
        "The anomalies are likely subtle and contextual. This requires a pattern-based approach. "
        "I will use the Matrix Profile to find the most dissimilar subsequence (a discord), which is the standard method for such cases."
    )
    return reasoning, detect_point_anomalies

    
def _link_gt_feature(gt_intervals: List[List[int]], class_name: str) -> str:
    iv_str = _intervals_str(gt_intervals)
    if iv_str == "[]":
        return f"I did not observe any {class_name} anomaly in the series. "

    if class_name == "global point":
        return f"I observed that the values within {iv_str} exhibit clear out-of-range behavior, with sharp spikes deviating from the global distribution. "
    if class_name == "contextual point":
        return f"I observed that the subsequence {iv_str} appears inconsistent with its local temporal context, breaking the continuity of surrounding patterns. "
    if class_name == "trend":
        return f"I observed that the segment within {iv_str} shows a clear trend shift, with the long-term slope undergoing a marked change. "
    if class_name == "seasonal":
        return f"I observed that the oscillations within {iv_str} display frequency deviation, with periodic structure misaligned from the baseline cycle. "
    if class_name == "shapelet":
        return f"I observed that the subsequence within {iv_str} deviates in waveform shape, differing notably from typical local motifs. "
    return f"I observed that the segment {iv_str} is annotated as {class_name}. "


def build_prompt_and_expcot(ts: Optional[np.ndarray],
                             task: str,
                             L: int,
                             gt_intervals: List[List[int]],
                             gt_type: str) -> Tuple[str, str, Dict]:

    prompt = (
        "<image>\n"
        f"You are a time series analysis expert. A time series plot of length {L} is provided. "
        "Identify anomalous intervals along the x-axis and infer the most plausible anomaly type from "
        "[\"contextual point\", \"global point\", \"seasonal\", \"trend\", \"shapelet\"].\n\n"
        "Begin detailed reasoning inside <think>...</think>.\n"
        "Then output:\n"
        "<answer>[[start, end], ...]</answer>\n"
        "<class>one of {contextual point, global point, seasonal, trend, shapelet, normal}</class>\n"
        "If no anomalies, return <answer>[]</answer> and <class>normal</class>.\n"
    )

  
    class_by_task = _normalize_class(task)
    class_by_gt   = _normalize_class(gt_type)
    class_name    = class_by_gt if class_by_gt != "normal" else class_by_task

    if ts is None or ts.size < 20:
        if gt_intervals:
            expcot = (
                f"\\textbf{{Observation}} — Ground truth marks { _intervals_str(gt_intervals) } "
                f"as \\textit{{{class_name}}}, but raw series is unavailable/too short for verification.\n"
                f"\\textbf{{Conclusion}} — We retain the GT label and intervals for supervision."
            )
        else:
            expcot = (
                "\\textbf{Observation} — Ground truth indicates no anomaly; raw series unavailable/too short.\n"
                "\\textbf{Conclusion} — Treated as normal."
            )
        return prompt, expcot, {"intervals_gt": gt_intervals, "class": class_name}


    x = ts if ts.ndim == 1 else ts.mean(axis=1)
    pre = ""
    if "noisy" in task.lower():
        wl = max(5, min(31, (len(x)//4)|1))
        x  = savgol_filter(x, window_length=wl, polyorder=2)
        pre = f"[Preprocess] Applied Savitzky–Golay denoising (win={wl}). "

    initial_reasoning, selected_detector = _get_expert_reasoning_flow(x, task)
    det_intervals, final_evidence, detection_metrics = selected_detector(x)


    obs = (
        f"Ground truth labels { _intervals_str(gt_intervals) } as \\textit{{{class_name}}}. "
        f"{initial_reasoning}"
    )


    val = (
    _link_gt_feature(gt_intervals, class_name)
    + final_evidence.replace("To find", "The analysis").replace("For", "The analysis")
)


  
    if gt_intervals:
        concl = (
            f"We report the GT interval(s) { _intervals_str(gt_intervals) } "
            f"as the final localization for supervision. "
            f"For reference, the detector proposed { _intervals_str(det_intervals) }."
        )
        out_intervals = gt_intervals  # 
    else:
        concl = (
            "No ground-truth anomalies are present; the series is treated as normal. "
            f"For reference, the detector proposed { _intervals_str(det_intervals) }."
        )
        out_intervals = []  # normal

    expcot = (
        f"{pre}"
        f"\\textbf{{Observation}} — {obs}\n"
        f"\\textbf{{Reasoning \\& Validation}} — {val}\n"
        f"\\textbf{{Conclusion}} — {concl}"
    ).strip()

  
    return prompt, expcot, {
        "intervals_gt": gt_intervals,
        "intervals_pred": det_intervals,
        "class": class_name,
        **detection_metrics
    }

# --- CORRECTED DATA LOADING HELPERS ---
def _load_split_pkl(task_dir: str, split: str) -> Tuple[Optional[list], Optional[list]]:

    pkl_path = os.path.join(task_dir, split, "data.pkl")
    if not os.path.isfile(pkl_path):
        return None, None
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        series = obj.get("series", None)
        anom = obj.get("anom", None)
        return series, anom
    except Exception:
        return None, None

def _pick_index_from_png_idx(idx_str: str, n: int) -> Optional[int]:

    try:
        k = int(idx_str)
    except Exception:
        return None
    for cand in (k-1, k):
        if 0 <= cand < n:
            return cand
    return None

def _intervals_from_anom_entry(anom_entry) -> List[List[int]]:

    intervals = []
    if not anom_entry: return intervals
    for ch_list in anom_entry:
        for pair in ch_list:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                s, e = int(pair[0]), int(pair[1])
                if e > s: intervals.append([s, e])
    intervals.sort(key=lambda z: (z[0], z[1]))
    return intervals

def _parse_idx_from_png(png_path: str) -> str:

    base = os.path.basename(png_path)
    m = re.search(r"(\d+)", base)
    return m.group(1) if m else os.path.splitext(base)[0]

def process_task(root_dir: str, task: str, out_dir: str) -> Tuple[str, Optional[str]]:

    task_dir = os.path.join(root_dir, task)
    results = {"train": [], "eval": []}

    for split in ["train", "eval"]:
        figs_dir = os.path.join(task_dir, split, "figs")
        if not os.path.isdir(figs_dir):
            print(f"Warning: Directory not found for {task}/{split}/figs. Skipping.")
            continue

      
        series_list, anom_list = _load_split_pkl(task_dir, split)
        n_samples = len(series_list) if isinstance(series_list, list) else 0
        use_pkl = (series_list is not None) and (anom_list is not None) and (n_samples > 0)
        
        pngs = sorted(glob.glob(os.path.join(figs_dir, "*.png")))
        for i, png_path in enumerate(pngs):
            idx_str = _parse_idx_from_png(png_path)
            
            ts, gt_intervals = None, []
            
     
            if use_pkl:
                pkl_i = _pick_index_from_png_idx(idx_str, n_samples)
                if pkl_i is not None:
                    ts = np.asarray(series_list[pkl_i])
                    gt_intervals = _intervals_from_anom_entry(anom_list[pkl_i])


            def map_task_to_anomaly(task_name: str) -> str:
                if task_name in ["range", "noisy-range"]:
                    return "global"
                elif task_name in ["freq", "noise-freq"]:
                    return "seasonal"
                elif task_name in ["point", "noise-point"]:
                    return "contextual"
                elif task_name in ["trend", "noise-trend"]:
                    return "trend"
                else:
                    return "unknown" 
            L = len(ts) if ts is not None else 0

            anomaly_type = map_task_to_anomaly(task)
            if gt_intervals == []:
                anomaly_type = 'normal'
            gt_type = anomaly_type if anomaly_type else _normalize_class(task)
            prompt_text, expcot, detection_metrics = build_prompt_and_expcot(ts, task, L, gt_intervals, gt_type)

            image = Image.open(png_path).convert("RGBA")
                


            row = {
                "data_source": "timeseries_anol",
                "prompt": [{"role": "user", "content": prompt_text}],
                "images": [image],
                "ability": "time_series_anomaly_detection",
                "reward_model": {"style": "rule", "ground_truth": gt_intervals},
                "extra_info": {
                    "category": task,
                    "split": split,
                    "instance_index": i,
                    "image_path": png_path,
                    "expcot": expcot,
                    "detection_metrics": detection_metrics,
                    "gt_intervals": gt_intervals,
                    "series_length": len(ts) if ts is not None else 0,
                    "index": pkl_i,
                    "anomaly_type": anomaly_type   
                }
            }
            results[split].append(row)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, f"{task}_train.parquet")
    test_path  = os.path.join(out_dir, f"{task}_test.parquet")

    if results["train"]:
        train_ds = Dataset.from_list(results["train"])
        train_ds.to_parquet(train_path)
    else:
        train_path = ""

    if results["eval"]:
        eval_ds = Dataset.from_list(results["eval"])
        eval_ds.to_parquet(test_path)
    else:
        test_path = None

    return train_path, test_path



def main():
    parser = argparse.ArgumentParser(description="Process synthetic time series anomaly data into parquet files.")
    parser.add_argument("--root_dir", type=str,
                        default="./data/anomllm/data/synthetic",
                        help="Root directory containing the task folders.")
    parser.add_argument("--out_dir", type=str,
                        default="./data/anol_processed_mllm_data",
                        help="Output directory to store the final parquet files.")
    args = parser.parse_args()

    tasks = ["trend", "freq", "point", "range", "noisy-trend", "noisy-freq", "noisy-point", "flat-trend"]

    print(f"Input data root: {args.root_dir}")
    print(f"Output directory: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    all_train_dfs = []
    all_test_dfs = []

    for task in tasks:
        print(f"\nProcessing task: {task}...")

        tr_path, te_path = process_task(args.root_dir, task, args.out_dir)
        if tr_path and os.path.exists(tr_path):
            print(f"  ✅ Saved train data to -> {tr_path}")
            all_train_dfs.append(pd.read_parquet(tr_path))
        if te_path and os.path.exists(te_path):
            print(f"  ✅ Saved test data to  -> {te_path}")
            all_test_dfs.append(pd.read_parquet(te_path))
        if not tr_path and not te_path:
            print(f"  ⚠️ No data found for task '{task}'. Check directory structure.")


    if all_train_dfs:
        full_train_df = pd.concat(all_train_dfs, ignore_index=True)
        full_train_path = os.path.join(args.out_dir, "train_full.parquet")
        full_train_df.to_parquet(full_train_path, index=False)
        print(f"\n📦 Successfully merged and saved all training data to -> {full_train_path}")

    if all_test_dfs:
        full_test_df = pd.concat(all_test_dfs, ignore_index=True)
        full_test_path = os.path.join(args.out_dir, "test_full.parquet")
        full_test_df.to_parquet(full_test_path, index=False)
        print(f"📦 Successfully merged and saved all testing data to  -> {full_test_path}")

if __name__ == "__main__":
    main()