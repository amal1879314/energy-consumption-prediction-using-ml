# model_utils.py
import os
import pickle
import pandas as pd
import numpy as np
import re
from collections.abc import Sequence

# ---------------------
# Model loading helpers
# ---------------------
def _extract_feature_names_from_model(obj):
    """
    Try to extract ordered feature names from:
      - saved meta inside a dict {"model": ..., "meta": {"feature_names": [...]}}
      - sklearn estimator attribute feature_names_in_
      - lightgbm booster_.feature_name() or _Booster.feature_name()
    Returns list or None.
    """
    if isinstance(obj, dict):
        meta = obj.get("meta", {}) or {}
        if meta.get("feature_names"):
            return list(meta["feature_names"])
        model = obj.get("model")
    else:
        model = obj

    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    try:
        if hasattr(model, "booster_") and model.booster_ is not None:
            return list(model.booster_.feature_name())
    except Exception:
        pass

    try:
        if hasattr(model, "_Booster") and model._Booster is not None:
            return list(model._Booster.feature_name())
    except Exception:
        pass

    return None

def load_model(path):
    """
    Loads a pickle object from path.
    Returns (model, meta_dict) where meta may contain 'feature_names'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        meta = obj.get("meta", {}) or {}
    else:
        model = obj
        meta = {}
        fn = _extract_feature_names_from_model(obj)
        if fn:
            meta["feature_names"] = fn

    return model, meta

# ---------------------
# Data cleaning helpers
# ---------------------
def _clean_value_for_numeric(v):
    """Try to clean and convert common numeric-like strings to numeric."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating, bool)):
        return v
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode()
        except Exception:
            v = str(v)
    if isinstance(v, str):
        s = v.strip()
        # common missing tokens
        if s == "" or s.lower() in {"na", "n/a", "none", "null", "-"}:
            return np.nan
        # remove thousands separators
        s = s.replace(",", "")
        # remove unit suffixes like 'kwh' or ' kWh' (common), keep numeric prefix
        m = re.match(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?).*", s)
        if m:
            s_num = m.group(1)
            try:
                if "." in s_num or "e" in s_num.lower():
                    return float(s_num)
                else:
                    return int(s_num)
            except Exception:
                try:
                    return float(s_num)
                except Exception:
                    return s
        try:
            if re.match(r"^[+-]?\d+$", s):
                return int(s)
            if re.match(r"^[+-]?\d*\.\d+(?:[eE][+-]?\d+)?$", s) or re.match(r"^[+-]?\d+(?:[eE][+-]?\d+)$", s, flags=re.I):
                return float(s)
            return float(s)
        except Exception:
            return s
    return v

def _attempt_column_numeric_coercion(df, columns=None):
    """
    Try to coerce columns to numeric types where possible.
    Returns (df_converted, report).
    report: {'coerced_to_numeric': [...], 'left_as_object': [...]}
    """
    df = df.copy()
    report = {"coerced_to_numeric": [], "left_as_object": []}
    cols = columns if columns is not None else df.columns.tolist()

    for col in cols:
        cleaned = df[col].apply(_clean_value_for_numeric)
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.notna().sum() > 0:
            df[col] = coerced
            report["coerced_to_numeric"].append(col)
        else:
            df[col] = cleaned
            report["left_as_object"].append(col)

    return df, report

# ---------------------
# Sequence handling
# ---------------------
def _is_list_like(x):
    if x is None:
        return False
    if isinstance(x, (str, bytes)):
        return False
    return isinstance(x, Sequence) or (hasattr(x, "shape") and getattr(x, "shape") != ())

def _handle_sequence_columns(df):
    df = df.copy()
    expanded = {}
    squeezed = []
    variable = {}

    for col in list(df.columns):
        lengths = []
        sample_vals = []
        has_seq = False
        for val in df[col].dropna().iloc[:500]:
            if _is_list_like(val):
                has_seq = True
                try:
                    lengths.append(len(val))
                    sample_vals.append(val)
                except Exception:
                    pass
        if not has_seq:
            continue

        unique_lengths = set(lengths)
        if len(unique_lengths) == 0:
            variable[col] = {"lengths": unique_lengths, "samples": sample_vals[:5]}
            continue

        if len(unique_lengths) == 1:
            k = unique_lengths.pop()
            if k == 1:
                def _squeeze_cell(v):
                    if _is_list_like(v):
                        try:
                            return v[0]
                        except Exception:
                            return v
                    return v
                df[col] = df[col].apply(_squeeze_cell)
                squeezed.append(col)
            else:
                k = int(k)
                new_cols = [f"{col}_{i}" for i in range(k)]
                expanded_values = []
                aborted = False
                for val in df[col].values:
                    if _is_list_like(val):
                        try:
                            if len(val) == k:
                                expanded_values.append(list(val))
                            else:
                                expanded[col] = ("inconsistent_length", len(val))
                                aborted = True
                                break
                        except Exception:
                            expanded_values.append([np.nan]*k)
                    else:
                        expanded_values.append([np.nan]*k)
                if aborted:
                    continue
                exp_df = pd.DataFrame(expanded_values, columns=new_cols, index=df.index)
                df = pd.concat([df.drop(columns=[col]), exp_df], axis=1)
                expanded[col] = k
        else:
            variable[col] = {"lengths": unique_lengths, "samples": sample_vals[:5]}

    seq_report = {"expanded": expanded, "squeezed": squeezed, "variable_length": variable}
    return df, seq_report

# ---------------------
# Feature alignment
# ---------------------
def align_features(df, feature_names, fill_value=np.nan, verbose=False):
    if feature_names is None:
        return df

    df = df.copy()
    incoming = list(df.columns)

    missing = [c for c in feature_names if c not in incoming]
    for c in missing:
        df[c] = fill_value

    extras = [c for c in incoming if c not in feature_names]
    if extras:
        df = df.drop(columns=extras)

    aligned = df[feature_names]

    if verbose:
        print("align_features debug:")
        print(" expected_len=", len(feature_names))
        print(" incoming_len=", len(incoming))
        print(" missing=", missing)
        print(" extras_dropped=", extras)
        print(" aligned_shape=", aligned.shape)

    return aligned

# ---------------------
# Public prepare_input
# ---------------------
def prepare_input(data, expected_feature_names=None, fill_value=np.nan, coerce_numeric=True):
    """
    Create DataFrame from dict or list-of-dicts, handle sequences, align and coerce numeric.
    Returns (df, conversion_report).
    """
    if isinstance(data, dict):
        if "features" in data and isinstance(data["features"], dict):
            data = data["features"]
        rec = {}
        for k, v in data.items():
            rec[k] = v
        df = pd.DataFrame([rec])
    elif isinstance(data, list):
        df = pd.DataFrame.from_records(data)
    else:
        raise ValueError("Unsupported input format. Send a dict or list of dicts.")

    df, seq_report = _handle_sequence_columns(df)

    alignment_report = None
    if expected_feature_names:
        incoming = list(df.columns)
        missing = [c for c in expected_feature_names if c not in incoming]
        extras = [c for c in incoming if c not in expected_feature_names]
        for c in missing:
            df[c] = fill_value
        if extras:
            df = df.drop(columns=extras)
        df = df[expected_feature_names]
        alignment_report = {"expected_len": len(expected_feature_names), "incoming_len": len(incoming),
                            "missing": missing, "extras": extras}

    numeric_report = None
    if coerce_numeric:
        df, numeric_report = _attempt_column_numeric_coercion(df, columns=df.columns.tolist())

    report = {"sequence_handling": seq_report, "alignment": alignment_report, "numeric_coercion": numeric_report}

    if seq_report["variable_length"]:
        raise ValueError(f"Found sequence-valued columns with variable lengths: {seq_report['variable_length']} — decide how to handle them.")

    return df, report