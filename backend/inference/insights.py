import datetime as dt
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import pandas as pd
from sqlalchemy import create_engine, text

from inference.prediction import (
    prepare_tickdb_dataframe_for_model,
    CarTrajectoryPredictor,
)
from inference.models import MODEL_PATH
from inference.constants import state, control
from inference.evaluation import (
    trajectory_to_state_positions,
    gates,
    score_modified_against_baseline,
)
from inference.evaluation.scoring import testIntersection


# ---------------------------------------------------------------------------
# 0. Config & types
# ---------------------------------------------------------------------------

TICKDB_URL = "postgresql+psycopg2://telemetry:telemetry@localhost:5432/telemetry"


@dataclass
class ControlModification:
    """
    Represents a single control modification to apply to the base dataframe.
    `apply` should take the *base* df and return a modified copy.

    The `then` method allows composition of modifications:
    mod_combined = mod_a.then(mod_b) applies mod_a, then mod_b.
    """
    name: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]

    def then(self, other: "ControlModification") -> "ControlModification":
        combined_name = f"{self.name}+{other.name}"

        def _apply(df: pd.DataFrame, a=self, b=other) -> pd.DataFrame:
            return b.apply(a.apply(df))

        return ControlModification(name=combined_name, apply=_apply)


@dataclass
class BestModificationInsight:
    """
    Represents the most beneficial control modification (or combination of two),
    relative to baseline.
    """
    name: str                  # e.g. "brakes_minus25pct" or "brakes_minus25pct+gear_plus1"
    kind: str                  # "single" or "combined"
    component_names: List[str] # ["brakes_minus25pct"] or ["brakes_minus25pct", "gear_plus1"]
    score: float               # Δt vs baseline (negative means faster)
    lat_true: pd.Series
    lon_true: pd.Series
    lat_pred: pd.Series
    lon_pred: pd.Series
    gates_with_intersections: Dict


# ---------------------------------------------------------------------------
# 1. Data acquisition
# ---------------------------------------------------------------------------

def load_tick_window(
    engine,
    vehicle_id: str,
    duration_s: float = 5.0,
) -> pd.DataFrame:
    """
    Load ~`duration_s` seconds of aligned tick data for one car.
    Assumes:
      - table: telem_tick
      - timestamp column: ts
      - vehicle identifier column: vehicle_id
      - signals: accx_can, accy_can, speed, gear, aps, nmot,
                 pbrake_f, pbrake_r, VBOX_Lat_Min, VBOX_Long_Minutes, Steering_Angle
    """
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(seconds=duration_s)

    query = text(
        """
        SELECT
            ts AS timestamp,
            accx_can,
            accy_can,
            speed,
            gear,
            aps,
            nmot,
            pbrake_f,
            pbrake_r,
            "VBOX_Lat_Min",
            "VBOX_Long_Minutes",
            "Steering_Angle"
        FROM telem_tick
        WHERE vehicle_id = :vehicle_id
          AND ts >= :start_time
          AND ts <  :end_time
        ORDER BY ts
        """
    )

    df = pd.read_sql(
        query,
        engine,
        params={
            "vehicle_id": vehicle_id,
            "start_time": start_time,
            "end_time": end_time,
        },
    )

    # Ensure time index is monotonic
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


# ---------------------------------------------------------------------------
# 2. Data processing (prep + control modifications)
# ---------------------------------------------------------------------------

def build_default_control_modifications(df_model: pd.DataFrame) -> List[ControlModification]:
    """
    Build a default set of control modifications to try.
    You can easily extend/replace this list from the caller.
    """

    mods: List[ControlModification] = []

    # Baseline: identity (no change)
    mods.append(
        ControlModification(
            name="baseline",
            apply=lambda df: df.copy(),
        )
    )

    # Steering x2 (if present)
    if "steering_angle" in df_model.columns:
        mods.append(
            ControlModification(
                name="steering_x2",
                apply=lambda df: df.assign(steering_angle=df["steering_angle"] * 2.0),
            )
        )
    else:
        print("WARNING: steering_angle not found in df_model columns!")

    # Brakes -25%
    mods.append(
        ControlModification(
            name="brakes_minus25pct",
            apply=lambda df: df.assign(
                pbrake_f=df["pbrake_f"] * 0.75,
                pbrake_r=df["pbrake_r"] * 0.75,
            ),
        )
    )

    # Gear +1
    mods.append(
        ControlModification(
            name="gear_plus1",
            apply=lambda df: df.assign(gear=df["gear"] + 1),
        )
    )

    # Gear -1
    mods.append(
        ControlModification(
            name="gear_minus1",
            apply=lambda df: df.assign(gear=df["gear"] - 1),
        )
    )

    return mods


# ---------------------------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------------------------

def make_predictor() -> CarTrajectoryPredictor:
    return CarTrajectoryPredictor(
        state_cols=state,
        control_cols=control,
        model_path=str(MODEL_PATH / "new_multicar_multistep_model.pt"),
        scaler_path=str(MODEL_PATH / "new_multicar_multistep_scaler.pkl"),
        seq_len=10,
        scale=100.0,
    )


def run_inference_for_modifications(
    predictor: CarTrajectoryPredictor,
    df_model: pd.DataFrame,
    modifications: List[ControlModification],
) -> Tuple[Dict[str, Tuple[pd.Series, pd.Series]], pd.Series, pd.Series]:
    """
    Run predictor for each control modification.
    Returns:
        pred_results: name -> (lat_pred, lon_pred)
        lat_true, lon_true: from unmodified df_model
    """
    # "Ground truth" prediction for the unmodified df_model
    lat_true, lon_true, _, _ = predictor.predict(df_model)

    pred_results: Dict[str, Tuple[pd.Series, pd.Series]] = {}

    for mod in modifications:
        print(f"Running variant: {mod.name}")
        df_mod = mod.apply(df_model)
        _, _, lat_pred, lon_pred = predictor.predict(df_mod)
        pred_results[mod.name] = (lat_pred, lon_pred)

    return pred_results, lat_true, lon_true


# ---------------------------------------------------------------------------
# 4. Evaluation (scoring + gate intersections)
# ---------------------------------------------------------------------------

def gate_key(gate):
    # convert gate [[lon1, lat1], [lon2, lat2]] into a hashable key
    return (tuple(gate[0]), tuple(gate[1]))


def evaluate_variants(
    pred_results: Dict[str, Tuple[pd.Series, pd.Series]],
    df_index: pd.DatetimeIndex,
) -> Tuple[
    Dict[str, float],
    Dict[Tuple[Tuple[float, float], Tuple[float, float]], list],
]:
    """
    - Compute baseline_state from 'baseline' variant
    - Score each modified variant vs baseline
    - Collect gates that any variant intersects

    Returns:
        scores: variant_name -> delta_t
        gates_with_intersections: gate_key -> [variant_names_that_hit_it]
    """

    if "baseline" not in pred_results:
        raise ValueError("pred_results must contain a 'baseline' entry")

    scores: Dict[str, float] = {}
    gates_with_intersections: Dict[
        Tuple[Tuple[float, float], Tuple[float, float]],
        list,
    ] = {}

    # Baseline state
    baseline_lat, baseline_lon = pred_results["baseline"]
    baseline_state = trajectory_to_state_positions(
        baseline_lat,
        baseline_lon,
        df_index,
    )

    # First: record which gates the baseline crosses
    for gate in gates:
        if testIntersection(baseline_state, gate) is not None:
            key = gate_key(gate)
            gates_with_intersections.setdefault(key, []).append("baseline")

    # Then: loop over modified variants, score, and record intersections
    for name, (lat_pred, lon_pred) in pred_results.items():
        if name == "baseline":
            continue

        modified_state = trajectory_to_state_positions(
            lat_pred,
            lon_pred,
            df_index,
        )

        delta_t = score_modified_against_baseline(
            baseline_state,
            modified_state,
            gates,
        )
        scores[name] = delta_t
        print(f"{name}: {delta_t}")

        for gate in gates:
            if testIntersection(modified_state, gate) is not None:
                key = gate_key(gate)
                gates_with_intersections.setdefault(key, []).append(name)

    return scores, gates_with_intersections


from typing import List, Dict, Tuple, Optional

def get_insights(
    vehicle_id: str = "GR86-040-3",
    duration_s: float = 5.0,
    modifications: List[ControlModification] | None = None,
    predictor: Optional[CarTrajectoryPredictor] = None,
    engine = None,
):
    """
    Runs all (single) control modifications and evaluates their Δt vs baseline.

    - Finds beneficial single modifications (numeric score < 0).
    - If there are at least two beneficial singles, tries the combination of the
      best two using ControlModification.then.
    - If the combined modification is even better (more negative Δt) and
      beneficial, it is added to pred_results and its gates merged.
    - Returns:
        lat_true,
        lon_true,
        pred_results,
        gates_with_intersections,
        best_controls,    # list[str], e.g. ["brakes_minus25pct"] or ["brakes_minus25pct", "gear_plus1"]
        best_improvement  # float (seconds) or None if no beneficial mod
    """

    # --- Data acquisition ---
    if engine is None:
        engine = create_engine(TICKDB_URL)

    df_window = load_tick_window(engine, vehicle_id, duration_s=duration_s)

    # --- Data processing: prep for model ---
    df_model = prepare_tickdb_dataframe_for_model(df_window, state, control)

    # If no modifications are passed in, use defaults
    if modifications is None:
        modifications = build_default_control_modifications(df_model)

    # --- Inference for all single mods (including baseline) ---
    if predictor is None:
        predictor = make_predictor()

    pred_results, lat_true, lon_true = run_inference_for_modifications(
        predictor,
        df_model,
        modifications,
    )

    # --- Evaluation for singles ---
    scores, gates_with_intersections = evaluate_variants(
        pred_results,
        df_window.index,
    )

    # Separate numeric from non-numeric scores
    numeric_scores: Dict[str, float] = {
        k: v for k, v in scores.items() if isinstance(v, (float, int))
    }
    beneficial_scores: Dict[str, float] = {
        k: v for k, v in numeric_scores.items() if v < 0.0
    }

    # Defaults if nothing beneficial is found
    best_controls: List[str] = []
    best_improvement: Optional[float] = None

    if not beneficial_scores:
        print("No beneficial control modifications found.")
        return (
            lat_true,
            lon_true,
            pred_results,
            gates_with_intersections,
            best_controls,
            best_improvement,
        )

    # Sort beneficial singles by score (most negative = best)
    sorted_beneficial = sorted(beneficial_scores.items(), key=lambda kv: kv[1])
    best_single_name, best_single_score = sorted_beneficial[0]
    print(f"Best single modification so far: {best_single_name}  Δt={best_single_score:.4f}")

    best_controls = [best_single_name]
    best_improvement = best_single_score

    # --- Try combining the best two beneficial mods (if at least 2) ---
    if len(sorted_beneficial) >= 2:
        second_best_name, second_best_score = sorted_beneficial[1]

        mods_by_name = {mod.name: mod for mod in modifications}
        if best_single_name in mods_by_name and second_best_name in mods_by_name:
            mod_a = mods_by_name[best_single_name]
            mod_b = mods_by_name[second_best_name]

            # Requires ControlModification.then(a, b) to be implemented
            combined_mod = mod_a.then(mod_b)

            # Build a small list: baseline + combined
            baseline_mods = [m for m in modifications if m.name == "baseline"]
            if baseline_mods:
                combo_mods = baseline_mods + [combined_mod]

                combo_pred_results, _, _ = run_inference_for_modifications(
                    predictor,
                    df_model,
                    combo_mods,
                )

                combo_scores, combo_gates_with_intersections = evaluate_variants(
                    combo_pred_results,
                    df_window.index,
                )

                combo_score = combo_scores.get(combined_mod.name)

                # If combo is numeric, beneficial, and better (more negative) than best single
                if isinstance(combo_score, (float, int)) and combo_score < 0.0 and combo_score < best_single_score:
                    print(
                        f"Combined modification {combined_mod.name} "
                        f"is better than best single: Δt={combo_score:.4f}"
                    )

                    # Add combined result to overall pred_results
                    pred_results[combined_mod.name] = combo_pred_results[combined_mod.name]

                    # Merge gate intersections (union of variants per gate)
                    for gate_key, variant_names in combo_gates_with_intersections.items():
                        existing = gates_with_intersections.setdefault(gate_key, [])
                        for vn in variant_names:
                            if vn not in existing:
                                existing.append(vn)

                    # Update "best" info to the combo
                    best_controls = [best_single_name, second_best_name]
                    best_improvement = combo_score
                else:
                    print(
                        f"Combined modification {combined_mod.name} not better than best single "
                        f"(combo Δt={combo_score}, best single Δt={best_single_score})"
                    )

    # Optional: print all numeric scores for debugging
    print("\n=== Variant scores (numeric only, Δt vs baseline) ===")
    for name, s in sorted(numeric_scores.items(), key=lambda kv: kv[1]):
        print(f"{name:20s}: {s:.4f}")
    if len(numeric_scores) != len(scores):
        print("Non-numeric scores:")
        for name, s in scores.items():
            if name not in numeric_scores:
                print(f"{name:20s}: {s}")

    return (
        lat_true,
        lon_true,
        pred_results,
        gates_with_intersections,
        best_controls,
        best_improvement,
    )