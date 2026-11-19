"""Continuous-action diagnostics with explicit units and tolerance definitions."""
from __future__ import annotations
import numpy as np

THRESHOLDS = (.025, .05, .10, .15, .20, .30, .50)


def action_metrics(prediction, target, action_std):
    prediction, target, action_std = [np.asarray(v, dtype=np.float64) for v in (prediction, target, action_std)]
    assert prediction.shape == target.shape and prediction.ndim == 3 and prediction.shape[-1] == 6
    assert np.isfinite(prediction).all() and np.isfinite(target).all() and (action_std > 0).all()
    error = prediction-target
    absolute = np.abs(error)
    scaled = absolute/action_std
    residual_sum = np.sum(error**2, axis=(0, 1))
    total_sum = np.sum((target-target.mean(axis=(0, 1)))**2, axis=(0, 1))
    assert (total_sum > 0).all(), 'R2 is not defined for constant targets'
    r2 = 1-residual_sum/total_sum
    return {
        'mae_recorded_units': float(absolute.mean()),
        'mse_recorded_units_squared': float(np.mean(error**2)),
        'rmse_recorded_units': float(np.sqrt(np.mean(error**2))),
        'normalized_mae': float(scaled.mean()),
        'normalized_rmse': float(np.sqrt(np.mean((error/action_std)**2))),
        'r2_uniform_joint_mean': float(r2.mean()),
        'mae_by_joint': absolute.mean(axis=(0, 1)).tolist(),
        'rmse_by_joint': np.sqrt(np.mean(error**2, axis=(0, 1))).tolist(),
        'normalized_mae_by_joint': scaled.mean(axis=(0, 1)).tolist(),
        'normalized_rmse_by_joint': np.sqrt(np.mean((error/action_std)**2, axis=(0, 1))).tolist(),
        'r2_by_joint': r2.tolist(),
        'normalized_error_p50': float(np.quantile(scaled, .50)),
        'normalized_error_p90': float(np.quantile(scaled, .90)),
        'normalized_error_p95': float(np.quantile(scaled, .95)),
        'component_hit_rate_pct': {str(t): float(100*np.mean(scaled<=t)) for t in THRESHOLDS},
        'six_joint_vector_hit_rate_pct': {str(t): float(100*np.mean(np.all(scaled<=t, axis=2))) for t in THRESHOLDS},
        'first_action_normalized_mae': float(scaled[:, 0, :].mean()),
        'first_action_component_hit_0_1sigma_pct': float(100*np.mean(scaled[:, 0, :]<=.1)),
        'first_action_vector_hit_0_1sigma_pct': float(100*np.mean(np.all(scaled[:, 0, :]<=.1, axis=1))),
        'probe_count': int(prediction.shape[0]),
        'horizon': int(prediction.shape[1]),
        'component_count': int(prediction.size),
        'vector_count': int(prediction.shape[0]*prediction.shape[1]),
    }


def curve_row(name, step, result):
    return {'model':name, 'step':step,
            **{k:result[k] for k in ['mae_recorded_units','rmse_recorded_units','normalized_mae',
               'normalized_rmse','r2_uniform_joint_mean','normalized_error_p95']},
            'component_hit_0_1sigma_pct': result['component_hit_rate_pct']['0.1'],
            'vector_hit_0_1sigma_pct': result['six_joint_vector_hit_rate_pct']['0.1']}
