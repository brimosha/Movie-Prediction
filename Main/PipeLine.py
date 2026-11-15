#!/usr/bin/env python3
"""
Movie gross -> F-factor prediction pipeline (leak-free CV, exhaustive evaluation).

Design notes
------------
* Test set is NEVER used for fitting, tuning, model selection, thresholds or
  conformal calibration.  CV happens on train only.
* Target transform (Yeo-Johnson on gross) is fitted INSIDE every fold.
* F-factor = (gross - budget) / budget   (budget is a feature, so predicted F-factor is
  derived from predicted gross).
"""
import argparse
import inspect
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor, VotingRegressor)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             explained_variance_score, f1_score,
                             matthews_corrcoef, max_error,
                             mean_absolute_error, mean_squared_error,
                             median_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
HIT_F_FACTOR = 0.5          # "success" threshold requested: F-factor > 0.5
PROFIT_F_FACTOR = 0.0       # break-even
F_FACTOR_BIN_EDGES = [-np.inf, -0.5, 0.0, 0.5, 2.0, 5.0, np.inf]
F_FACTOR_BIN_LABELS = ["<-0.5 deep loss", "-0.5..0 loss", "0..0.5 low return",
                  "0.5..2 good", "2..5 strong", ">5 blockbuster"]
LOW_Q, HIGH_Q = 0.05, 0.95   # the 5 / 95 quantile outlier definition
TARGET_RAW, TARGET_YJ = "gross", "gross_yj"   # gross_yj is a leak -> never a feature
SCORE_ENCODED = ["actor1_score", "actor1_is_star", "actor2_score", "actor2_is_star",
                 "actor3_score", "actor3_is_star", "cast_star_score", "cast_star_max",
                 "n_star_actors", "director1_score", "director1_is_star"]
POST_RELEASE = ["num_voted_users", "num_user_for_reviews", "num_critic_for_reviews",
                "movie_facebook_likes", "imdb_score"]


# ----------------------------------------------------------------------------
# PRINT HELPERS
# ----------------------------------------------------------------------------
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, x):
        for f in self.files:
            f.write(x)

    def flush(self):
        for f in self.files:
            f.flush()


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def sub(title):
    print("\n--- " + title + " " + "-" * max(0, 92 - len(title)))


def fmt(v):
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        if abs(v) >= 1e5:
            return f"{v:,.0f}"
        return f"{v:,.4f}"
    return str(v)


def show(d, title=None, width=58):
    if title:
        sub(title)
    for k, v in d.items():
        print(f"  {k:<{width}s}: {fmt(v)}")


def money(v):
    return f"${v:,.0f}"


# ----------------------------------------------------------------------------
# CORE DEFINITIONS
# ----------------------------------------------------------------------------
def f_factor_fn(gross, budget):
    """F-factor: (gross - budget) / budget."""
    gross, budget = np.asarray(gross, float), np.asarray(budget, float)
    return (gross - budget) / budget


class YJTargetRegressor(BaseEstimator, RegressorMixin):
    """Fits Yeo-Johnson+standardise on y INSIDE fit() (no leakage), trains the
    wrapped regressor on the transformed target, clips predictions to the
    training range (+margin) and inverts back to dollars."""

    def __init__(self, regressor=None, clip_margin=0.25):
        self.regressor = regressor
        self.clip_margin = clip_margin

    def fit(self, X, y):
        y = np.asarray(y, float).reshape(-1, 1)
        self.pt_ = PowerTransformer("yeo-johnson", standardize=True).fit(y)
        z = self.pt_.transform(y).ravel()
        self.lo_, self.hi_ = z.min() - self.clip_margin, z.max() + self.clip_margin
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        X = self.imputer_.fit_transform(X)
        self.regressor_ = clone(self.regressor).fit(X, z)
        return self

    def to_z(self, y):
        return self.pt_.transform(np.asarray(y, float).reshape(-1, 1)).ravel()

    def predict_z(self, X):
        return np.clip(self.regressor_.predict(self.imputer_.transform(X)), self.lo_, self.hi_)

    def predict(self, X):
        return self.pt_.inverse_transform(self.predict_z(X).reshape(-1, 1)).ravel()


class BudgetMultiple(BaseEstimator, RegressorMixin):
    """Naive baseline: gross = budget * median(train gross/budget)."""

    def fit(self, X, y):
        self.mult_ = float(np.median(np.asarray(y) / X["budget"].values))
        self.pt_ = PowerTransformer("yeo-johnson", standardize=True).fit(
            np.asarray(y, float).reshape(-1, 1))
        z = self.pt_.transform(np.asarray(y, float).reshape(-1, 1)).ravel()
        self.lo_, self.hi_ = z.min(), z.max()
        return self

    def to_z(self, y):
        return self.pt_.transform(np.asarray(y, float).reshape(-1, 1)).ravel()

    def predict(self, X):
        return X["budget"].values * self.mult_

    def predict_z(self, X):
        return self.to_z(self.predict(X))


# ----------------------------------------------------------------------------
# METRIC SUITES
# ----------------------------------------------------------------------------
def reg_metrics(y, p, k=None, positive=False):
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    r = p - y
    d = {}
    d["Sample size n"] = n
    d["MAE (mean absolute error)"] = mean_absolute_error(y, p)
    d["MSE (mean squared error)"] = mean_squared_error(y, p)
    d["RMSE (root mean squared error)"] = float(np.sqrt(d["MSE (mean squared error)"]))
    d["Median absolute error"] = median_absolute_error(y, p)
    d["Max absolute error"] = max_error(y, p)
    d["Mean error / bias (pred - true)"] = float(r.mean())
    d["R-squared"] = r2_score(y, p)
    if k is not None and n > k + 1:
        d["Adjusted R-squared"] = 1 - (1 - d["R-squared"]) * (n - 1) / (n - k - 1)
    d["Explained variance"] = explained_variance_score(y, p)
    d["Pearson r"] = stats.pearsonr(y, p)[0] if np.std(p) > 0 else np.nan
    d["Spearman rho"] = stats.spearmanr(y, p)[0]
    d["Kendall tau"] = stats.kendalltau(y, p)[0]
    d["Residual skewness"] = float(stats.skew(r))
    d["Residual excess kurtosis"] = float(stats.kurtosis(r))
    if positive:
        ape = np.abs(r) / np.abs(y)
        d["MAPE (%)"] = 100 * ape.mean()
        d["Median APE (%)"] = 100 * np.median(ape)
        d["sMAPE (%)"] = 100 * np.mean(2 * np.abs(r) / (np.abs(y) + np.abs(p)))
        d["RMSLE (root mean squared log error)"] = float(
            np.sqrt(np.mean((np.log1p(np.maximum(p, 0)) - np.log1p(y)) ** 2)))
        for t in (0.10, 0.25, 0.50):
            d[f"Within +/-{int(t*100)}% of true value (% of films)"] = 100 * np.mean(ape <= t)
        d["Within a factor of 2 (% of films)"] = 100 * np.mean(
            (p / y >= 0.5) & (p / y <= 2.0))
    return d


def cls_metrics(truth, pred, score=None):
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[False, True]).ravel()
    d = {}
    d["Actual positive rate (%)"] = 100 * truth.mean()
    d["Predicted positive rate (%)"] = 100 * pred.mean()
    d["Accuracy"] = float((tp + tn) / max(len(truth), 1))
    d["Balanced accuracy"] = balanced_accuracy_score(truth, pred) if truth.any() and (~truth).any() else np.nan
    d["Precision (PPV)"] = precision_score(truth, pred, zero_division=0)
    d["Recall (sensitivity / TPR)"] = recall_score(truth, pred, zero_division=0)
    d["Specificity (TNR)"] = tn / max(tn + fp, 1)
    d["F1"] = f1_score(truth, pred, zero_division=0)
    pr, rc = d["Precision (PPV)"], d["Recall (sensitivity / TPR)"]
    d["F2 (recall-weighted)"] = 5 * pr * rc / (4 * pr + rc) if (4 * pr + rc) > 0 else 0.0
    d["Matthews corr. coeff. (MCC)"] = matthews_corrcoef(truth, pred)
    d["Cohen kappa"] = cohen_kappa_score(truth, pred)
    if score is not None and truth.any() and (~truth).any():
        d["ROC-AUC (ranking by predicted F-factor)"] = roc_auc_score(truth, score)
        d["Average precision (PR-AUC)"] = average_precision_score(truth, score)
    else:
        d["ROC-AUC (ranking by predicted F-factor)"] = np.nan
        d["Average precision (PR-AUC)"] = np.nan
    d["True positives"], d["False positives"] = int(tp), int(fp)
    d["False negatives"], d["True negatives"] = int(fn), int(tn)
    return d


def band_metrics(f_factor_t, f_factor_p):
    bt = pd.cut(f_factor_t, F_FACTOR_BIN_EDGES, labels=False)
    bp = pd.cut(f_factor_p, F_FACTOR_BIN_EDGES, labels=False)
    bt, bp = np.asarray(bt, int), np.asarray(bp, int)
    return {
        "Exact F-factor-band accuracy": float(np.mean(bt == bp)),
        "Within-one-band accuracy": float(np.mean(np.abs(bt - bp) <= 1)),
        "Quadratic-weighted kappa (bands)": cohen_kappa_score(bt, bp, weights="quadratic"),
        "Mean band distance": float(np.mean(np.abs(bt - bp))),
    }


def score_all(y, gp, bud, k=None, win=(None, None), zt=None, zp=None):
    """Flat dict of every metric block for one evaluation set."""
    out = {}
    if zt is not None:
        for kk, v in reg_metrics(zt, zp, k=k).items():
            out[f"[YJ-scale target] {kk}"] = v
    for kk, v in reg_metrics(y, gp, k=k, positive=True).items():
        out[f"[Gross in $] {kk}"] = v
    for kk, v in reg_metrics(np.log(y), np.log(np.maximum(gp, 1.0)), k=k).items():
        out[f"[log(gross)] {kk}"] = v
    rt, rp = f_factor_fn(y, bud), f_factor_fn(gp, bud)
    for kk, v in reg_metrics(rt, rp).items():
        out[f"[F-factor raw] {kk}"] = v
    lo, hi = win
    if lo is not None:
        for kk, v in reg_metrics(np.clip(rt, lo, hi), np.clip(rp, lo, hi)).items():
            out[f"[F-factor winsorised] {kk}"] = v
    for kk, v in reg_metrics(np.log1p(rt), np.log1p(np.maximum(rp, -0.999999))).items():
        out[f"[log-multiple log(gross/budget)] {kk}"] = v
    for kk, v in cls_metrics(rt > HIT_F_FACTOR, rp > HIT_F_FACTOR, rp).items():
        out[f"[Hit: true F-factor>{HIT_F_FACTOR}] {kk}"] = v
    for kk, v in cls_metrics(rt > PROFIT_F_FACTOR, rp > PROFIT_F_FACTOR, rp).items():
        out[f"[Profit: true F-factor>0] {kk}"] = v
    for kk, v in band_metrics(rt, rp).items():
        out[f"[F-factor bands] {kk}"] = v
    return out


# ----------------------------------------------------------------------------
# REPORT BLOCKS
# ----------------------------------------------------------------------------
def print_grouped(flat, title):
    """Print a flat metric dict grouped by its [block] prefix."""
    section(title)
    blocks = {}
    for k, v in flat.items():
        b, m = k.split("] ", 1)
        blocks.setdefault(b + "]", {})[m] = v
    for b, d in blocks.items():
        show(d, f"Block {b}", width=52)


def residual_quantiles(y, gp, bud, label):
    sub(f"Residual 1/5/10/25/50/75/90/95/99 percentiles - {label}")
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    rows = {
        "gross error ($, pred - true)": np.percentile(gp - y, qs),
        "log error log(pred/true)": np.percentile(np.log(np.maximum(gp, 1) / y), qs),
        "F-factor error (pred - true)": np.percentile(f_factor_fn(gp, bud) - f_factor_fn(y, bud), qs),
    }
    df = pd.DataFrame(rows, index=[f"p{q}" for q in qs]).T
    print(df.to_string(float_format=lambda v: f"{v:,.3f}"))
    le = np.log(np.maximum(gp, 1) / y)
    print(f"\n  The central 90% of log-errors lies in [{np.percentile(le,5):.3f}, {np.percentile(le,95):.3f}]"
          f", i.e. predictions are within a factor of [{np.exp(np.percentile(le,5)):.2f}x, "
          f"{np.exp(np.percentile(le,95)):.2f}x] of the truth for 90% of films.")
    ae = np.abs(gp - y)
    print(f"  Absolute $ error: median {money(np.median(ae))}, 90th pct {money(np.percentile(ae,90))}, "
          f"95th pct {money(np.percentile(ae,95))}, 99th pct {money(np.percentile(ae,99))}.")


def outlier_flag_report(f_factor_t, f_factor_p, lo_q, hi_q, pred_lo_thr, pred_hi_thr, label):
    """How well does predicted F-factor detect true 5/95-quantile F-factor outliers?"""
    sub(f"Outlier (5/95 quantile) detection quality - {label}")
    print(f"  True-F-factor outlier thresholds come from TRAIN: q05={lo_q:.4f}, q95={hi_q:.4f}.")
    print(f"  Predicted-F-factor alarm thresholds come from OOF predictions: p05={pred_lo_thr:.4f}, p95={pred_hi_thr:.4f}"
          " (predictions are shrunk toward the mean, so raw train thresholds would never fire).")
    show(cls_metrics(f_factor_t > hi_q, f_factor_p > pred_hi_thr, f_factor_p), "HIGH-F-factor outliers (true F-factor > q95), alarm = pred F-factor > OOF p95")
    show(cls_metrics(f_factor_t < lo_q, f_factor_p < pred_lo_thr, -f_factor_p), "LOW-F-factor outliers (true F-factor < q05), alarm = pred F-factor < OOF p05")


def band_table(f_factor_t, f_factor_p, label):
    sub(f"F-factor band confusion matrix (rows=true, cols=predicted) - {label}")
    bt = pd.cut(f_factor_t, F_FACTOR_BIN_EDGES, labels=F_FACTOR_BIN_LABELS)
    bp = pd.cut(f_factor_p, F_FACTOR_BIN_EDGES, labels=F_FACTOR_BIN_LABELS)
    ct = pd.crosstab(bt, bp, rownames=["true"], colnames=["pred"], dropna=False)
    print(ct.to_string())
    print("\n  Row-normalised recall per true band:")
    print((ct.div(ct.sum(1).replace(0, np.nan), axis=0)).round(3).to_string())


def strategy_report(f_factor_t, f_factor_p, gross, bud, label):
    sub(f"Investment-strategy view (realised results) - {label}")

    def agg(mask, name):
        if mask.sum() == 0:
            print(f"  {name:<44s}: no films selected")
            return
        g, b, r = gross[mask], bud[mask], f_factor_t[mask]
        print(f"  {name:<44s}: n={mask.sum():5d} | pooled F-factor={(g.sum()-b.sum())/b.sum():8.3f} | "
              f"mean F-factor={r.mean():8.3f} | median F-factor={np.median(r):7.3f} | "
              f"hit-rate(F-factor>{HIT_F_FACTOR})={100*np.mean(r>HIT_F_FACTOR):5.1f}% | loss-rate={100*np.mean(r<0):5.1f}%")

    allm = np.ones(len(f_factor_t), bool)
    agg(allm, "Fund every film (baseline)")
    agg(f_factor_p > HIT_F_FACTOR, f"Fund films with predicted F-factor > {HIT_F_FACTOR}")
    agg(f_factor_p > PROFIT_F_FACTOR, "Fund films with predicted F-factor > 0")
    for q in (0.9, 0.8, 0.5):
        thr = np.quantile(f_factor_p, q)
        agg(f_factor_p >= thr, f"Top {int(round((1-q)*100))}% by predicted F-factor")
    agg(f_factor_p <= np.quantile(f_factor_p, 0.2), "Bottom 20% by predicted F-factor (should be worst)")
    agg(f_factor_t > HIT_F_FACTOR, f"Oracle: films with true F-factor > {HIT_F_FACTOR}")
    base_hit = np.mean(f_factor_t > HIT_F_FACTOR)
    sel = f_factor_p >= np.quantile(f_factor_p, 0.9)
    if sel.sum():
        print(f"\n  Lift of top-10% selection on hit-rate: {np.mean(f_factor_t[sel] > HIT_F_FACTOR) / base_hit:.2f}x "
              f"(base hit-rate {100*base_hit:.1f}%).")


def interval_report(y, lo, hi, bud, lo_q, hi_q, label, nominal=0.90):
    sub(f"Prediction interval quality (5%-95% quantile band) - {label}")
    cov = (y >= lo) & (y <= hi)
    d = {
        f"Nominal coverage": nominal,
        "Empirical coverage": cov.mean(),
        "Coverage gap (empirical - nominal)": cov.mean() - nominal,
        "Below-interval rate (truth < lower)": np.mean(y < lo),
        "Above-interval rate (truth > upper)": np.mean(y > hi),
        "Median interval width ($)": np.median(hi - lo),
        "Median upper/lower ratio": np.median(hi / lo),
        "Mean log-width": np.mean(np.log(hi / lo)),
    }
    for a, q in ((0.05, lo), (0.95, hi)):
        e = y - q
        d[f"Pinball loss, q={a} (in log$)"] = np.mean(np.maximum(a * (np.log(y) - np.log(q)),
                                                                 (a - 1) * (np.log(y) - np.log(q))))
    rt = f_factor_fn(y, bud)
    d["Interval score (Winkler, log$, lower=better)"] = np.mean(
        (np.log(hi) - np.log(lo))
        + (2 / (1 - nominal)) * np.maximum(np.log(lo) - np.log(y), 0)
        + (2 / (1 - nominal)) * np.maximum(np.log(y) - np.log(hi), 0))
    f_factor_lo, f_factor_hi = f_factor_fn(lo, bud), f_factor_fn(hi, bud)
    ch = f_factor_lo > HIT_F_FACTOR
    cl = f_factor_hi < PROFIT_F_FACTOR
    d[f"'Confident hits' (lower-bound F-factor>{HIT_F_FACTOR}) count"] = int(ch.sum())
    d["  ...of which truly F-factor>hit threshold (precision)"] = float(np.mean(rt[ch] > HIT_F_FACTOR)) if ch.any() else np.nan
    d["'Confident losses' (upper-bound F-factor<0) count"] = int(cl.sum())
    d["  ...of which truly F-factor<0 (precision)"] = float(np.mean(rt[cl] < 0)) if cl.any() else np.nan
    out_hi, out_lo = rt > hi_q, rt < lo_q
    d["Coverage on true HIGH-F-factor outliers (>q95)"] = float(cov[out_hi].mean()) if out_hi.any() else np.nan
    d["Coverage on true LOW-F-factor outliers (<q05)"] = float(cov[out_lo].mean()) if out_lo.any() else np.nan
    d["Coverage on non-outliers"] = float(cov[~(out_hi | out_lo)].mean())
    show(d, width=58)


def segment_report(df, label):
    """df has columns: y, gp, bud, seg (categorical)."""
    sub(f"Segment performance - {label}")
    rows = []
    for s, g in df.groupby("seg", observed=True):
        if len(g) < 5:
            continue
        le = np.log(np.maximum(g.gp, 1) / g.y)
        rt, rp = f_factor_fn(g.y, g.bud), f_factor_fn(g.gp, g.bud)
        rows.append({"segment": str(s), "n": len(g),
                     "R2(log gross)": r2_score(np.log(g.y), np.log(np.maximum(g.gp, 1))) if len(g) > 10 else np.nan,
                     "RMSE(log)": np.sqrt(np.mean(le ** 2)), "bias(log)": le.mean(),
                     "MdAPE(%)": 100 * np.median(np.abs(g.gp - g.y) / g.y),
                     "MAE($M)": np.mean(np.abs(g.gp - g.y)) / 1e6,
                     "true hit-rate(%)": 100 * np.mean(rt > HIT_F_FACTOR),
                     "pred hit-rate(%)": 100 * np.mean(rp > HIT_F_FACTOR),
                     "hit F1": f1_score(rt > HIT_F_FACTOR, rp > HIT_F_FACTOR, zero_division=0)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


def fold_summary(fold_df, title, cols=None):
    sub(title)
    cols = cols or list(fold_df.columns)
    n = len(fold_df)
    rows = []
    for c in cols:
        v = fold_df[c].astype(float)
        se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({"metric": c, "mean": v.mean(), "std": v.std(ddof=1), "min": v.min(),
                     "max": v.max(), "mean-1.96se": v.mean() - 1.96 * se, "mean+1.96se": v.mean() + 1.96 * se})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"  (n = {n} folds; the CI treats folds as independent, which repeated CV is not, so it is mildly optimistic.)")


# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
def hgb_base(seed, max_iter=300):
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=15,
                                         min_samples_leaf=20, l2_regularization=1.0,
                                         early_stopping=False, random_state=seed)


def build_models(seed, fast, n_jobs):
    ntree = 50 if fast else 400
    hgb_iter = 100 if fast else 300
    from scipy.stats import loguniform
    space = {"learning_rate": loguniform(0.02, 0.15),
             "max_iter": [50, 100] if fast else [150, 250, 350],
             "max_leaf_nodes": [7, 15, 31], "min_samples_leaf": [10, 20, 40],
             "l2_regularization": loguniform(1e-2, 10.0), "max_features": [0.4, 0.6, 0.8, 1.0]}
    tuned = RandomizedSearchCV(HistGradientBoostingRegressor(early_stopping=False, random_state=seed),
                               space, n_iter=2 if fast else 8, cv=3,
                               scoring="neg_mean_squared_error", random_state=seed, n_jobs=1)
    rf = RandomForestRegressor(n_estimators=ntree, min_samples_leaf=3, max_features=0.4,
                               n_jobs=n_jobs, random_state=seed)
    et = ExtraTreesRegressor(n_estimators=ntree, min_samples_leaf=2, max_features=0.6,
                             n_jobs=n_jobs, random_state=seed)
    gbr = GradientBoostingRegressor(n_estimators=100 if fast else 300, learning_rate=0.04, max_depth=3,
                                    subsample=0.8, min_samples_leaf=10, random_state=seed)
    ridge = make_pipeline(PowerTransformer(), RidgeCV(alphas=np.logspace(-2, 3, 25)))
    models = {
        "Baseline: budget x median multiple": BudgetMultiple(),
        "Baseline: predict train mean": YJTargetRegressor(DummyRegressor(strategy="mean")),
        "Ridge (power-transformed X)": YJTargetRegressor(ridge),
        "RandomForest": YJTargetRegressor(rf),
        "ExtraTrees": YJTargetRegressor(et),
        "HistGradientBoosting (fixed)": YJTargetRegressor(hgb_base(seed, hgb_iter)),
        "GradientBoosting (fixed)": YJTargetRegressor(gbr),
        "HistGB (nested-tuned)": YJTargetRegressor(tuned),
        "Voting ensemble (RF+ET+HGB+GBR)": YJTargetRegressor(VotingRegressor([
            ("rf", clone(rf)), ("et", clone(et)), ("hgb", hgb_base(seed, hgb_iter)), ("gbr", clone(gbr))])),
    }
    return models


def budget_missing_mask(frame):
    """Return the explicit budget-imputation flag, or all False if unavailable."""
    if "budget_missing" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    flag = pd.to_numeric(frame["budget_missing"], errors="raise")
    if flag.isna().any() or not flag.isin([0, 1]).all():
        raise ValueError("'budget_missing' must contain only non-null 0/1 values.")
    return flag.eq(1).to_numpy()


# ----------------------------------------------------------------------------
# CV ENGINE
# ----------------------------------------------------------------------------
def run_cv(name, est, X, y, bud, splits, win, k, folds=5):
    rows, oof = [], {}
    t0 = time.time()
    for i, (tr, va) in enumerate(splits):
        m = clone(est).fit(X.iloc[tr], y[tr])
        zp = m.predict_z(X.iloc[va])
        zt = m.to_z(y[va])
        gp = m.pt_.inverse_transform(zp.reshape(-1, 1)).ravel() if hasattr(m, "regressor_") else m.predict(X.iloc[va])
        rt_tr = f_factor_fn(y[tr], bud[tr])
        w = (np.quantile(rt_tr, 0.01), np.quantile(rt_tr, 0.99))
        r = score_all(y[va], gp, bud[va], k=k, win=w, zt=zt, zp=zp)
        r["repeat"], r["fold"] = i // folds, i % folds
        rows.append(r)
        oof.setdefault(i // folds, np.full(len(y), np.nan))[va] = gp
    print(f"  {name:<40s} done in {time.time()-t0:6.1f}s  ({len(splits)} folds)")
    return pd.DataFrame(rows), oof


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--train", default=os.path.join(script_dir, "train_encoded.csv"))
    ap.add_argument("--test", default=os.path.join(script_dir, "test_encoded.csv"))
    ap.add_argument("--out-dir", default=os.path.join(script_dir, "outputs"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true",
                    help="Use at most 3 folds, fewer estimators and reduced diagnostics for a quicker local run.")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="Parallel jobs for tree ensembles (default: all cores, or 1 with --fast).")
    a = ap.parse_args()
    if a.folds < 2:
        ap.error("--folds must be at least 2")
    if a.repeats < 1:
        ap.error("--repeats must be at least 1")
    if a.n_jobs == 0:
        ap.error("--n-jobs cannot be 0")
    if a.fast:
        a.folds = min(a.folds, 3)
        a.repeats = 1
    n_jobs = a.n_jobs if a.n_jobs is not None else (-1 if not a.fast else 1)
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, "pipeline_report.txt"), "w")
    sys.stdout = Tee(sys.__stdout__, log)
    if a.fast:
        a.repeats = 1
    seed = a.seed
    np.random.seed(seed)

    # ------------------------------------------------------------------ 1. LOAD + CLEAN
    section("1. DATA LOADING, CLEANING AND CATEGORICAL-COLUMN CHECK")
    tr_raw, te_raw = pd.read_csv(a.train), pd.read_csv(a.test)
    print(f"Train file: {a.train}  -> {tr_raw.shape[0]:,} rows x {tr_raw.shape[1]} columns")
    print(f"Test  file: {a.test}  -> {te_raw.shape[0]:,} rows x {te_raw.shape[1]} columns")
    missing_test = sorted(set(tr_raw.columns) - set(te_raw.columns))
    extra_test = sorted(set(te_raw.columns) - set(tr_raw.columns))
    if missing_test or extra_test:
        raise ValueError(f"Train/test columns differ; missing from test: {missing_test}; "
                         f"extra in test: {extra_test}. Re-encode both files with matching columns.")
    te_raw = te_raw[tr_raw.columns]
    print("Train and test have identical column names/order: True")
    required = [TARGET_RAW, "budget", "title_year", "decade"]
    missing_required = [c for c in required if c not in tr_raw.columns]
    if missing_required:
        raise ValueError(f"Encoded files are missing required pipeline columns: {missing_required}")
    for frame in (tr_raw, te_raw):
        bool_cols = frame.select_dtypes(include="bool").columns
        if len(bool_cols):
            frame[bool_cols] = frame[bool_cols].astype("int8")
    budget_missing_tr = budget_missing_mask(tr_raw)
    budget_missing_te = budget_missing_mask(te_raw)
    has_budget_flag = "budget_missing" in tr_raw.columns
    if has_budget_flag:
        print(f"'budget_missing'==1 in train: {int(budget_missing_tr.sum())} films "
              f"({100*budget_missing_tr.mean():.1f}%), test: {int(budget_missing_te.sum())} "
              f"({100*budget_missing_te.mean():.1f}%).")
        budget_status_label = "budget_missing==0"
    else:
        print("'budget_missing' is absent; budget-imputation status is unknown, so budget-subset metrics use all rows.")
        budget_status_label = "all rows (no budget_missing flag)"
    if (tr_raw[TARGET_RAW].isna().any() or te_raw[TARGET_RAW].isna().any()
            or (tr_raw[TARGET_RAW] <= 0).any() or (te_raw[TARGET_RAW] <= 0).any()):
        raise ValueError(f"'{TARGET_RAW}' must be positive and non-null in both encoded files.")
    if (tr_raw["budget"].isna().any() or te_raw["budget"].isna().any()
            or (tr_raw["budget"] <= 0).any() or (te_raw["budget"] <= 0).any()):
        raise ValueError("'budget' must be positive and non-null to calculate F-factor.")
    cat_cols = [c for c in tr_raw.columns
                if tr_raw[c].dtype == object or str(tr_raw[c].dtype) in ("category", "string")]
    cat_cols += [c for c in te_raw.columns if (te_raw[c].dtype == object) and c not in cat_cols]
    if cat_cols:
        print(f"Categorical/text columns found and DROPPED before anything else: {cat_cols}")
        tr_raw, te_raw = tr_raw.drop(columns=cat_cols), te_raw.drop(columns=cat_cols)
    else:
        print("Categorical/text (object, category, string, bool) columns found: 0 -> nothing to drop.")
        print("  The genre_*, rating_* and country_* columns are already numeric 0/1 dummies and are kept.")
    for nm, d in (("train", tr_raw), ("test", te_raw)):
        print(f"[{nm}] missing values: {int(d.isna().sum().sum())} | infinite values: "
              f"{int(np.isinf(d.select_dtypes('number')).sum().sum())} | duplicate rows: {int(d.duplicated().sum())}")
    ov = pd.merge(tr_raw.drop(columns=[TARGET_YJ], errors="ignore"),
                  te_raw.drop(columns=[TARGET_YJ], errors="ignore"), how="inner")
    print(f"Exact row overlap between train and test (feature columns): {len(ov)} rows")

    drop_feat = [TARGET_RAW, TARGET_YJ]
    feats = [c for c in tr_raw.columns if c not in drop_feat]
    Xtr, Xte = tr_raw[feats].copy(), te_raw[feats].copy()
    ytr, yte = tr_raw[TARGET_RAW].values.astype(float), te_raw[TARGET_RAW].values.astype(float)
    btr, bte = tr_raw["budget"].values.astype(float), te_raw["budget"].values.astype(float)
    print(f"\nTarget = '{TARGET_RAW}' (US$).  Dropped from features: {drop_feat}.")
    print(f"  '{TARGET_YJ}' is a deterministic Yeo-Johnson transform of gross (I verified max |diff| ~ 1e-15 when "
          "refitting on train) -> using it as a feature would be perfect target leakage.")
    print(f"  The Yeo-Johnson transform is therefore refitted INSIDE every CV fold instead of using the pre-fitted column.")
    print(f"Number of model features: {len(feats)}")
    if has_budget_flag:
        print("ASSUMPTION: flag=1 means budget was imputed, so F-factor for those films is unreliable; "
              "F-factor metrics are also reported on the non-imputed subset.")
    k = len(feats)

    # ------------------------------------------------------------------ 2. F-factor + OUTLIERS
    section("2. F-factor DEFINITION, DISTRIBUTION, 5/95 QUANTILE OUTLIERS AND F-factor>0.5 BANDS")
    print("F-factor = (gross - budget) / budget.   F-factor=0 is break-even, F-factor=0.5 means gross is 1.5x budget,")
    print("F-factor>0.5 is the 'hit' threshold used for classification metrics below.")
    rtr, rte = f_factor_fn(ytr, btr), f_factor_fn(yte, bte)
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    dist = pd.DataFrame({"train F-factor": np.percentile(rtr, qs), "test F-factor": np.percentile(rte, qs)},
                        index=[f"p{q}" for q in qs])
    sub("F-factor percentile table")
    print(dist.to_string(float_format=lambda v: f"{v:,.4f}"))
    for nm, r in (("train", rtr), ("test", rte)):
        print(f"[{nm}] mean F-factor {r.mean():.3f}, median {np.median(r):.3f}, std {r.std():.3f}, "
              f"skew {stats.skew(r):.1f}, excess kurtosis {stats.kurtosis(r):.1f}  (extremely heavy right tail)")
    lo_q, hi_q = np.quantile(rtr, LOW_Q), np.quantile(rtr, HIGH_Q)
    q1, q3 = np.percentile(rtr, [25, 75])
    tuk_lo, tuk_hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    sub("Outlier thresholds (all derived from TRAIN only)")
    print(f"  5/95 quantile F-factor band       : [{lo_q:.4f}, {hi_q:.4f}]")
    print(f"  Tukey 1.5xIQR fences on F-factor  : [{tuk_lo:.4f}, {tuk_hi:.4f}]")
    g_lo, g_hi = np.quantile(ytr, [LOW_Q, HIGH_Q])
    b_lo, b_hi = np.quantile(btr, [LOW_Q, HIGH_Q])
    print(f"  5/95 quantile gross band     : [{money(g_lo)}, {money(g_hi)}]")
    print(f"  5/95 quantile budget band    : [{money(b_lo)}, {money(b_hi)}]")

    def count_out(name, v, lo, hi, ref_n):
        below, above = int((v < lo).sum()), int((v > hi).sum())
        inside = len(v) - below - above
        print(f"  {name:<22s}: below={below:4d} ({100*below/len(v):5.2f}%)  inside={inside:4d} ({100*inside/len(v):5.2f}%)  "
              f"above={above:4d} ({100*above/len(v):5.2f}%)  -> OUT total={below+above:4d} ({100*(below+above)/len(v):5.2f}%)")

    for nm, y_, b_, r_ in (("TRAIN", ytr, btr, rtr), ("TEST", yte, bte, rte)):
        sub(f"How many {nm} films are OUT of the train 5/95 band? (expected ~5% per tail if no shift)")
        count_out("F-factor (5/95)", r_, lo_q, hi_q, len(r_))
        count_out("F-factor (Tukey fences)", r_, tuk_lo, tuk_hi, len(r_))
        count_out("Gross (5/95)", y_, g_lo, g_hi, len(y_))
        count_out("Budget (5/95)", b_, b_lo, b_hi, len(b_))
        print(f"  Outside train min/max range: F-factor={int(((r_<rtr.min())|(r_>rtr.max())).sum())}, "
              f"gross={int(((y_<ytr.min())|(y_>ytr.max())).sum())}, budget={int(((b_<btr.min())|(b_>btr.max())).sum())}")
    sub("F-factor > 0.5 range and F-factor bands")
    for nm, r_ in (("TRAIN", rtr), ("TEST", rte)):
        print(f"  [{nm}] F-factor > {HIT_F_FACTOR}: {int((r_>HIT_F_FACTOR).sum()):,} of {len(r_):,} films "
              f"({100*np.mean(r_>HIT_F_FACTOR):.2f}%) | F-factor > 0 (profitable): {int((r_>0).sum()):,} "
              f"({100*np.mean(r_>0):.2f}%) | F-factor <= 0.5: {int((r_<=HIT_F_FACTOR).sum()):,}")
    bt_tr = pd.cut(rtr, F_FACTOR_BIN_EDGES, labels=F_FACTOR_BIN_LABELS).value_counts().sort_index()
    bt_te = pd.cut(rte, F_FACTOR_BIN_EDGES, labels=F_FACTOR_BIN_LABELS).value_counts().sort_index()
    print(pd.DataFrame({"train n": bt_tr, "train %": 100 * bt_tr / len(rtr), "test n": bt_te,
                        "test %": 100 * bt_te / len(rte)}).to_string(float_format=lambda v: f"{v:,.2f}"))
    nm_mask = ~budget_missing_te
    print(f"\n  Test F-factor>0.5 on NON-imputed budgets only: {int((rte[nm_mask]>HIT_F_FACTOR).sum())} of {int(nm_mask.sum())} "
          f"({100*np.mean(rte[nm_mask]>HIT_F_FACTOR):.2f}%)")
    sub("Covariate shift check: share of test rows outside the train 5/95 band, per feature (top 10)")
    sh = {}
    for c in feats:
        if Xtr[c].nunique() > 2:
            lo, hi = Xtr[c].quantile([LOW_Q, HIGH_Q])
            sh[c] = 100 * ((Xte[c] < lo) | (Xte[c] > hi)).mean()
    print(pd.Series(sh).sort_values(ascending=False).head(10).round(2).to_string())
    print("  (about 10% is expected for identically distributed data; large values signal drift)")
    wlo, whi = np.quantile(rtr, 0.01), np.quantile(rtr, 0.99)
    print(f"\nF-factor winsorisation limits for robust F-factor metrics (train p1/p99): [{wlo:.3f}, {whi:.3f}]")

    # ------------------------------------------------------------------ 3. CV DESIGN
    section("3. CROSS-VALIDATION DESIGN (fixes the 'CV on the entire set' problem)")
    print("  * CV is run on the TRAIN file only. The test file is touched exactly once, at the very end.")
    print("  * Target transform, tuning, scaling and any thresholds are fitted inside the training part of each fold.")
    print(f"  * {a.repeats} x {a.folds}-fold Repeated STRATIFIED CV (stratified on 10 quantile bins of log gross), "
          f"seed={seed}.")
    print("  * Hyper-parameter tuning for 'HistGB (nested-tuned)' happens in an INNER 3-fold loop within each outer "
          "training fold (nested CV), so its score is not optimistically biased.")
    print("  * Model selection uses mean CV RMSE on the Yeo-Johnson scale; test data play no role.")
    strat = pd.qcut(np.log(ytr), 10, labels=False)
    rskf = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=seed)
    splits = list(rskf.split(Xtr, strat))
    print(f"  * Total outer folds: {len(splits)}; train/validation sizes ~ {len(splits[0][0])}/{len(splits[0][1])}.")

    # ------------------------------------------------------------------ 4. CV ALL MODELS
    section("4. MODEL COMPARISON (repeated stratified CV on train)")
    models = build_models(seed, a.fast, n_jobs)
    cv_res, cv_oof = {}, {}
    for name, est in models.items():
        cv_res[name], cv_oof[name] = run_cv(name, est, Xtr, ytr, btr, splits, (wlo, whi), k, a.folds)
    key = {"YJ RMSE": "[YJ-scale target] RMSE (root mean squared error)",
           "YJ R2": "[YJ-scale target] R-squared",
           "Gross MAE $M": "[Gross in $] MAE (mean absolute error)",
           "log RMSE": "[log(gross)] RMSE (root mean squared error)",
           "F-factor-win MAE": "[F-factor winsorised] MAE (mean absolute error)",
           "F-factor Spearman": "[F-factor raw] Spearman rho",
           f"Hit F1": f"[Hit: true F-factor>{HIT_F_FACTOR}] F1",
           f"Hit AUC": f"[Hit: true F-factor>{HIT_F_FACTOR}] ROC-AUC (ranking by predicted F-factor)",
           "Hit MCC": f"[Hit: true F-factor>{HIT_F_FACTOR}] Matthews corr. coeff. (MCC)",
           "Band acc": "[F-factor bands] Exact F-factor-band accuracy"}
    rows = []
    for name, df in cv_res.items():
        r = {"model": name}
        for lab, col in key.items():
            v = df[col].astype(float)
            sc = 1e-6 if lab == "Gross MAE $M" else 1
            r[lab] = f"{v.mean()*sc:.4f}+/-{v.std()*sc:.4f}"
        r["_sort"] = df[key["YJ RMSE"]].mean()
        rows.append(r)
    summ = pd.DataFrame(rows).sort_values("_sort")
    print("\nMean +/- std across folds (sorted by YJ RMSE, lower is better):")
    print(summ.drop(columns="_sort").to_string(index=False))
    cand = {n: d[key["YJ RMSE"]].mean() for n, d in cv_res.items() if not n.startswith("Baseline")}
    best = min(cand, key=cand.get)
    print(f"\n>>> Selected model by CV: '{best}' (mean YJ RMSE {cand[best]:.4f}).")
    base_rmse = cv_res["Baseline: predict train mean"][key["YJ RMSE"]].mean()
    print(f"    Relative RMSE reduction vs predict-the-mean baseline: {100*(1-cand[best]/base_rmse):.1f}%")

    # paired comparison best vs others
    sub("Paired fold-wise comparison of selected model vs each other model (YJ RMSE)")
    b_arr = cv_res[best][key["YJ RMSE"]].values
    for n, df in cv_res.items():
        if n == best:
            continue
        o = df[key["YJ RMSE"]].values
        t, p = stats.ttest_rel(o, b_arr)
        w = stats.wilcoxon(o, b_arr).pvalue if np.any(o != b_arr) else np.nan
        print(f"  {n:<40s}: mean diff (other - best) = {np.mean(o-b_arr):+.4f} | selected wins in "
              f"{100*np.mean(b_arr<o):.0f}% of folds | paired t p={p:.2g} | Wilcoxon p={w:.2g}")
    print("  (p-values are optimistic because repeated-CV folds share training data.)")
    cv_res[best].to_csv(os.path.join(a.out_dir, "cv_fold_metrics_best.csv"), index=False)

    # ------------------------------------------------------------------ 5. DETAILED CV (best)
    section(f"5. EXHAUSTIVE CV METRICS FOR SELECTED MODEL: {best}")
    metric_cols = [c for c in cv_res[best].columns if c not in ("repeat", "fold")]
    for b in dict.fromkeys(c.split("] ")[0] + "]" for c in metric_cols):
        cols = [c for c in metric_cols if c.startswith(b) and "Sample size" not in c]
        fold_summary(cv_res[best], f"CV fold distribution, block {b}", cols)

    sub("Repeat-to-repeat stability (mean YJ R2 and log RMSE per repeat)")
    print(cv_res[best].groupby("repeat")[[key["YJ R2"], key["log RMSE"]]].mean().to_string(float_format=lambda v: f"{v:.4f}"))

    oof_best = np.nanmean(np.vstack([cv_oof[best][r] for r in sorted(cv_oof[best])]), axis=0)
    pt_full = PowerTransformer("yeo-johnson", standardize=True).fit(ytr.reshape(-1, 1))
    zt_oof = pt_full.transform(ytr.reshape(-1, 1)).ravel()
    zp_oof = pt_full.transform(oof_best.reshape(-1, 1)).ravel()
    flat_oof = score_all(ytr, oof_best, btr, k=k, win=(wlo, whi), zt=zt_oof, zp=zp_oof)
    print_grouped(flat_oof, f"5b. POOLED OUT-OF-FOLD METRICS (all {len(ytr):,} train films, predictions averaged over repeats)")
    oof_f_factor = f_factor_fn(oof_best, btr)
    residual_quantiles(ytr, oof_best, btr, "pooled OOF")
    pred_lo_thr, pred_hi_thr = np.quantile(oof_f_factor, LOW_Q), np.quantile(oof_f_factor, HIGH_Q)
    outlier_flag_report(rtr, oof_f_factor, lo_q, hi_q, pred_lo_thr, pred_hi_thr, "pooled OOF")
    band_table(rtr, oof_f_factor, "pooled OOF")
    strategy_report(rtr, oof_f_factor, ytr, btr, "pooled OOF")
    sub(f"F-factor metrics restricted to {budget_status_label}, pooled OOF")
    m0 = ~budget_missing_tr
    sc = score_all(ytr[m0], oof_best[m0], btr[m0], k=k, win=(wlo, whi))
    for kk in [x for x in sc if x.startswith(("[F-factor winsorised]", "[Hit", "[log-multiple"))]:
        if any(s in kk for s in ("RMSE (", "MAE", "R-squared", "Spearman", "F1", "ROC-AUC", "Matthews", "Precision", "Recall")):
            print(f"  {kk:<92s}: {fmt(sc[kk])}")

    seg = pd.DataFrame({"y": ytr, "gp": oof_best, "bud": btr})
    seg["seg"] = pd.qcut(btr, 5, labels=["budget Q1 (lowest)", "budget Q2", "budget Q3", "budget Q4", "budget Q5 (highest)"])
    segment_report(seg, "pooled OOF by budget quintile")
    seg["seg"] = pd.cut(tr_raw.decade.values, [0, 1980, 1990, 2000, 2010, 3000],
                        labels=["<=1980s", "1990s", "2000s", "2010s", ">2010s"], right=False)
    seg["seg"] = tr_raw.decade.astype(int).astype(str) + "s"
    segment_report(seg, "pooled OOF by decade")
    seg["seg"] = np.where(m0, "budget not flagged imputed", "budget flagged imputed")
    segment_report(seg, f"pooled OOF by budget imputation status ({budget_status_label})")
    seg["seg"] = np.where(rtr < lo_q, "F-factor < q05 (low outlier)", np.where(rtr > hi_q, "F-factor > q95 (high outlier)", "F-factor inside 5/95"))
    segment_report(seg, "pooled OOF by true-F-factor outlier status")
    seg["seg"] = pd.cut(rtr, F_FACTOR_BIN_EDGES, labels=F_FACTOR_BIN_LABELS)
    segment_report(seg, "pooled OOF by true F-factor band")

    # ------------------------------------------------------------------ 6. LEAKAGE / ROBUSTNESS
    section("6. LEAKAGE AUDIT, FEATURE-GROUP ABLATION AND TEMPORAL ROBUSTNESS")
    print("The *_score / *_is_star / cast_star_* columns look target-encoded (centred near 0). If they were computed from")
    print("gross on the full train set BEFORE CV, every CV fold sees leaked target information. Ablation below shows")
    print("how much performance depends on them. Post-release columns (votes, reviews, imdb_score, fb likes) are not")
    print("known at greenlight time, so the 'pre-release only' row estimates a real forecasting scenario.")
    base_est = YJTargetRegressor(hgb_base(seed, 100 if a.fast else 300))
    ab_splits = list(RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=1, random_state=seed + 1).split(Xtr, strat))
    groups = {"all features": [],
              "drop score-encoded group": SCORE_ENCODED,
              "drop post-release group": POST_RELEASE,
              "drop BOTH (pre-release, no target-encodings)": SCORE_ENCODED + POST_RELEASE}
    arows = []
    for gname, drop in groups.items():
        cols = [c for c in feats if c not in drop]
        res, _ = run_cv(f"ablation: {gname}", base_est, Xtr[cols], ytr, btr, ab_splits,
                        (wlo, whi), len(cols), a.folds)
        arows.append({"feature set": gname, "n feats": len(cols),
                      "YJ R2": res[key["YJ R2"]].mean(), "YJ RMSE": res[key["YJ RMSE"]].mean(),
                      "log RMSE": res[key["log RMSE"]].mean(), "Hit F1": res[key["Hit F1"]].mean(),
                      "Hit AUC": res[key["Hit AUC"]].mean(), "F-factor Spearman": res[key["F-factor Spearman"]].mean()})
    ab = pd.DataFrame(arows)
    print(ab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    drop_r2 = ab.loc[0, "YJ R2"] - ab.loc[1, "YJ R2"]
    print(f"\n  Removing the score-encoded group changes CV R2 by {-drop_r2:+.4f}. "
          + ("A big drop would suggest leakage/over-reliance; small drop means they add little." ))
    print("  Caveat: I cannot prove leakage from the encoded CSV alone - re-derive those scores inside each CV fold "
          "from the raw data to be certain.")

    sub("Temporal robustness: train on older films, validate on newer (expanding window by title_year)")
    yrs = Xtr["title_year"].values
    order = np.flatnonzero(np.isfinite(yrs))
    order = order[np.argsort(yrs[order], kind="stable")]
    print(f"  Excluded {len(yrs) - len(order)} films with unknown title_year from this temporal-only diagnostic.")
    cuts = np.linspace(0.5, 1.0, 6)
    for i in range(5):
        tr_i, va_i = order[: int(cuts[i] * len(order))], order[int(cuts[i] * len(order)): int(cuts[i + 1] * len(order))]
        m = clone(base_est).fit(Xtr.iloc[tr_i], ytr[tr_i])
        gp = m.predict(Xtr.iloc[va_i])
        print(f"  window {i+1}: train years {int(yrs[tr_i].min())}-{int(yrs[tr_i].max())} (n={len(tr_i)}), "
              f"validate years {int(yrs[va_i].min())}-{int(yrs[va_i].max())} (n={len(va_i)}): "
              f"R2(log gross)={r2_score(np.log(ytr[va_i]), np.log(np.maximum(gp,1))):.3f}, "
              f"RMSE(log)={np.sqrt(np.mean((np.log(np.maximum(gp,1))-np.log(ytr[va_i]))**2)):.3f}, "
              f"F-factor Spearman={stats.spearmanr(f_factor_fn(ytr[va_i], btr[va_i]), f_factor_fn(gp, btr[va_i]))[0]:.3f}")

    # ------------------------------------------------------------------ 7. QUANTILE INTERVALS
    section("7. 5% / 95% QUANTILE PREDICTION INTERVALS (quantile GBM + conformal calibration)")
    ylog = np.log(ytr)

    def qgbr(alpha):
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            GradientBoostingRegressor(loss="quantile", alpha=alpha,
                                      n_estimators=80 if a.fast else 250, learning_rate=0.04,
                                      max_depth=3, subsample=0.8, min_samples_leaf=15, random_state=seed))
    lo_oof, hi_oof = np.zeros(len(ytr)), np.zeros(len(ytr))
    for tr_i, va_i in StratifiedKFold(a.folds, shuffle=True, random_state=seed).split(Xtr, strat):
        lo_oof[va_i] = qgbr(LOW_Q).fit(Xtr.iloc[tr_i], ylog[tr_i]).predict(Xtr.iloc[va_i])
        hi_oof[va_i] = qgbr(HIGH_Q).fit(Xtr.iloc[tr_i], ylog[tr_i]).predict(Xtr.iloc[va_i])
    lo_oof, hi_oof = np.minimum(lo_oof, hi_oof), np.maximum(lo_oof, hi_oof)
    interval_report(ytr, np.exp(lo_oof), np.exp(hi_oof), btr, lo_q, hi_q, "raw quantile GBM, OOF on train")
    conf = np.maximum(lo_oof - ylog, ylog - hi_oof)
    nq = len(conf)
    qhat = np.quantile(conf, min(1.0, np.ceil((nq + 1) * 0.90) / nq))
    print(f"\n  Conformal (CQR) correction from OOF scores: widen log-interval by {qhat:+.4f} on each side "
          f"(factor {np.exp(qhat):.3f}x).")
    interval_report(ytr, np.exp(lo_oof - qhat), np.exp(hi_oof + qhat), btr, lo_q, hi_q,
                    "conformalised, OOF on train (calibration data itself, slightly optimistic)")

    # ------------------------------------------------------------------ 8. FINAL TEST
    section(f"8. FINAL HELD-OUT TEST EVALUATION ({len(yte):,} films) - model: {best}")
    t0 = time.time()
    final = clone(models[best]).fit(Xtr, ytr)
    print(f"Final model refitted on ALL train rows in {time.time()-t0:.1f}s.")
    inner = getattr(final, "regressor_", None)
    if isinstance(inner, RandomizedSearchCV):
        print(f"Tuned hyper-parameters (inner CV on full train): {inner.best_params_}")
    zp_te = final.predict_z(Xte)
    gp_te = final.predict(Xte)
    zt_te = final.to_z(yte)
    flat_te = score_all(yte, gp_te, bte, k=k, win=(wlo, whi), zt=zt_te, zp=zp_te)
    print_grouped(flat_te, "8a. TEST METRICS (all blocks)")

    cv_mean = cv_res[best]
    sub("Test vs CV consistency (a large gap would indicate CV leakage or distribution shift)")
    for lab in ("YJ R2", "YJ RMSE", "log RMSE", "Hit F1", "Hit AUC", "F-factor Spearman"):
        col = key[lab]
        tv = flat_te[col]
        print(f"  {lab:<14s}: CV {cv_mean[col].mean():.4f} +/- {cv_mean[col].std():.4f}   |   TEST {tv:.4f}   "
              f"| z-score of test within CV folds: {(tv-cv_mean[col].mean())/cv_mean[col].std():+.2f}")

    residual_quantiles(yte, gp_te, bte, "TEST")
    f_factor_te_pred = f_factor_fn(gp_te, bte)
    outlier_flag_report(rte, f_factor_te_pred, lo_q, hi_q, pred_lo_thr, pred_hi_thr, "TEST")
    band_table(rte, f_factor_te_pred, "TEST")
    strategy_report(rte, f_factor_te_pred, yte, bte, "TEST")

    bootstrap_samples = 250 if a.fast else 2000
    sub(f"Bootstrap 95% confidence intervals on TEST metrics ({bootstrap_samples} resamples)")
    rng = np.random.default_rng(seed)
    bs = {"R2 (log gross)": [], "RMSE (log gross)": [], "MAE gross ($M)": [], "F-factor Spearman": [],
          "Hit F1": [], "Hit AUC": [], "Hit MCC": []}
    for _ in range(bootstrap_samples):
        ix = rng.integers(0, len(yte), len(yte))
        yy, pp, bb = yte[ix], gp_te[ix], bte[ix]
        ly, lp = np.log(yy), np.log(np.maximum(pp, 1))
        bs["R2 (log gross)"].append(r2_score(ly, lp))
        bs["RMSE (log gross)"].append(np.sqrt(np.mean((ly - lp) ** 2)))
        bs["MAE gross ($M)"].append(np.mean(np.abs(yy - pp)) / 1e6)
        rt_, rp_ = f_factor_fn(yy, bb), f_factor_fn(pp, bb)
        bs["F-factor Spearman"].append(stats.spearmanr(rt_, rp_)[0])
        tt = rt_ > HIT_F_FACTOR
        bs["Hit F1"].append(f1_score(tt, rp_ > HIT_F_FACTOR, zero_division=0))
        bs["Hit AUC"].append(roc_auc_score(tt, rp_) if tt.any() and (~tt).any() else np.nan)
        bs["Hit MCC"].append(matthews_corrcoef(tt, rp_ > HIT_F_FACTOR))
    for kk, v in bs.items():
        v = np.array(v, float)
        print(f"  {kk:<20s}: median {np.nanmedian(v):.4f}   95% CI [{np.nanpercentile(v,2.5):.4f}, {np.nanpercentile(v,97.5):.4f}]")

    sub(f"F-factor metrics on TEST restricted to {budget_status_label}")
    sc = score_all(yte[nm_mask], gp_te[nm_mask], bte[nm_mask], k=k, win=(wlo, whi))
    for kk in sc:
        if kk.startswith(("[F-factor winsorised]", "[Hit", "[log-multiple")) and any(
                s in kk for s in ("RMSE (", "MAE", "R-squared", "Spearman", "F1", "ROC-AUC", "Matthews", "Precision", "Recall")):
            print(f"  {kk:<92s}: {fmt(sc[kk])}")

    # test intervals
    lo_te = qgbr(LOW_Q).fit(Xtr, ylog).predict(Xte)
    hi_te = qgbr(HIGH_Q).fit(Xtr, ylog).predict(Xte)
    lo_te, hi_te = np.minimum(lo_te, hi_te), np.maximum(lo_te, hi_te)
    interval_report(yte, np.exp(lo_te), np.exp(hi_te), bte, lo_q, hi_q, "TEST, raw quantile GBM")
    interval_report(yte, np.exp(lo_te - qhat), np.exp(hi_te + qhat), bte, lo_q, hi_q, "TEST, conformalised")

    seg = pd.DataFrame({"y": yte, "gp": gp_te, "bud": bte})
    seg["seg"] = pd.cut(bte, np.concatenate([[-np.inf], np.quantile(btr, [0.2, 0.4, 0.6, 0.8]), [np.inf]]),
                        labels=["budget Q1 (lowest)", "budget Q2", "budget Q3", "budget Q4", "budget Q5 (highest)"])
    segment_report(seg, "TEST by train-budget quintile")
    seg["seg"] = te_raw.decade.astype(int).astype(str) + "s"
    segment_report(seg, "TEST by decade")
    seg["seg"] = np.where(nm_mask, "budget not flagged imputed", "budget flagged imputed")
    segment_report(seg, f"TEST by budget imputation status ({budget_status_label})")
    seg["seg"] = np.where(rte < lo_q, "F-factor < q05 (low outlier)", np.where(rte > hi_q, "F-factor > q95 (high outlier)", "F-factor inside 5/95"))
    segment_report(seg, "TEST by true-F-factor outlier status")
    seg["seg"] = pd.cut(rte, F_FACTOR_BIN_EDGES, labels=F_FACTOR_BIN_LABELS)
    segment_report(seg, "TEST by true F-factor band")

    sub("20 largest test errors (by |log error|)")
    le = np.log(np.maximum(gp_te, 1) / yte)
    worst = np.argsort(-np.abs(le))[:20]
    print(pd.DataFrame({"row": worst, "year": Xte["title_year"].values[worst].astype(int),
                        "budget": bte[worst], "true gross": yte[worst], "pred gross": gp_te[worst],
                        "true F-factor": rte[worst], "pred F-factor": f_factor_te_pred[worst],
                        "budget_missing": budget_missing_te[worst].astype(int),
                        "log err": le[worst]}).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    importance_repeats = 1 if a.fast else 5
    sub(f"Permutation importance on TEST (drop in R2 of log gross; {importance_repeats} repeats; top 20)")
    from sklearn.metrics import make_scorer
    scorer = make_scorer(lambda yt, yp: r2_score(np.log(yt), np.log(np.maximum(yp, 1))))
    pi = permutation_importance(final, Xte, yte, scoring=scorer, n_repeats=importance_repeats,
                                random_state=seed, n_jobs=1)
    imp = pd.DataFrame({"feature": feats, "mean drop": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("mean drop", ascending=False).head(20)
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = pd.DataFrame({"title_year": Xte["title_year"].values, "budget": bte,
                        "budget_missing": budget_missing_te.astype(int),
                        "true_gross": yte, "pred_gross": gp_te, "pred_gross_q05_conformal": np.exp(lo_te - qhat),
                        "pred_gross_q95_conformal": np.exp(hi_te + qhat), "true_f_factor": rte, "pred_f_factor": f_factor_te_pred,
                        "true_f_factor_out_5_95": np.where(rte < lo_q, "low", np.where(rte > hi_q, "high", "in")),
                        "true_f_factor_gt_0.5": rte > HIT_F_FACTOR, "pred_f_factor_gt_0.5": f_factor_te_pred > HIT_F_FACTOR})
    out.to_csv(os.path.join(a.out_dir, "test_predictions.csv"), index=False)
    pd.DataFrame({"oof_pred_gross": oof_best, "true_gross": ytr, "budget": btr}).to_csv(
        os.path.join(a.out_dir, "train_oof_predictions.csv"), index=False)

    section("9. ONE-SCREEN SUMMARY")
    print(f"  Selected model            : {best}")
    print(f"  CV  YJ R2 / log RMSE      : {cv_mean[key['YJ R2']].mean():.4f} / {cv_mean[key['log RMSE']].mean():.4f}")
    print(f"  TEST YJ R2 / log RMSE     : {flat_te[key['YJ R2']]:.4f} / {flat_te[key['log RMSE']]:.4f}")
    print(f"  TEST films out of 5/95 F-factor band: {int(((rte<lo_q)|(rte>hi_q)).sum())} of {len(rte)} "
          f"(low {int((rte<lo_q).sum())}, high {int((rte>hi_q).sum())})")
    print(f"  TEST films with F-factor > {HIT_F_FACTOR}  : {int((rte>HIT_F_FACTOR).sum())} of {len(rte)}")
    print(f"  TEST hit F1 / AUC / MCC   : {flat_te[key['Hit F1']]:.4f} / {flat_te[key['Hit AUC']]:.4f} / "
          f"{flat_te[key['Hit MCC']]:.4f}")
    print(f"  Files written to {a.out_dir}: pipeline_report.txt, test_predictions.csv, "
          "train_oof_predictions.csv, cv_fold_metrics_best.csv")
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()