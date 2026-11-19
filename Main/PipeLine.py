import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, mean_absolute_error,
                             median_absolute_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.inspection import permutation_importance

#Intial Configuration
FOLDS, SEED = 15, 42
HIT = 2.65
POST_RELEASE = ["num_voted_users", "num_user_for_reviews", "num_critic_for_reviews",
                "movie_facebook_likes", "has_movie_fb", "imdb_score"]   # only known after release (also catches log_* copies)

#Reading the data and coverting to numpy array
tr, te = pd.read_csv("train_encoded.csv"), pd.read_csv("test_encoded.csv")
all_feats = [c for c in tr.columns if c not in ["gross", "gross_yj"]]
ytr, yte = tr["gross"].to_numpy(float), te["gross"].to_numpy(float)
btr, bte = tr["budget"].to_numpy(float), te["budget"].to_numpy(float)

lo, hi = np.percentile(ytr / btr, [5, 95]) #F-factor = gross / budget, outlier band from TRAIN only
yj = PowerTransformer().fit(ytr.reshape(-1, 1)) #only used to report metrics on the Yeo-Johnson scale
strat = pd.qcut(np.log(ytr), 10, labels=False)

model = TransformedTargetRegressor(
    regressor=HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                            min_samples_leaf=20, l2_regularization=1.0,
                                            early_stopping=False, random_state=SEED),
    transformer=PowerTransformer(method="yeo-johnson", standardize=True))

#Defining functions for printing statistics for pre and post without doing it seperately
summary = {}
for label, banned in [("PRE-RELEASE", POST_RELEASE), ("POST-RELEASE", [])]:
    feats = [c for c in all_feats if not any(s in c for s in banned)]
    Xtr, Xte = tr[feats], te[feats]
    print("\n" + label + " model: " + str(len(feats)) + " features")

    # k-fold CV on train: out-of-fold predictions + train vs validation R2 per fold
    oof, r2_train, r2_val = np.zeros(len(ytr)), [], []
    for tr_i, va_i in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(Xtr, strat):
        m = clone(model).fit(Xtr.iloc[tr_i], ytr[tr_i])
        oof[va_i] = m.predict(Xtr.iloc[va_i])
        r2_train.append(r2_score(ytr[tr_i], m.predict(Xtr.iloc[tr_i])))
        r2_val.append(r2_score(ytr[va_i], oof[va_i]))
    print("CV R2 (gross $), mean over " + str(FOLDS) + " folds: train " +
          "{:.3f}".format(np.mean(r2_train)) + " | validation " +
          "{:.3f}".format(np.mean(r2_val)) + " +/- " + "{:.3f}".format(np.std(r2_val)))

    final_model = clone(model).fit(Xtr, ytr)
    pte = final_model.predict(Xte)

    result = permutation_importance(
        final_model, Xte, yte,
        n_repeats=10,
        random_state=SEED,
        scoring="r2"
    )

    top_idx = np.argsort(result.importances_mean)[::-1][:30]
    top_features = Xtr.columns.to_numpy()[top_idx]
    top_weights = result.importances_mean[top_idx]

    print("\n" + label + " top 30 features:")
    print(pd.DataFrame({
        "feature": top_features,
        "importance": top_weights
    }).to_string(index=False))

#Defining maual classifier
    for name, X, y, b, p in [("CV", Xtr, ytr, btr, oof), ("TEST", Xte, yte, bte, pte)]:
        p = np.maximum(p, 1.0)  #dollars; avoids log of <= 0
        r, rp = y / b, p / b    #true/predicted F-factor
        err, ferr = p - y, rp - r
        out = (r < lo) | (r > hi) #true F-factor outside the train 5/95 band
        z, zp = yj.transform(y.reshape(-1, 1)).ravel(), yj.transform(p.reshape(-1, 1)).ravel()

        th, ph = r >= HIT, rp >= HIT

        summary[(label, name)] = {
            "R2 gross ($)": r2_score(y, p),
            "R2 log gross": r2_score(np.log(y), np.log(p)),
            "R2 Yeo-Johnson gross": r2_score(z, zp),
            "log RMSE": np.sqrt(np.mean((np.log(p) - np.log(y)) ** 2)),
            "MedAE gross ($M)": median_absolute_error(y, p) / 1e6,
            "MAE gross ($M)": mean_absolute_error(y, p) / 1e6,
            "gross error p5 ($M)": np.percentile(err, 5) / 1e6,
            "gross error p95 ($M)": np.percentile(err, 95) / 1e6,
            "MedAE F-factor": np.median(np.abs(ferr)),
            "F-factor error p5": np.percentile(ferr, 5),
            "F-factor error p95": np.percentile(ferr, 95),
            "films outside F-factor band": int(out.sum()),
            "MedAE gross inside band ($M)": np.median(np.abs(err[~out])) / 1e6,
            "MedAE gross outside band ($M)": np.median(np.abs(err[out])) / 1e6,
            "Hit F1": f1_score(th, ph, zero_division=0),
            "Hit recall": recall_score(th, ph, zero_division=0),
            "Hit precision": precision_score(th, ph, zero_division=0),
            "Hit accuracy": accuracy_score(th, ph),
            "Hit MCC": matthews_corrcoef(th, ph),
        }
        #for TEST part however i'll also have to plot the curves
        if name == "TEST":
            lim = [min(y.min(), p.min()) / 1e6, max(y.max(), p.max()) / 1e6]
            plt.figure(figsize=(7, 6))
            plt.scatter(y / 1e6, p / 1e6, color="#945121", alpha=0.5, edgecolors="k")
            plt.plot(lim, lim, "k--")                           # perfect-prediction line
            plt.xscale("log")
            plt.yscale("log")
            plt.title(label + " HistGB: actual vs predicted gross (TEST)")
            plt.xlabel("Actual gross (Millions USD, log scale)")
            plt.ylabel("Predicted gross (Millions USD, log scale)")
            plt.tight_layout()
            plt.savefig(label.lower() + "_test_scatter.png", dpi=150)
            plt.close()

print("\nSUMMARY")
print(pd.DataFrame(summary).to_string(float_format=lambda v: "{:,.3f}".format(v)))
