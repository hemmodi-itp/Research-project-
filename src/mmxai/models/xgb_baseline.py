"""Gradient-boosted trees on structured data only: the additional structured baseline of Sec. 3.2."""

import numpy as np


def xgb_matrix(pre, df):
    # Trees handle missing values natively, so NaNs are kept rather than imputed.
    cat = pre.category_codes(df).astype(float)
    cat[cat == 0] = np.nan
    return np.hstack([pre.winsorized(df).to_numpy(dtype=float), cat])


def train_xgboost(pre, train_df, val_df, cfg, seed):
    from xgboost import XGBClassifier

    xcfg = cfg["xgboost"]
    y_train = train_df["label"].to_numpy()
    model = XGBClassifier(
        n_estimators=xcfg["n_estimators"],
        learning_rate=xcfg["learning_rate"],
        max_depth=xcfg["max_depth"],
        subsample=xcfg["subsample"],
        colsample_bytree=xcfg["colsample_bytree"],
        scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()),  # same balancing as Eq. 6
        eval_metric="auc",
        early_stopping_rounds=xcfg["early_stopping_rounds"],
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(xgb_matrix(pre, train_df), y_train,
              eval_set=[(xgb_matrix(pre, val_df), val_df["label"].to_numpy())], verbose=False)
    return model
