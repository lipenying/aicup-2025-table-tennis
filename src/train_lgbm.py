"""
最終版本：LightGBM + GroupKFold（Public Leaderboard 0.770139）。

把每個 unique_id 的 27×34 揮拍特徵彙整成 612 維檔案層級特徵，
四個任務各自用 LGBMClassifier 訓練，以 player_id 分組做 5-fold 交叉驗證。

註：比賽時最終版本只保留了程式截圖。extract_features()、測試資料處理、
fit_predict_classifier() 前半段、機率總和檢查與 AUC 輸出照截圖還原；
fit_predict_classifier() 後半段、round_and_normalize_probs() 與輸出上傳檔
依報告文字重建。

用法：
    python src/train_lgbm.py --data-root data
"""
from collections import Counter
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize


def extract_features(df):
    feats = {}
    for col in df.columns:
        if col in ['irrelevant', 'index', 'time']:
            continue
        arr = df[col].values
        feats[f'{col}_mean'] = arr.mean()
        feats[f'{col}_std'] = arr.std()
        feats[f'{col}_max'] = arr.max()
        feats[f'{col}_min'] = arr.min()
        feats[f'{col}_median'] = np.median(arr)
        feats[f'{col}_skew'] = pd.Series(arr).skew()
        feats[f'{col}_kurt'] = pd.Series(arr).kurt()
        feats[f'{col}_q25'] = np.percentile(arr, 25)
        feats[f'{col}_q75'] = np.percentile(arr, 75)
        feats[f'{col}_diffmaxmin'] = arr.max() - arr.min()
        # 新增
        feats[f'{col}_mean_first5'] = arr[:5].mean()
        feats[f'{col}_mean_last5'] = arr[-5:].mean()
        feats[f'{col}_diff_first_last'] = arr[0] - arr[-1]
        feats[f'{col}_max_pos'] = np.argmax(arr)
        feats[f'{col}_min_pos'] = np.argmin(arr)
        feats[f'{col}_sum'] = arr.sum()
        # 差分特徵
        diff = np.diff(arr)
        feats[f'{col}_diff_mean'] = diff.mean()
        feats[f'{col}_diff_std'] = diff.std()
    return feats


def load_file_features(datapath):
    """讀取特徵 CSV，只保留 27 列（比賽規定）的檔案，每個檔案彙整成一筆樣本。"""
    datalist = sorted(Path(datapath).glob('**/*.csv'), key=lambda x: int(x.stem))
    feats_list, unique_ids = [], []
    for file in datalist:
        unique_id = int(file.stem)
        data = pd.read_csv(file)
        if len(data) != 27:
            print(f"Skip {file}, rows: {len(data)}")
            continue
        feats_list.append(extract_features(data))
        unique_ids.append(unique_id)
    return pd.DataFrame(feats_list), unique_ids


def fit_predict_classifier(X, y, groups, task_type, X_pred, n_classes=None):
    if task_type == "binary":
        model = LGBMClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    else:
        model = LGBMClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    aucs = []
    oof_pred = np.zeros((X.shape[0], n_classes)) if n_classes else np.zeros(X.shape[0])
    for fold, (train_idx, valid_idx) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        # 檢查這個 fold 的標籤分布
        y_val = y[valid_idx]
        print(f"Fold {fold} valid set label distribution: {Counter(y_val)}")
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr = y[train_idx]
        model.fit(X_tr, y_tr)

        if task_type == "binary":
            oof_pred[valid_idx] = model.predict_proba(X_val)[:, 1]
        else:
            # 訓練 fold 可能缺少某些類別，依 model.classes_ 對齊到完整欄位
            prob = np.zeros((len(valid_idx), n_classes))
            prob[:, model.classes_] = model.predict_proba(X_val)
            oof_pred[valid_idx] = prob

        # fold 內只有單一類別時無法計算 AUC
        if len(np.unique(y_val)) < 2:
            print(f"Fold {fold}: only one class in validation set, skip AUC.")
            continue
        if task_type == "binary":
            aucs.append(roc_auc_score(y_val, oof_pred[valid_idx]))
        else:
            y_bin = label_binarize(y_val, classes=list(range(n_classes)))
            aucs.append(roc_auc_score(y_bin, oof_pred[valid_idx], average='micro'))

    if task_type == "binary":
        oof_auc = roc_auc_score(y, oof_pred)
    else:
        oof_auc = roc_auc_score(label_binarize(y, classes=list(range(n_classes))), oof_pred, average='micro')
    print(f"Fold AUCs: {np.round(aucs, 3)}, OOF AUC: {oof_auc:.3f}")

    # 用全部訓練資料重新訓練後預測測試集
    model.fit(X, y)
    if task_type == "binary":
        pred = model.predict_proba(X_pred)[:, 1]
    else:
        pred = np.zeros((X_pred.shape[0], n_classes))
        pred[:, model.classes_] = model.predict_proba(X_pred)
    return oof_auc, pred


def round_and_normalize_probs(probs, decimals=4):
    """多分類機率四捨五入後重新正規化，讓每一列總和為 1（上傳規範）。"""
    probs = np.round(probs, decimals)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return np.round(probs, decimals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data', help='放官方資料與特徵 CSV 的資料夾')
    parser.add_argument('--output', default='submission.csv')
    args = parser.parse_args()
    root = Path(args.data_root)

    # 訓練資料：檔案層級特徵 + 標籤
    info = pd.read_csv(root / '39_Training_Dataset' / 'train_info.csv')
    x_train_fe, train_unique_ids = load_file_features(root / 'tabular_data_train')
    info = info.set_index('unique_id').loc[train_unique_ids]
    groups = info['player_id'].values

    # 處理預測資料（只保留27筆）
    x_predict_fe, test_unique_ids = load_file_features(root / 'tabular_data_test')

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_fe)
    x_predict_scaled = scaler.transform(x_predict_fe[x_train_fe.columns])

    assert len(x_train_scaled) == len(info) == len(groups)

    final_results = {}
    encoders = {}
    for name, col in [('gender', 'gender'), ('hold', 'hold racket handed'),
                      ('years', 'play years'), ('level', 'level')]:
        # 每個目標欄位各自 fit LabelEncoder，避免編碼混淆
        le = LabelEncoder()
        y = le.fit_transform(info[col].values)
        encoders[name] = le
        task_type = "binary" if name in ['gender', 'hold'] else "multi"
        n_classes = None if task_type == "binary" else len(le.classes_)
        print(f"Training {name} model...")
        final_results[f'auc_{name}'], final_results[name] = fit_predict_classifier(
            x_train_scaled, y, groups, task_type, x_predict_scaled, n_classes)

    auc_gender, auc_hold = final_results['auc_gender'], final_results['auc_hold']
    auc_years, auc_level = final_results['auc_years'], final_results['auc_level']
    scores = [auc_gender, auc_hold, auc_years, auc_level]
    print(f'AUC (gender): {auc_gender:.3f}, (hold): {auc_hold:.3f}, (years): {auc_years:.3f}, (level): {auc_level:.3f}')
    print(f'Final Score: {np.mean(scores):.3f}')

    for name in ['years', 'level']:
        final_results[name] = round_and_normalize_probs(final_results[name])

    # 檢查多分類總和是否為1
    for name in ['years', 'level']:
        s = np.abs(final_results[name].sum(axis=1) - 1)
        print(f'{name} max abs(sum-1):', s.max())

    submission = pd.DataFrame({
        'unique_id': test_unique_ids,
        'gender': final_results['gender'],
        'hold racket handed': final_results['hold'],
    })
    for i, cls in enumerate(encoders['years'].classes_):
        submission[f'play years_{cls}'] = final_results['years'][:, i]
    for i, cls in enumerate(encoders['level'].classes_):
        submission[f'level_{cls}'] = final_results['level'][:, i]
    submission.to_csv(args.output, index=False)
    print(f"Submission file created with {len(submission)} entries.")


if __name__ == '__main__':
    main()
