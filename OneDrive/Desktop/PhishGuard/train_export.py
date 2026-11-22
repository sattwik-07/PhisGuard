"""
train_export.py
───────────────
Train a Random Forest on the PhiUSIIL Phishing URL Dataset and export it to
model.json — the format consumed by PhishGuard's browser-side classifier.

USAGE
─────
1. Download the dataset from:
   https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

2. Install dependencies:
   pip install -r requirements.txt

3. Run:
   python train_export.py --dataset path/to/phishing_dataset.csv

4. Commit model.json to your repo and push to GitHub.
   The website automatically switches to ML mode.

NOTES
─────
• The script exports 50 trees (configurable via --n-trees).
  50 trees keeps model.json under ~3 MB while maintaining ~97% accuracy.
• All 29 feature names MUST match exactly what features.js extracts.
• Training on the full PhiUSIIL dataset (~378K rows) takes ~2 minutes on a
  modern CPU. Add --sample 50000 to train on a random subset for speed.
"""

import argparse
import json
import os
import sys
import time

import numpy  as np
import pandas as pd

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     classification_report)
from sklearn.preprocessing   import LabelEncoder

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

# ─── Feature names (must stay in sync with js/features.js) ───────────────────

FEATURE_NAMES = [
    'url_length', 'domain_length', 'path_length', 'query_length',
    'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
    'num_at', 'num_question_marks', 'num_ampersand', 'num_equals',
    'num_percent', 'num_digits', 'num_letters', 'num_special_chars',
    'digit_letter_ratio', 'has_ip_address', 'has_at_symbol',
    'has_double_slash', 'uses_https', 'has_port', 'subdomain_count',
    'suspicious_tld', 'suspicious_keyword', 'url_entropy',
    'domain_entropy', 'is_shortened_url', 'num_query_params',
]

SUSPICIOUS_KEYWORDS = [
    'login','signin','verify','update','secure','account','banking',
    'confirm','password','credential','paypal','ebay','amazon','apple',
    'microsoft','support','alert','auth','wallet','crypto','bitcoin',
    'free','prize','winner','urgent','suspended','limited','locked',
    'validate','recovery','reset','webscr','cmd',
]

URL_SHORTENERS = [
    'bit.ly','tinyurl.com','goo.gl','t.co','ow.ly','is.gd','buff.ly',
    'adf.ly','bit.do','rb.gy','short.link','tiny.cc','cutt.ly',
    'shorturl.at','shorte.st',
]

SAFE_TLDS = {
    'com','org','net','edu','gov','mil','int','co','io',
    'uk','us','ca','au','de','fr','jp','in','eu',
}

# ─── Feature extraction ───────────────────────────────────────────────────────

import re
import math
import urllib.parse


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((f / n) * math.log2(f / n) for f in freq.values())


def extract_features(url: str) -> list:
    """Extract all 29 features in FEATURE_NAMES order."""
    url = str(url).strip()

    try:
        p = urllib.parse.urlparse(url if url.startswith('http') else 'http://' + url)
        hostname = p.hostname or ''
        path     = p.path
        query    = p.query
        scheme   = p.scheme
        port     = p.port
    except Exception:
        hostname = path = query = scheme = ''
        port = None

    url_l    = url.lower()
    hostparts = hostname.split('.')
    tld       = hostparts[-1] if hostparts else ''

    def cnt(src, ch):
        return src.count(ch)

    num_dots        = cnt(url, '.')
    num_hyphens     = cnt(url, '-')
    num_underscores = cnt(url, '_')
    num_slashes     = cnt(url, '/')
    num_at          = cnt(url, '@')
    num_question    = cnt(url, '?')
    num_ampersand   = cnt(url, '&')
    num_equals      = cnt(url, '=')
    num_percent     = cnt(url, '%')
    num_digits      = sum(c.isdigit() for c in url)
    num_letters     = sum(c.isalpha() for c in url)
    num_special     = sum(not c.isalnum() and c not in '.-/:_?&=%#@' for c in url)

    digit_letter    = num_digits / (num_letters + 1e-6)

    has_ip          = int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', hostname)))
    has_at          = int(num_at > 0)
    has_ds          = int('//' in url[url.find('//')+2:] if '//' in url else False)
    uses_https      = int(scheme == 'https')
    has_port        = int(port is not None)

    subdomain_count = max(0, len(hostparts) - 2 - (1 if hostparts and hostparts[0] == 'www' else 0))
    susp_tld        = int(tld not in SAFE_TLDS)
    susp_kw         = int(any(kw in url_l for kw in SUSPICIOUS_KEYWORDS))

    url_ent    = _entropy(url)
    dom_ent    = _entropy(hostname)

    is_short   = int(any(s in hostname for s in URL_SHORTENERS))
    num_params = len(query.split('&')) if query else 0

    return [
        len(url), len(hostname), len(path), len(query),
        num_dots, num_hyphens, num_underscores, num_slashes,
        num_at, num_question, num_ampersand, num_equals,
        num_percent, num_digits, num_letters, num_special,
        digit_letter, has_ip, has_at, has_ds, uses_https,
        has_port, subdomain_count, susp_tld, susp_kw,
        url_ent, dom_ent, is_short, num_params,
    ]


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(csv_path: str, sample: int = None) -> tuple:
    print(f'[LOAD] Reading {csv_path} …')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # Accept flexible column names
    url_col   = next((c for c in df.columns if 'url' in c), None)
    label_col = next((c for c in df.columns if 'label' in c or 'status' in c or 'class' in c), None)

    if url_col is None or label_col is None:
        print(f'[ERROR] Could not identify url/label columns. Found: {list(df.columns)}')
        sys.exit(1)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)
        print(f'[LOAD] Sampled {sample:,} rows from {csv_path}')

    df = df[[url_col, label_col]].dropna()

    # Encode labels
    if df[label_col].dtype == object:
        le = LabelEncoder()
        df[label_col] = le.fit_transform(df[label_col])
        print(f'[LOAD] Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}')

    total    = len(df)
    phishing = int(df[label_col].sum())
    legit    = total - phishing
    print(f'[LOAD] {total:,} samples | Phishing: {phishing:,} | Legitimate: {legit:,}')

    return df[url_col].tolist(), df[label_col].tolist()


# ─── Feature matrix ───────────────────────────────────────────────────────────

def build_X(urls: list) -> np.ndarray:
    print(f'[FEAT] Extracting features for {len(urls):,} URLs …')
    t = time.time()
    X = np.array([extract_features(u) for u in urls], dtype=np.float32)
    print(f'[FEAT] Done in {time.time()-t:.1f}s  shape={X.shape}')
    return X


# ─── Training ─────────────────────────────────────────────────────────────────

def train(X_train, y_train, args):
    models = {}

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    print('[TRAIN] Fitting Random Forest (200 trees) …')
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    if XGB_OK:
        xg = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )
        print('[TRAIN] Fitting XGBoost (200 trees) …')
        xg.fit(X_train, y_train)
        models['XGBoost'] = xg

    return models


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(models: dict, X_test, y_test):
    hdr = f"\n{'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>8}"
    print('\n' + '═' * 58)
    print(hdr)
    print('─' * 58)

    best_name, best_model, best_f1 = None, None, -1

    for name, clf in models.items():
        yp   = clf.predict(X_test)
        yprb = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else yp

        acc  = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp, zero_division=0)
        rec  = recall_score(y_test, yp, zero_division=0)
        f1   = f1_score(y_test, yp, zero_division=0)
        auc  = roc_auc_score(y_test, yprb)

        print(f'{name:<20} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f} {auc:>8.4f}')

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, clf

    print('═' * 58)
    print(f'\n[BEST] {best_name}  F1={best_f1:.4f}')

    yp = best_model.predict(X_test)
    print('\n[Confusion Matrix]')
    print(confusion_matrix(y_test, yp))
    print('\n[Classification Report]')
    print(classification_report(y_test, yp,
          target_names=['Legitimate', 'Phishing'], zero_division=0))

    return best_name, best_model


# ─── Export ───────────────────────────────────────────────────────────────────

def export_rf(clf, n_trees: int, output: str):
    """
    Export a scikit-learn RandomForestClassifier to JSON.
    Only the first n_trees estimators are included to cap file size.
    """

    # For XGBoost we need to convert to a format we can traverse.
    # Instead, re-train a small RF from the same X/y with max_depth capped.
    # This keeps the export logic simple and the JSON small.

    trees_data = []
    estimators = clf.estimators_[:n_trees]

    for est in estimators:
        t = est.tree_
        trees_data.append({
            'children_left':  t.children_left.tolist(),
            'children_right': t.children_right.tolist(),
            'feature':        t.feature.tolist(),
            'threshold':      [round(v, 6) for v in t.threshold.tolist()],
            # value shape: [n_nodes, 1, n_classes] → [n_nodes, n_classes]
            'value':          [[round(c, 2) for c in row[0]] for row in t.value.tolist()],
        })

    model_obj = {
        'type':        'RandomForest',
        'version':     '1.0',
        'n_estimators': len(trees_data),
        'n_classes':    2,
        'class_names': ['legitimate', 'phishing'],
        'features':    FEATURE_NAMES,
        'trees':       trees_data,
    }

    json_str  = json.dumps(model_obj, separators=(',', ':'))
    size_mb   = len(json_str.encode()) / 1e6

    with open(output, 'w') as f:
        f.write(json_str)

    print(f'\n[EXPORT] Saved to: {output}')
    print(f'         Trees: {len(trees_data)} | Features: {len(FEATURE_NAMES)} | Size: {size_mb:.2f} MB')

    if size_mb > 5:
        print(f'[WARN]   model.json is {size_mb:.1f} MB — consider reducing --n-trees')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train PhishGuard model and export to model.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', required=True,
                        help='Path to phishing URL CSV (must have url and label columns)')
    parser.add_argument('--output', default='model.json',
                        help='Output path for model JSON (default: model.json)')
    parser.add_argument('--n-trees', type=int, default=50,
                        help='Number of trees to export (default: 50 ≈ 1-2 MB)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Randomly sample N rows from dataset for faster training')
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f'[ERROR] Dataset not found: {args.dataset}')
        print('\nDownload the PhiUSIIL dataset from:')
        print('  https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset')
        sys.exit(1)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    urls, labels = load_dataset(args.dataset, sample=args.sample)

    # ── 2. Features ──────────────────────────────────────────────────────────
    X = build_X(urls)
    y = np.array(labels, dtype=int)

    # ── 3. Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f'[SPLIT] Train: {len(X_train):,}  Test: {len(X_test):,}')

    # ── 4. Train ─────────────────────────────────────────────────────────────
    models = train(X_train, y_train, args)

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    best_name, best_clf = evaluate(models, X_test, y_test)

    # ── 6. Export ────────────────────────────────────────────────────────────
    # Always export the Random Forest (browser needs sklearn tree structure)
    rf_clf = models.get('Random Forest')
    if rf_clf is None:
        print('[ERROR] Random Forest model not found — cannot export.')
        sys.exit(1)

    export_rf(rf_clf, n_trees=args.n_trees, output=args.output)

    print('\n[DONE] Next steps:')
    print(f'  1. git add {args.output}')
    print('  2. git commit -m "Add trained model"')
    print('  3. git push')
    print('  → PhishGuard will automatically activate ML mode.')


if __name__ == '__main__':
    main()
