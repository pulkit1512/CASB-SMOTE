"""Reusable numeric binary KEEL/CSV benchmark and audited custom SMOTE samplers.

Python >=3.11; dependencies: numpy, scipy, scikit-learn, imbalanced-learn.
Notebook workflow also uses pandas, matplotlib, nbformat, nbclient and ipykernel.
PowerShell: python -B binary_oversampling.py --help
Current study stores results in per-dataset notebooks; legacy report commands remain available. Import this file to
use UGASMOTE(...).fit_resample(X_train, y_train); candidate records are in audit_.
Inputs to samplers must be finite, numeric and scaled using training data only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import time
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from threadpoolctl import threadpool_limits


def binary_labels(y):
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"Expected exactly two classes; got {labels.tolist()}")
    return labels[np.argmin(counts)], labels[np.argmax(counts)], counts


def normalized(a):
    return (a - a.min()) / max(float(np.ptp(a)), 1e-12)


def same_neighbors(X, k):
    # Explicit index removal handles duplicated coordinates correctly.
    nn = NearestNeighbors(n_neighbors=min(len(X), k + 1)).fit(X)
    ids = nn.kneighbors(X, return_distance=False)
    return np.asarray([row[row != i][:min(k, len(X)-1)] for i, row in enumerate(ids)])


class UGASMOTE(BaseEstimator):
    """PDF method. Binary labels are arbitrary; fit_resample returns (X, y).

    mode: uncertainty, rarity, combined, full. No class weighting in the PDF
    selection model. Its probability cutoff is the TRAIN minority prior by
    default (documented predeclared setting, not selected from test outcomes).
    Underfilling is explicit; rejected points are never silently replaced.
    Full candidate audit is available in audit_; compact totals in diagnostics_.
    """
    def __init__(self, sampling_strategy=1.0, k_same=5, k_overlap=10,
                 top_fraction=0.25, alpha=0.5, overlap_threshold=0.5,
                 probability_threshold=None, max_attempt_factor=20,
                 mode="full", random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_same = k_same
        self.k_overlap = k_overlap
        self.top_fraction = top_fraction
        self.alpha = alpha
        self.overlap_threshold = overlap_threshold
        self.probability_threshold = probability_threshold
        self.max_attempt_factor = max_attempt_factor
        self.mode = mode
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        minority, _, counts = binary_labels(y)
        if not np.isfinite(X).all():
            raise ValueError("Impute missing values before resampling")
        if self.mode not in ("uncertainty", "rarity", "combined", "full"):
            raise ValueError(self.mode)
        if not 0 < self.top_fraction <= 1 or not 0 < self.sampling_strategy <= 1:
            raise ValueError("Fractions must lie in (0, 1]")
        idx = np.flatnonzero(y == minority)
        Xm = X[idx]
        needed = max(0, math.ceil(counts.max()*self.sampling_strategy)-len(Xm))
        self.audit_ = []
        self.diagnostics_ = dict(requested=needed, attempted=0, accepted=0)
        if not needed:
            return X.copy(), y.copy()
        if len(Xm) < 2:
            raise ValueError("SMOTE interpolation needs two minority observations")
        rng = np.random.default_rng(self.random_state)
        model = LogisticRegression(max_iter=3000, random_state=self.random_state).fit(X, y)
        column = int(np.flatnonzero(model.classes_ == minority)[0])
        uncertainty = normalized(1-model.predict_proba(Xm).max(axis=1))
        n_clusters = min(5, max(2, int(np.sqrt(len(Xm)/2))), len(np.unique(Xm, axis=0)))
        clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.random_state).fit_predict(Xm)
        sizes = np.bincount(clusters)
        rarity = normalized(1-sizes[clusters]/sizes.max())
        priority = (uncertainty if self.mode == "uncertainty" else rarity if self.mode == "rarity"
                    else self.alpha*uncertainty+(1-self.alpha)*rarity)
        sources = np.argsort(-priority, kind="stable")[:max(1, math.ceil(len(Xm)*self.top_fraction))]
        neighbors = same_neighbors(Xm, self.k_same)
        all_nn = NearestNeighbors(n_neighbors=min(self.k_overlap, len(X))).fit(X)
        cutoff = len(Xm)/len(X) if self.probability_threshold is None else self.probability_threshold
        accepted, generated = [], 0
        max_attempts = needed*self.max_attempt_factor
        while generated < needed and len(self.audit_) < max_attempts:
            size = min(256, max_attempts-len(self.audit_), needed-generated)
            src = rng.choice(sources, size=size)
            dst = np.array([rng.choice(neighbors[i]) for i in src])
            lam = rng.random(size)
            candidate = Xm[src]+lam[:, None]*(Xm[dst]-Xm[src])
            overlap = (y[all_nn.kneighbors(candidate, return_distance=False)] != minority).mean(axis=1)
            probability = model.predict_proba(candidate)[:, column]
            for i in range(size):
                reason = "accepted"
                if self.mode == "full":
                    if overlap[i] > self.overlap_threshold:
                        reason = "high_overlap"
                    elif probability[i] < cutoff:
                        reason = "low_probability"
                self.audit_.append(dict(source_id=int(idx[src[i]]), neighbor_id=int(idx[dst[i]]),
                    uncertainty=float(uncertainty[src[i]]), rarity=float(rarity[src[i]]),
                    priority=float(priority[src[i]]), cluster=int(clusters[src[i]]),
                    neighbor_distance=float(np.linalg.norm(Xm[dst[i]]-Xm[src[i]])),
                    lambda_value=float(lam[i]), overlap=float(overlap[i]),
                    probability=float(probability[i]), status=reason))
                if reason == "accepted":
                    accepted.append(candidate[i])
                    generated += 1
        self.diagnostics_ = dict(requested=needed, attempted=len(self.audit_), accepted=generated,
            underfilled=needed-generated, rejection_counts=dict(Counter(a["status"] for a in self.audit_)),
            sources=len(sources), clusters=n_clusters, probability_threshold=float(cutoff))
        if not accepted:
            return X.copy(), y.copy()
        return np.vstack([X, accepted]), np.concatenate([y, np.repeat(minority, generated)])


class LocalMarginSMOTE(BaseEstimator):
    """Local-margin interpolation; exploratory method, no novelty claim.

    Each step is capped at half the source-to-nearest-majority distance. By the
    triangle inequality the generated point is at least as close to its minority
    source as to any observed majority point (not a population safety guarantee).
    Uniform sources retain minority coverage. Optional soft reliability weighting
    reduces isolated-source influence without deleting observations/subgroups.
    """
    def __init__(self, sampling_strategy=1.0, k_same=5, margin_fraction=0.5,
                 reliability=False, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_same = k_same
        self.margin_fraction = margin_fraction
        self.reliability = reliability
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = np.asarray(X,float), np.asarray(y)
        minority, _, counts = binary_labels(y)
        if not np.isfinite(X).all(): raise ValueError("Inputs must be finite")
        if not 0 < self.sampling_strategy <= 1 or not 0 <= self.margin_fraction <= 0.5:
            raise ValueError("ratio must be in (0,1], margin_fraction in [0,0.5]")
        idx = np.flatnonzero(y == minority)
        Xm, Xmajor = X[idx], X[y != minority]
        needed = max(0, math.ceil(counts.max()*self.sampling_strategy)-len(Xm))
        self.audit_ = []
        self.diagnostics_ = dict(requested=needed, attempted=needed, accepted=needed)
        if not needed: return X.copy(), y.copy()
        if len(Xm)<2: raise ValueError("At least two minority observations required")
        neighbors = same_neighbors(Xm,self.k_same)
        dmajor = NearestNeighbors(n_neighbors=1).fit(Xmajor).kneighbors(Xm)[0][:,0]
        dminor = np.linalg.norm(Xm[neighbors]-Xm[:,None,:],axis=2).mean(axis=1)
        quality = dmajor/(dmajor+dminor+1e-12)
        weights = 0.25+0.75*quality**2 if self.reliability else np.ones(len(Xm))
        rng = np.random.default_rng(self.random_state)
        src = rng.choice(len(Xm),size=needed,p=weights/weights.sum())
        dst = np.array([rng.choice(neighbors[i]) for i in src])
        distances = np.linalg.norm(Xm[dst]-Xm[src],axis=1)
        limit = np.minimum(1,self.margin_fraction*dmajor[src]/np.maximum(distances,1e-12))
        lam = rng.random(needed)*limit
        candidates = Xm[src]+lam[:,None]*(Xm[dst]-Xm[src])
        for i in range(needed):
            self.audit_.append(dict(source_id=int(idx[src[i]]),neighbor_id=int(idx[dst[i]]),
                lambda_value=float(lam[i]),lambda_limit=float(limit[i]),
                majority_distance=float(dmajor[src[i]]),reliability=float(quality[src[i]]),
                status="accepted"))
        self.diagnostics_.update(mean_lambda=float(lam.mean()),
            clipped_fraction=float(np.mean(limit<1)),zero_step_fraction=float(np.mean(lam*distances==0)),
            distinct_sources=int(len(np.unique(src))),reliability_weighting=self.reliability)
        return np.vstack([X,candidates]),np.concatenate([y,np.repeat(minority,needed)])


class CASBSMOTE(BaseEstimator):
    """CASB PDF reference algorithm with per-candidate audits.

    Matches the appendix including original-branch behavior after an empty-pool
    fallback: a rare fallback still uses two neighbors and the rare margin.
    Balanced input is a no-op, fixing the reference's ambiguous equal-count labels.
    Geometry is measured before optional majority removal. No held-out data used.
    """
    def __init__(self, ratio=0.75, k_all=10, k_same=5, weights=(0.5,0.3,0.2),
                 margin=0.5, rare_margin=0.35, drop_majority=0.0, random_state=42,
                 unrestricted=False, generator='smote'):
        self.ratio = ratio
        self.k_all = k_all
        self.k_same = k_same
        self.weights = weights
        self.margin = margin
        self.rare_margin = rare_margin
        self.drop_majority = drop_majority
        self.random_state = random_state
        self.unrestricted = unrestricted
        self.generator = generator

    def fit_resample(self, X, y):
        X,y = np.asarray(X,float),np.asarray(y)
        if X.ndim!=2 or y.ndim!=1 or len(X)!=len(y):
            raise ValueError('X must be 2D with matching 1D y')
        if not np.isfinite(X).all(): raise ValueError('Impute missing values first')
        if self.generator not in ('smote', 'ros'):
            raise ValueError('Supported CASB generators: smote, ros')
        minority,majority,counts = binary_labels(y)
        if not 0<self.ratio<=1 or not 0<=self.drop_majority<=0.3:
            raise ValueError('ratio in (0,1], drop_majority in [0,0.3] required')
        if self.k_same<1 or self.k_all<1 or self.margin<=0 or self.rare_margin<=0:
            raise ValueError('Neighbors and margins must be positive')
        weights = np.asarray(self.weights,float)
        if weights.shape!=(3,) or np.any(weights<0) or not np.isclose(weights.sum(),1):
            raise ValueError('Three nonnegative weights summing to one required')
        if counts.min()<2: raise ValueError('Each class needs at least two observations')
        self.audit_, self.dropped_indices_ = [], np.array([],int)
        self.diagnostics_ = dict(requested=0,accepted=0,majority_dropped=0)
        if counts[0]==counts[1]:
            self.generated_ = np.empty((0,X.shape[1]))
            return X.copy(),y.copy()
        rng = np.random.default_rng(self.random_state)
        ids_min,ids_maj = np.flatnonzero(y==minority),np.flatnonzero(y==majority)
        Xm,Xj = X[ids_min],X[ids_maj]
        nmin,nmaj = len(Xm),len(Xj)
        all_nn = NearestNeighbors(n_neighbors=min(len(X),self.k_all+2)).fit(X)
        all_rows = all_nn.kneighbors(Xm,return_distance=False)
        other = [row[row!=ids_min[i]][:self.k_all] for i,row in enumerate(all_rows)]
        overlap = np.array([np.mean(y[row]==majority) for row in other])
        support = np.array([np.sum(y[row]==minority) for row in other])
        km = min(nmin-1,self.k_same)
        near_min = NearestNeighbors(n_neighbors=nmin if nmin<=km+1 else km+2).fit(Xm)
        neighbor_ids = [row[row!=i][:km] for i,row in enumerate(near_min.kneighbors(Xm,return_distance=False))]
        neighbor_dist = [np.linalg.norm(Xm[row]-Xm[i],axis=1) for i,row in enumerate(neighbor_ids)]
        dmajor = NearestNeighbors(n_neighbors=1).fit(Xj).kneighbors(Xm)[0].ravel()
        third = np.array([d[min(2,len(d)-1)] for d in neighbor_dist])
        border = np.flatnonzero((overlap>0.2)&(overlap<=0.7)&(support>=2)&(dmajor>0))
        rare = np.flatnonzero((third>=np.quantile(third,0.75))&(support>=2)&(dmajor>0))
        keep = np.ones(len(y),bool)
        if self.drop_majority>0 and nmaj>3:
            major_rows = all_nn.kneighbors(Xj,return_distance=False)
            safe = []
            for i,row in enumerate(major_rows):
                other_major = row[row!=ids_maj[i]][:self.k_all]
                if len(other_major) and not np.any(y[other_major]==minority): safe.append(i)
            if safe:
                near_major = NearestNeighbors(n_neighbors=min(nmaj,5)).fit(Xj)
                rows = near_major.kneighbors(Xj,return_distance=False)
                redundancy = np.zeros(nmaj)
                for i,row in enumerate(rows):
                    other_major = row[row!=i]
                    redundancy[i] = np.linalg.norm(Xj[other_major[min(2,len(other_major)-1)]]-Xj[i])
                safe = np.array(safe,int)
                remove = min(int(np.floor(self.drop_majority*nmaj)),len(safe),nmaj-2)
                chosen = safe[np.argsort(redundancy[safe],kind='stable')[:remove]]
                self.dropped_indices_ = ids_maj[chosen]
                keep[self.dropped_indices_] = False
        surviving = nmaj-len(self.dropped_indices_)
        needed = max(nmin,math.ceil(self.ratio*surviving))-nmin
        allocation = np.floor(weights*needed).astype(int)
        for i in np.argsort(-(weights*needed-allocation),kind='stable')[:needed-allocation.sum()]:
            allocation[i]+=1
        modes = ['coverage','border','rare']
        pools = [np.arange(nmin),border,rare]
        created,mode_counts = [],Counter()
        for mode_idx,number in enumerate(allocation):
            pool,mode = pools[mode_idx],modes[mode_idx]
            if not len(pool): pool,mode = pools[0],'coverage_fallback'
            for _ in range(int(number)):
                s = int(rng.choice(pool))
                neighbors = neighbor_ids[s]
                if mode_idx==2 and len(neighbors)>=2: neighbors=neighbors[:2]
                t = int(rng.choice(neighbors))
                edge = Xm[t]-Xm[s]
                length = np.linalg.norm(edge)
                if length==0: limit=0.
                elif self.unrestricted or (mode_idx==0 and overlap[s]<=0.2): limit=1.
                else: limit=min(1.,(self.rare_margin if mode_idx==2 else self.margin)*dmajor[s]/length)
                lam = rng.uniform(0,limit) if limit>0 else 0.
                if self.generator == 'ros':
                    lam = 0.0
                created.append(Xm[s]+lam*edge)
                mode_counts[mode]+=1
                if lam==0: mode_counts['zero_step_duplicates']+=1
                self.audit_.append(dict(source_id=int(ids_min[s]),neighbor_id=int(ids_min[t]),
                    lambda_value=float(lam),lambda_limit=float(limit),overlap=float(overlap[s]),
                    same_support=int(support[s]),majority_distance=float(dmajor[s]),
                    branch=modes[mode_idx],actual_branch=mode,status='accepted'))
        xr,yr = X[keep].copy(),y[keep].copy()
        self.generated_ = np.asarray(created).reshape(-1,X.shape[1])
        if created:
            xr=np.vstack([xr,self.generated_])
            yr=np.concatenate([yr,np.repeat(minority,len(created))])
        self.diagnostics_=dict(requested=needed,accepted=len(created),attempted=len(created),
            original_minority=nmin,original_majority=nmaj,majority_dropped=len(self.dropped_indices_),
            target_ratio=self.ratio,final_ratio=float(np.sum(yr==minority)/np.sum(yr==majority)),
            modes=dict(mode_counts),unique_sources_used=len({a['source_id'] for a in self.audit_}),
            border_pool=len(border),rare_pool=len(rare),allocation=allocation.tolist())
        return xr,yr


class CASBEdited(BaseEstimator):
    """Experimental CASB revision: broad coverage, full interpolation, majority ENN.

    This combines established mechanisms; it is not claimed to be a novel ENN
    algorithm. Preserve all original minority observations. ENN only targets the
    ORIGINAL majority label even if oversampling makes class counts equal.
    Ratio is the pre-cleaning target; post-cleaning counts are explicitly audited.
    """
    def __init__(self, sampling_strategy=0.75, clean=True, random_state=42):
        self.sampling_strategy=sampling_strategy
        self.clean=clean
        self.random_state=random_state

    def fit_resample(self,X,y):
        X,y=np.asarray(X,float),np.asarray(y)
        minority,majority,counts=binary_labels(y)
        generator=CASBSMOTE(ratio=self.sampling_strategy,weights=(0.7,0.2,0.1),
                            unrestricted=True,random_state=self.random_state)
        xr,yr=generator.fit_resample(X,y)
        self.audit_=generator.audit_
        self.diagnostics_=dict(generator.diagnostics_)
        self.generator_=generator
        self.removed_indices_=np.array([],int)
        if self.clean and counts.min()!=counts.max():
            cleaner=EditedNearestNeighbours(sampling_strategy=[majority],n_neighbors=3,kind_sel='mode')
            cleaned_x,cleaned_y=cleaner.fit_resample(xr,yr)
            self.removed_indices_=np.setdiff1d(np.arange(len(yr)),cleaner.sample_indices_)
            if sum(cleaned_y==majority)<2:
                raise ValueError('Majority cleaning left fewer than two observations')
            xr,yr=cleaned_x,cleaned_y
        self.diagnostics_.update(majority_enn_removed=len(self.removed_indices_),
            final_minority=int(sum(yr==minority)),final_majority=int(sum(yr==majority)),
            final_ratio=float(sum(yr==minority)/sum(yr==majority)))
        return xr,yr


def read_data(path, target=None, allow_nominal=False):
    """Numeric KEEL .dat/.arff or headered CSV; reject nominal inputs explicitly.

    Categorical interpolation needs a separate SMOTENC comparison; it is never
    silently treated as a continuous variable. '?' and empty values are missing.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in (".dat", ".arff"):
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data")
        attrs = [re.match(r"\s*@attribute\s+('.*?'|\".*?\"|\S+)\s+(.*)", line, re.I)
                 for line in lines[:start] if line.strip().lower().startswith("@attribute")]
        names = [m.group(1).strip("'\"") for m in attrs]
        outputs = [line.split(None, 1)[1].strip(" '\"") for line in lines[:start]
                   if line.strip().lower().startswith("@outputs")]
        target = target or (outputs[0] if outputs else names[-1])
        col = names.index(target)
        nominal = [names[i] for i, m in enumerate(attrs) if i != col and "{" in m.group(2)]
        if nominal and not allow_nominal:
            raise ValueError(f"{path.name}: nominal features {nominal}; numeric-only benchmark")
        rows = list(csv.reader(line for line in lines[start+1:] if line.strip() and not line.lstrip().startswith(('%', '#'))))
    else:
        rows = list(csv.reader(text.splitlines()))
        names = rows.pop(0)
        col = names.index(target) if target else len(names)-1
    rows = [[v.strip().strip("'\"") for v in row] for row in rows]
    y = np.asarray([row[col] for row in rows])
    categorical = [i for i,m in enumerate(attrs) if i != col and '{' in m.group(2)] if path.suffix.lower() in ('.dat','.arff') else []
    X = np.array([[np.nan if v in ("?", "", "NaN", "nan") else v if i in categorical else float(v)
                   for i, v in enumerate(row) if i != col] for row in rows], object if categorical else float)
    binary_labels(y)
    return X, y


def classifier(name, seed, weighted=False):
    weight = "balanced" if weighted else None
    if name == "lr":
        return LogisticRegression(max_iter=3000, class_weight=weight, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
            class_weight=weight, random_state=seed, n_jobs=1)
    if name == 'svm':
        return SVC(C=1.0,kernel='rbf',gamma='scale',class_weight=weight,random_state=seed)
    if name == 'knn':
        if weighted: raise ValueError('KNN does not support class_weight; no substitute used')
        return KNeighborsClassifier(n_neighbors=7,weights='distance',n_jobs=1)
    if name == 'gnb':
        if weighted: raise ValueError('GaussianNB does not support class_weight; no substitute used')
        return GaussianNB(var_smoothing=1e-9)
    raise ValueError(name)


BASELINES = ["original", "weighted", "ros", "rus", "smote", "bsmote", "adasyn", "smoteenn", "smotetomek"]
PDF_METHODS = ["uga_uncertainty", "uga_rarity", "uga_combined", "uga_full"]
NEW_METHODS = ["margin", "margin_reliable"]


def sampler(name, y, seed, ratio=1.0):
    _, _, counts = binary_labels(y)
    if name in ("original", "weighted") or ratio <= counts.min()/counts.max():
        return None
    kw = dict(sampling_strategy=ratio, random_state=seed)
    k = min(5, counts.min()-1)
    if k < 1 and name not in ("ros", "rus"):
        raise ValueError("Too few minority observations for neighbor resampling")
    if name == "ros": return RandomOverSampler(**kw)
    if name == "rus": return RandomUnderSampler(**kw)
    if name == "smote": return SMOTE(k_neighbors=k, **kw)
    if name == "bsmote": return BorderlineSMOTE(k_neighbors=k, kind="borderline-1", **kw)
    if name == "adasyn": return ADASYN(n_neighbors=k, **kw)
    if name == "smoteenn": return SMOTEENN(smote=SMOTE(k_neighbors=k, **kw), **kw)
    if name == "smotetomek": return SMOTETomek(smote=SMOTE(k_neighbors=k, **kw), **kw)
    if name.startswith("uga_"): return UGASMOTE(mode=name[4:], **kw)
    if name in NEW_METHODS: return LocalMarginSMOTE(reliability=name=="margin_reliable", **kw)
    if name == 'casb': return CASBSMOTE(ratio=ratio,random_state=seed)
    if name == 'casb_ros': return CASBSMOTE(ratio=ratio,random_state=seed,generator='ros')
    if name == 'casb_rus10': return CASBSMOTE(ratio=ratio,drop_majority=0.1,random_state=seed)
    if name == 'casb_relaxed': return CASBEdited(clean=False,**kw)
    if name == 'casb_edited': return CASBEdited(clean=True,**kw)
    raise ValueError(name)


def evaluate_fit(Xtrain, ytrain, Xtest, ytest, method, clf, seed, ratio):
    start = time.perf_counter()
    prep = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    if Xtrain.dtype == object:
        categorical = [i for i in range(Xtrain.shape[1]) if any(isinstance(v,str) for v in Xtrain[:,i])]
        numeric = [i for i in range(Xtrain.shape[1]) if i not in categorical]
        prep = ColumnTransformer([
            ('numeric',prep,numeric),
            ('categorical',make_pipeline(SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(handle_unknown='ignore',sparse_output=False)),categorical)],sparse_threshold=0)
    train = prep.fit_transform(Xtrain)
    test = prep.transform(Xtest)
    s = sampler(method, ytrain, seed, ratio)
    if s is not None:
        train, yr = s.fit_resample(train, ytrain)
    else:
        yr = ytrain
    sample_seconds = time.perf_counter()-start
    model = classifier(clf, seed, method == "weighted")
    # Tiny KEEL minority classes can leave <7 observations after RUS. Apply the
    # same feasibility rule to every method; no change on earlier larger fits.
    if clf=='knn': model.set_params(n_neighbors=min(7,len(yr)))
    model.fit(train, yr)
    pred = model.predict(test)
    score = (model.predict_proba(test)[:, list(model.classes_).index(1)]
             if hasattr(model,'predict_proba') else model.decision_function(test))
    tn, fp, fn, tp = confusion_matrix(ytest, pred, labels=[0, 1]).ravel()
    result = dict(ba=float(balanced_accuracy_score(ytest, pred)),
        f1=float(f1_score(ytest, pred, zero_division=0)),
        macro_f1=float(f1_score(ytest, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(ytest, pred, average="weighted", zero_division=0)),
        precision=float(precision_score(ytest, pred, zero_division=0)),
        recall=float(recall_score(ytest, pred, zero_division=0)),
        specificity=float(tn/max(tn+fp, 1)), gmean=float(np.sqrt(tp/max(tp+fn,1)*tn/max(tn+fp,1))),
        ap=float(average_precision_score(ytest, score)), auc=float(roc_auc_score(ytest, score)),
        confusion=[int(v) for v in (tn, fp, fn, tp)], resampled_counts=np.bincount(yr, minlength=2).tolist(),
        sample_seconds=round(sample_seconds, 5), seconds=round(time.perf_counter()-start, 5))
    if hasattr(s, "diagnostics_"):
        result["audit"] = s.diagnostics_
    if clf=='knn': result['effective_classifier_neighbors']=model.n_neighbors
    return result


def choose_ratio(X, y, method, clf, seed, ratios, selection='ba'):
    if len(ratios) == 1 or method in ("original", "weighted"):
        return ratios[0], []
    # Identical train-only validation split and candidate budget for all samplers.
    tr, va = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed)
    scores = []
    for ratio in ratios:
        try:
            metrics = evaluate_fit(X[tr], y[tr], X[va], y[va], method, clf, seed, ratio)
            value = metrics['ba'] if selection=='ba' else (metrics['gmean']+metrics['f1'])/2
            scores.append(dict(ratio=ratio, ba=value))
        except ValueError as error:
            scores.append(dict(ratio=ratio, error=str(error)))
    valid = [s for s in scores if "ba" in s]
    if not valid:
        raise ValueError(f"All validation candidates failed: {scores}")
    # Predeclared tie breaker: smaller resampling ratio.
    best = max(valid, key=lambda s: (s["ba"], -s["ratio"]))
    return best["ratio"], scores


CASB_VARIANTS = ('casb', 'casb_relaxed', 'casb_edited')


def select_validation_candidate(validation_by_method, methods=CASB_VARIANTS):
    """Select using inner-validation scores ONLY, never outer-test outcomes.

    Historical validation field 'ba' contains the gmean_f1 objective when that
    selection mode was used. Ties: lower ratio, then declared method order.
    Caller must ensure all trials use identical inner splits and objective.
    """
    candidates = []
    for order, method in enumerate(methods):
        trials = validation_by_method.get(method, [])
        if not trials:
            raise ValueError(f'Missing validation trials for {method}')
        for trial in trials:
            if 'ba' in trial and np.isfinite(trial['ba']):
                candidates.append((float(trial['ba']), -float(trial['ratio']),
                                   -order, method, float(trial['ratio'])))
    if not candidates:
        raise ValueError('No successful validation candidate')
    score, _, _, method, ratio = max(candidates)
    return dict(method=method, ratio=ratio, validation_score=score)


def choose_casb_variant(X, y, clf, seed, ratios=(.5, .75, 1.)):
    """Deployable training-only search; nine inner fits for the default grid.

    Refit the selected method/ratio on all training data with evaluate_fit.
    This generic model-selection procedure is not itself a novelty claim.
    """
    if len(ratios) < 2:
        raise ValueError('Variant search requires at least two ratio candidates')
    validation = {}
    for method in CASB_VARIANTS:
        _, validation[method] = choose_ratio(
            X, y, method, clf, seed, list(ratios), 'gmean_f1')
    return select_validation_candidate(validation), validation


def data_splits(root, dataset, folds, seed, target=None, allow_nominal=False):
    directory = root/dataset
    if directory.is_dir():
        for fold in range(1, folds+1):
            # A few KEEL directory names differ from their file prefix.
            train_matches = list(directory.glob(f"*-{folds}-{fold}tra.dat"))
            test_matches = list(directory.glob(f"*-{folds}-{fold}tst.dat"))
            if len(train_matches) != 1 or len(test_matches) != 1:
                raise ValueError(f"Expected one train/test pair for {directory}, fold {fold}")
            tra, tst = train_matches[0], test_matches[0]
            X, labels = read_data(tra,allow_nominal=allow_nominal)
            Xt, labels_t = read_data(tst,allow_nominal=allow_nominal)
            minority, _, _ = binary_labels(labels)
            if not set(labels_t).issubset(set(labels)):
                raise ValueError("Test labels absent from training")
            yield fold, X, (labels == minority).astype(int), Xt, (labels_t == minority).astype(int), [tra, tst]
    else:
        path = Path(dataset)
        X, labels = read_data(path, target)
        for fold, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, labels), 1):
            minority, _, _ = binary_labels(labels[tr])
            yield fold, X[tr], (labels[tr] == minority).astype(int), X[te], (labels[te] == minority).astype(int), [path]


def append(path, text):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text)


def summarize(records):
    ok = [r for r in records if "ba" in r]
    groups = defaultdict(list)
    for r in ok:
        groups[(r["dataset"], r["classifier"], r["method"])].append(r)
    rows = ["\n### Dataset/classifier results\n", "Values are mean +/- sample SD across folds/seeds, not independent confidence intervals.\n",
            "| Dataset | Model | Method | n | Balanced accuracy | Minority F1 | AP | G-mean |\n",
            "|---|---|---|---:|---:|---:|---:|---:|\n"]
    for key, values in sorted(groups.items()):
        def stat(m):
            a = [v[m] for v in values]
            return f"{np.mean(a):.4f} +/- {np.std(a, ddof=1) if len(a)>1 else 0:.4f}"
        rows.append(f"| {' | '.join(key)} | {len(values)} | {stat('ba')} | {stat('f1')} | {stat('ap')} | {stat('gmean')} |\n")
    rows.extend(["\n### Equal-weight dataset/classifier averages\n", "| Method | Groups | BA | F1 | AP | G-mean |\n", "|---|---:|---:|---:|---:|---:|\n"])
    aggregates = defaultdict(list)
    for (_, _, method), values in groups.items():
        aggregates[method].append({m:np.mean([v[m] for v in values]) for m in ("ba", "f1", "ap", "gmean")})
    for method, values in sorted(aggregates.items(), key=lambda p:-np.mean([v["ba"] for v in p[1]])):
        rows.append(f"| {method} | {len(values)} | " + " | ".join(f"{np.mean([v[m] for v in values]):.4f}" for m in ("ba", "f1", "ap", "gmean")) + " |\n")
    rows.append("\nFailures are recorded, never silently substituted. Cross-method averages require equal coverage; inspect errors below.\n")
    for r in records:
        if "error" in r: rows.append(f"- ERROR: {r}\n")
    return "".join(rows)


def saved_records(path, tag):
    active, records = False, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Run: "):
            active = line.removeprefix("## Run: ") == tag
            if active:
                records = []  # Latest invocation wins; earlier evidence stays in Markdown.
        if active and line.startswith('{"dataset":'):
            records.append(json.loads(line))
    if not records: raise ValueError(f"No records for run {tag}")
    return records


def verify_saved(path, tag, root, candidate, folds):
    """Refit the first successful candidate setting to detect metric/code drift."""
    row = next(r for r in saved_records(path,tag) if 'ba' in r and r['method']==candidate)
    for fold,X,y,Xt,yt,_ in data_splits(root,row['dataset'],folds,row['seed']):
        if fold != row['fold']: continue
        actual = evaluate_fit(X,y,Xt,yt,row['method'],row['classifier'],row['seed'],row['ratio'])
        for metric in ('ba','f1','macro_f1','weighted_f1','precision','recall','specificity','gmean','ap','auc'):
            if not np.isclose(actual[metric],row[metric],rtol=0,atol=1e-12):
                raise AssertionError(f'{tag}: {metric} changed')
        if actual['confusion'] != row['confusion']:
            raise AssertionError(f'{tag}: confusion changed')
        print(f"PASS: saved {tag}/{candidate} first setting reproduced (10 metrics and confusion matrix)")
        return


def analyze_run(path, tag, candidate="margin"):
    """Read saved evidence; compare paired settings without refitting models."""
    records = saved_records(path,tag)
    key = lambda r:(r['dataset'],r['classifier'],r['seed'],r['fold'])
    candidates = {key(r):r for r in records if r['method']==candidate and 'ba' in r}
    if not candidates: raise ValueError(f"No {candidate} results")
    lines = [f"\n## Paired comparison: {tag}, candidate {candidate}\n\n",
        "Differences are candidate minus baseline, in percentage points. Dataset means average both models, all folds and seeds. Seeds reuse the supplied folds; these are not independent data replications.\n\n",
        "| Baseline | Paired fits | BA difference | F1 difference | AP difference | Dataset wins / ties / losses |\n",
        "|---|---:|---:|---:|---:|---:|\n"]
    by_dataset = defaultdict(dict)
    for baseline in BASELINES:
        controls = {key(r):r for r in records if r['method']==baseline and 'ba' in r}
        shared = sorted(candidates.keys() & controls.keys())
        if not shared: continue
        differences = defaultdict(list)
        for k in shared:
            differences[k[0]].append({m:candidates[k][m]-controls[k][m] for m in ('ba','f1','ap')})
        means = {d:{m:float(np.mean([v[m] for v in values])) for m in ('ba','f1','ap')}
                 for d,values in differences.items()}
        values = [v['ba'] for v in means.values()]
        win,loss = sum(v>1e-12 for v in values),sum(v < -1e-12 for v in values)
        lines.append(f"| {baseline} | {len(shared)} | "+" | ".join(f"{100*np.mean([v[m] for v in means.values()]):+.3f}" for m in ('ba','f1','ap'))+f" | {win} / {len(values)-win-loss} / {loss} |\n")
        for d,values in means.items(): by_dataset[d][baseline] = values['ba']
    core = ['smote','bsmote','ros','rus']
    lines.extend(["\n| Dataset | vs SMOTE | vs BSMOTE | vs ROS | vs RUS |\n","|---|---:|---:|---:|---:|\n"])
    for d,values in sorted(by_dataset.items()):
        lines.append(f"| {d} | "+" | ".join(f"{100*values[b]:+.3f}" if b in values else 'missing' for b in core)+" |\n")
    errors = [r for r in records if 'error' in r]
    lines.append(f"\nSaved records: {len(records)}; failures: {len(errors)}. Candidate fits: {len(candidates)}.\n")
    lines.append("Small dataset count and correlated folds preclude a strong significance/generalization claim. No fold-level t-test is used.\n")
    text = ''.join(lines)
    append(path,text)
    print(text)


def inventory(root):
    print("dataset,rows,features,minority,majority,imbalance_ratio,status")
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        try:
            raw = [p for p in d.glob('*.dat') if not re.search(r'-\d+-\d+(tra|tst)\.dat$',p.name)]
            if len(raw) != 1:
                raise ValueError(f"Expected one raw data file, found {len(raw)}")
            X, y = read_data(raw[0])
            _, _, counts = binary_labels(y)
            print(f"{d.name},{len(y)},{X.shape[1]},{counts.min()},{counts.max()},{counts.max()/counts.min():.2f},numeric")
        except ValueError as error:
            print(f"{d.name},,,,,,unsupported: {error}")


def casb_self_test():
    from sklearn.datasets import make_classification
    X,y=make_classification(n_samples=150,n_features=5,n_informative=3,
                            weights=[0.8,0.2],random_state=8)
    X=StandardScaler().fit_transform(X)
    for drop in (0.,0.1,0.2):
        original=X.copy(),y.copy()
        s=CASBSMOTE(drop_majority=drop,random_state=42)
        xr,yr=s.fit_resample(X,y)
        second=CASBSMOTE(drop_majority=drop,random_state=42).fit_resample(X,y)
        assert np.array_equal(xr,second[0]) and np.array_equal(yr,second[1])
        assert np.array_equal(X,original[0]) and np.array_equal(y,original[1])
        assert sum(yr==1)==max(sum(y==1),math.ceil(s.ratio*sum(yr==0)))
        assert sum(yr==0)==sum(y==0)-len(s.dropped_indices_)
        assert len(s.audit_)==len(s.generated_)
        nn=NearestNeighbors(n_neighbors=12).fit(X)
        for i in s.dropped_indices_:
            row=nn.kneighbors(X[i:i+1],return_distance=False)[0]
            assert y[i]==0 and np.all(y[row[row!=i][:10]]==0)
        for point,a in zip(s.generated_,s.audit_):
            i,j=a['source_id'],a['neighbor_id']
            assert i!=j and y[i]==y[j]==1
            assert 0<=a['lambda_value']<=a['lambda_limit']<=1
            assert np.allclose(point,X[i]+a['lambda_value']*(X[j]-X[i]))
            if a['branch']!='coverage' or a['overlap']>0.2:
                assert np.linalg.norm(point-X[i])<=0.5*a['majority_distance']+1e-10
        assert set(CASBSMOTE(drop_majority=drop).fit_resample(X,np.where(y,'minor','major'))[1])=={'minor','major'}
    # Identical features and coincident opposing labels force zero steps and empty
    # supported pools. Explicit original-branch fallback still generates exactly.
    coincident=np.zeros((12,2)); labels=np.array([1]*3+[0]*9)
    s=CASBSMOTE(ratio=1.)
    xr,yr=s.fit_resample(coincident,labels)
    assert s.diagnostics_['border_pool']==s.diagnostics_['rare_pool']==0
    assert s.diagnostics_['modes']['zero_step_duplicates']==6
    assert s.diagnostics_['modes']['coverage_fallback']==3
    assert len(xr)==18 and np.isfinite(xr).all()
    assert len(CASBSMOTE().fit_resample(X[:10],np.array([0,1]*5))[0])==10
    for bad in (np.zeros(10),np.arange(10)%3,np.array([1]+[0]*9)):
        try: CASBSMOTE().fit_resample(X[:10],bad)
        except ValueError: pass
        else: raise AssertionError('Invalid class configuration accepted')
    print('PASS: CASB deterministic, immutable, reconstructable, ratio-correct; selective removal, duplicate/fallback and tiny-class checks')


def notebook_benchmark(dataset, methods, seeds=(42,137,2026), ratio=0.75,
                       root=Path('All'), report=Path('progress.md'), tag_prefix='casb_pdf'):
    """Shared notebook execution: persisted evidence, identical splits and models."""
    root,report=Path(root),Path(report)
    tag=f'{tag_prefix}_{dataset}'
    config=dict(dataset=dataset,methods=methods,seeds=list(seeds),ratio=ratio,
        folds=5,classifiers=['lr','rf'],selection='fixed ratio, no tuning, default prediction rule',
        primary=['gmean','f1'],pdf_parameters=CASBSMOTE(ratio=ratio).get_params(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','imbalanced-learn']})
    append(report,f'\n## Run: {tag}\n\n```json\n{json.dumps(config,indent=2)}\n```\n\n<details><summary>Raw notebook fold evidence</summary>\n\n```jsonl\n')
    records,hashes=[],{}
    with threadpool_limits(limits=1):
        for seed in seeds:
            for fold,X,y,Xt,yt,paths in data_splits(root,dataset,5,seed):
                for path in paths: hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                shared=len(set(map(tuple,np.nan_to_num(X,nan=-1e99))) & set(map(tuple,np.nan_to_num(Xt,nan=-1e99))))
                for clf in ('lr','rf'):
                    for method in methods:
                        row=dict(dataset=dataset,seed=seed,fold=fold,classifier=clf,method=method,
                            ratio=ratio,train_counts=np.bincount(y).tolist(),test_counts=np.bincount(yt).tolist(),
                            train_test_shared_feature_vectors=shared)
                        try: row.update(evaluate_fit(X,y,Xt,yt,method,clf,seed,ratio))
                        except ValueError as error: row['error']=str(error)
                        records.append(row)
                        append(report,json.dumps(row,separators=(',',':'))+'\n')
                print(f'{dataset}: seed {seed}, fold {fold}/5 finished',flush=True)
    append(report,'```\n</details>\n\n<details><summary>Input hashes</summary>\n\n```json\n'+json.dumps(hashes,indent=2)+'\n```\n</details>\n'+summarize(records))
    return records,config


CASB_DATASETS=['pima','haberman','yeast3','ecoli4','vowel0','segment0']
CASB_COMPARATORS=['original','weighted','ros','rus','smote','bsmote','adasyn','smoteenn','margin','casb']


def create_notebooks():
    """Create the six requested notebooks; all experiment logic stays in this file."""
    import nbformat as nbf
    for index,dataset in enumerate(CASB_DATASETS,1):
        role='previously used development dataset' if index<=3 else 'not previously benchmarked in this session'
        cells=[nbf.v4.new_markdown_cell(f'''# CASB-SMOTE comparison: {dataset}

This notebook executes the **unchanged CASB PDF method** first, alongside SMOTE,
Borderline-SMOTE (borderline-1), ROS and RUS. Additional controls: no resampling,
class weighting, ADASYN, SMOTE-ENN, and the earlier LocalMarginSMOTE.

Dataset status: **{role}**. The Ecoli and Yeast families are related to previously
seen benchmarks, so new dataset names do not guarantee independent populations.

Protocol fixed before seeing CASB outcomes: original KEEL five outer folds, seeds
42/137/2026, training-only median imputation and standardization; identical logistic
regression and random forest configurations. Fixed target ratio **0.75** for every
sampler; default prediction threshold for every model. No parameter/threshold tuning.
Primary co-outcomes: **G-mean and minority F1**, reported separately. Their average
is used only to order tables, not to conceal a decline in either metric.

Repeating seeds reuses the folds; SD measures variability, not an independent confidence
interval. Exact repeated-feature overlap is audited. Group-disjoint sensitivity and
fairly tuned nested comparisons remain further work.

The shared implementation is `binary_oversampling.py`; the experiment record is
`progress.md`. All tables and figures below are computed from actual executed folds.
No improvement is introduced unless the original CASB comparison fails the target.
'''),nbf.v4.new_markdown_cell('''## Setup

Run from the project directory with a Python environment containing numpy, scipy,
scikit-learn, imbalanced-learn, pandas, matplotlib and ipykernel. The delivered run
uses the existing scientific Python environment and temporary imbalanced-learn
installation; the conditional path below supports that setup without a build folder.
For the verified local setup, select this Python interpreter as the notebook kernel:
`D:/SEM_5/HCII_PROJECT/eeg_emotion_lrm/.venv/Scripts/python.exe`.
Rerunning the experiment cell performs real fitting and appends evidence to progress.md.
'''),nbf.v4.new_code_cell('''import os, sys, json, hashlib, importlib.metadata
from pathlib import Path
sys.dont_write_bytecode = True
temporary_libraries = Path(os.environ.get('TEMP', '/tmp')) / 'ml_project_runtime'
if temporary_libraries.is_dir():
    sys.path.insert(0, str(temporary_libraries))
PROJECT = Path.cwd()
if not (PROJECT / 'binary_oversampling.py').exists():
    raise RuntimeError('Start this notebook in the ML_PROJECT directory')
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from binary_oversampling import notebook_benchmark, CASB_COMPARATORS, read_data
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 180)
'''),nbf.v4.new_code_cell(f'''DATASET = {dataset!r}
METHODS = list(CASB_COMPARATORS)
SEEDS = (42, 137, 2026)
RATIO = 0.75
X_raw, labels_raw = read_data(PROJECT / 'All' / DATASET / f'{{DATASET}}.dat')
classes, counts = np.unique(labels_raw, return_counts=True)
display(pd.DataFrame({{'class': classes, 'count': counts}}))
print(f'Rows: {{len(labels_raw)}}, numeric features: {{X_raw.shape[1]}}, imbalance ratio: {{counts.max()/counts.min():.2f}}')
print('Inventory labels above are descriptive only; minority labels for modeling come from each training fold.')
'''),nbf.v4.new_markdown_cell('''## Execute the original CASB comparison

CASB uses coverage/border/rare budgets 50%/30%/20%, k_all=10, k_same=5,
border margin 0.50, rare margin 0.35, and no majority removal. Empty pools follow
the appendix's coverage fallback, preserving the original branch's interpolation rule.
The optional hybrid is not mixed into this main comparison.
'''),nbf.v4.new_code_cell('''records, configuration = notebook_benchmark(
    DATASET, METHODS, seeds=SEEDS, ratio=RATIO,
    root=PROJECT/'All', report=PROJECT/'progress.md', tag_prefix='casb_pdf')
results = pd.DataFrame(records)
display(Markdown('**Executed configuration and source fingerprint**'))
print(json.dumps(configuration, indent=2))
failures = results[results['error'].notna()] if 'error' in results else pd.DataFrame()
print(f'Completed rows: {len(results)}; failed fits: {len(failures)}')
if len(failures): display(failures[['classifier','method','seed','fold','error']])
valid = results[results['ba'].notna()].copy()
valid['selection_score'] = (valid['gmean'] + valid['f1'])/2
'''),nbf.v4.new_markdown_cell('''## Comparison by classifier

Scores are fold/seed means and sample SD. Higher is better for every score shown;
lower runtime is faster. Compare within the same classifier. Mean F1 is calculated
from fold F1 values, not from averaged precision and recall.
'''),nbf.v4.new_code_cell('''metrics = ['gmean','f1','precision','recall','specificity','ba','ap','auc','seconds']
means = valid.groupby(['classifier','method'])[metrics+['selection_score']].mean()
stds = valid.groupby(['classifier','method'])[metrics].std()
for classifier_name in ['lr','rf']:
    display(Markdown(f'### {classifier_name.upper()}'))
    order = means.loc[classifier_name].sort_values('selection_score', ascending=False).index
    display(means.loc[classifier_name].loc[order].round(4))
    display(Markdown('Fold/seed standard deviations:'))
    display(stds.loc[classifier_name].loc[order, ['gmean','f1','precision','recall']].round(4))
'''),nbf.v4.new_code_cell('''fig, axes = plt.subplots(2,2, figsize=(15,10), constrained_layout=True)
colors = ['#7c3aed' if m=='casb' else '#147d92' for m in METHODS]
for row, classifier_name in enumerate(['lr','rf']):
    for col, metric in enumerate(['gmean','f1']):
        ax = axes[row,col]
        data = means.loc[classifier_name].reindex(METHODS)
        err = stds.loc[classifier_name].reindex(METHODS)[metric]
        ax.barh(METHODS, data[metric], xerr=err, color=colors, alpha=0.9, capsize=2)
        ax.set_xlim(0,1.05)
        ax.set_xlabel(f'{metric} mean (error bars: fold/seed SD)')
        ax.set_title(f'{DATASET}: {classifier_name.upper()}')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.2)
plt.show()
'''),nbf.v4.new_markdown_cell('''## Paired CASB differences against the required baselines

Every subtraction matches classifier, seed and fold. Positive differences favor CASB.
Values are percentage points. A win on the average of two scores does not mean both improved.
'''),nbf.v4.new_code_cell('''paired_rows = []
keys = ['classifier','seed','fold']
casb = valid[valid.method=='casb'].set_index(keys)
for baseline in ['smote','bsmote','ros','rus']:
    control = valid[valid.method==baseline].set_index(keys)
    common = casb.index.intersection(control.index)
    for classifier_name in ['lr','rf']:
        subset = common[common.get_level_values('classifier')==classifier_name]
        delta = casb.loc[subset, ['gmean','f1','precision','recall','ba','ap']] - control.loc[subset, ['gmean','f1','precision','recall','ba','ap']]
        paired_rows.append({'classifier':classifier_name,'baseline':baseline,'paired_fits':len(subset),
            **{f'{m}_delta_pp':100*delta[m].mean() for m in delta.columns}})
paired = pd.DataFrame(paired_rows)
display(paired.round(3))
both_improve = (paired.gmean_delta_pp>0)&(paired.f1_delta_pp>0)
print(f'CASB improves both G-mean and F1 in {both_improve.sum()} of {len(paired)} classifier/baseline comparisons.')
print('No universal superiority claim; inspect precision/recall tradeoffs and cross-dataset results in progress.md.')
'''),nbf.v4.new_markdown_cell('''## Sampler diagnostics and data limitations'''),nbf.v4.new_code_cell('''casb_rows = valid[valid.method=='casb']
diagnostics = pd.json_normalize(casb_rows['audit'].tolist())
display(diagnostics.select_dtypes(include='number').describe().round(3))
print('Maximum exact shared feature vectors across supplied train/test folds:',
      int(results.train_test_shared_feature_vectors.max()))
print('Repeated feature vectors may be different observations; grouping them changes the evaluation question.')
print('Raw metrics, confusion matrices, class counts, runtimes, hashes and audit totals are in progress.md.')
print('Per-candidate source/neighbor/lambda records can be regenerated from CASBSMOTE.audit_.')
''')]
        nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')))
        path=Path(f'{index:02d}_{dataset}_CASB.ipynb')
        if path.exists(): raise FileExistsError(f'Will not overwrite existing notebook: {path}')
        nbf.write(nb,path)
        print(f'Created {path}')


def execute_notebooks():
    """Run actual kernels, checkpointing outputs in the six notebook files."""
    import sys
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    for index,dataset in enumerate(CASB_DATASETS,1):
        path=Path(f'{index:02d}_{dataset}_CASB.ipynb')
        nb=nbformat.read(path,as_version=4)
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(nb,km=km,timeout=3600,resources={'metadata':{'path':str(Path.cwd())}})
        print(f'EXECUTING {path}',flush=True)
        try:
            client.execute(cleanup_kc=True)
        finally:
            nbformat.write(nb,path)
        print(f'COMPLETED {path}',flush=True)


def summarize_casb(report):
    """Aggregate original CASB evidence only; never choose a variant on these tests."""
    import pandas as pd
    records=[r for d in CASB_DATASETS for r in saved_records(report,f'casb_pdf_{d}')]
    frame=pd.DataFrame(records)
    if len(frame)!=1800: raise ValueError(f'Expected 1800 original CASB comparison rows, got {len(frame)}')
    errors=frame[frame.error.notna()] if 'error' in frame else pd.DataFrame()
    metrics=['gmean','f1','precision','recall','ba','ap']
    mean=frame.groupby(['dataset','classifier','method'])[metrics].mean()
    overall=mean.groupby('method')[metrics].mean()
    overall['ordering_score']=(overall.gmean+overall.f1)/2
    overall=overall.sort_values('ordering_score',ascending=False)
    lines=['\n## CASB six-notebook results\n\n',f'Executed {len(frame)} outer model fits; failures: {len(errors)}. Six datasets, five folds, three seeds, two classifiers, ten methods. All comparisons below use fixed ratio 0.75.\n\n',
        '| Method | G-mean | Minority F1 | Precision | Recall | Balanced accuracy | AP |\n',
        '|---|---:|---:|---:|---:|---:|---:|\n']
    for method,row in overall.iterrows():
        lines.append(f'| {method} | '+' | '.join(f'{row[m]:.4f}' for m in metrics)+' |\n')
    lines.extend(['\n### CASB minus required controls\n\n',
        'Equal-weight averages across dataset/classifier means. Differences in percentage points. Dataset wins require both G-mean AND F1 to increase after averaging LR/RF. Fold repeats are correlated, so no naive t-test is reported.\n\n',
        '| Baseline | G-mean difference | F1 difference | Precision difference | Recall difference | Datasets with both gains | Model/dataset cells with both gains |\n',
        '|---|---:|---:|---:|---:|---:|---:|\n'])
    comparisons={}
    for baseline in ['smote','bsmote','ros','rus']:
        delta=mean.xs('casb',level='method')-mean.xs(baseline,level='method')
        comparisons[baseline]=delta
        dataset_delta=delta.groupby('dataset').mean()
        both=int(((dataset_delta.gmean>0)&(dataset_delta.f1>0)).sum())
        cell_both=int(((delta.gmean>0)&(delta.f1>0)).sum())
        lines.append(f'| {baseline} | '+' | '.join(f'{100*delta[m].mean():+.3f}' for m in ['gmean','f1','precision','recall'])+f' | {both}/6 | {cell_both}/12 |\n')
    lines.extend(['\n### Dataset-level CASB differences (G-mean / F1 percentage points)\n\n',
        '| Dataset | vs SMOTE | vs BSMOTE | vs ROS | vs RUS |\n','|---|---:|---:|---:|---:|\n'])
    for dataset in CASB_DATASETS:
        values=[]
        for baseline,delta in comparisons.items():
            row=delta.loc[dataset].mean()
            values.append(f'{100*row.gmean:+.3f} / {100*row.f1:+.3f}')
        lines.append(f'| {dataset} | '+' | '.join(values)+' |\n')
    lines.extend(['\n### Separate classifier means\n\n','| Model | Method | G-mean | F1 | Precision | Recall |\n','|---|---|---:|---:|---:|---:|\n'])
    for (clf,method),row in mean.groupby(['classifier','method']).mean().iterrows():
        lines.append(f'| {clf} | {method} | '+' | '.join(f'{row[m]:.4f}' for m in ['gmean','f1','precision','recall'])+' |\n')
    text=''.join(lines)
    append(report,text)
    print(text)


def self_test():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=160, n_features=5, n_informative=3,
                               weights=[0.85,0.15], random_state=71)
    X = StandardScaler().fit_transform(X)
    for mode in ("uncertainty", "rarity", "combined", "full"):
        xcopy, ycopy = X.copy(), y.copy()
        s = UGASMOTE(mode=mode, random_state=3)
        xr, yr = s.fit_resample(X, y)
        xr2, yr2 = UGASMOTE(mode=mode, random_state=3).fit_resample(X, y)
        assert np.array_equal(xr,xr2) and np.array_equal(yr,yr2)
        assert np.array_equal(X,xcopy) and np.array_equal(y,ycopy)
        assert np.array_equal(xr[:len(X)], X) and np.isfinite(xr).all()
        assert len(xr)-len(X) == s.diagnostics_["accepted"]
        generated = []
        for a in s.audit_:
            i,j = a["source_id"],a["neighbor_id"]
            assert i != j and y[i] == y[j] == 1
            if a["status"] == "accepted":
                generated.append(X[i]+a["lambda_value"]*(X[j]-X[i]))
                if mode == "full":
                    assert a["overlap"] <= s.overlap_threshold
                    assert a["probability"] >= s.diagnostics_["probability_threshold"]
        if generated: assert np.allclose(generated,xr[len(X):])
    # No synthesis when already balanced; stable arbitrary string labels.
    balanced = np.array([0,1]*80)
    assert len(UGASMOTE().fit_resample(X, balanced)[0]) == len(X)
    assert set(UGASMOTE().fit_resample(X, np.where(y,"minor","major"))[1]) == {"minor","major"}
    repeated = np.array([[0.,0.],[0.,0.],[1.,1.]])
    for i,neighbors in enumerate(same_neighbors(repeated,5)):
        assert i not in neighbors and len(neighbors)==2
    for cls in (UGASMOTE,LocalMarginSMOTE):
        try:
            cls().fit_resample(X,np.arange(len(y))%3)
        except ValueError:
            pass
        else:
            raise AssertionError('Multiclass input should be rejected')
        tiny_y = np.zeros(len(y),int)
        tiny_y[0] = 1
        try:
            cls().fit_resample(X,tiny_y)
        except ValueError:
            pass
        else:
            raise AssertionError('Singleton minority should be rejected')
    for reliable in (False,True):
        s = LocalMarginSMOTE(reliability=reliable,random_state=17)
        xr,yr = s.fit_resample(X,y)
        assert np.array_equal(xr,LocalMarginSMOTE(reliability=reliable,random_state=17).fit_resample(X,y)[0])
        assert len(xr) == 2*sum(y==0) and np.array_equal(xr[:len(X)],X)
        for point,a in zip(xr[len(X):],s.audit_):
            i,j = a['source_id'],a['neighbor_id']
            assert i!=j and y[i]==y[j]==1
            assert np.allclose(point,X[i]+a['lambda_value']*(X[j]-X[i]))
            assert np.linalg.norm(point-X[i]) <= np.linalg.norm(X[y==0]-point,axis=1).min()+1e-10
    for method in BASELINES + PDF_METHODS + NEW_METHODS:
        result = evaluate_fit(X[:120], y[:120], X[120:], y[120:], method,"lr",42,1.0)
        assert 0 <= result["ba"] <= 1
    ratio,values = choose_ratio(X[:120],y[:120],'margin','lr',42,[0.5,1.0])
    assert ratio in (0.5,1.0) and len(values)==2
    with tempfile.TemporaryDirectory(prefix='oversampling_checks_') as tmp:
        path = Path(tmp)/'binary.csv'
        path.write_text('label,a,b\nminor,1,?\nmajor,2,3\nmajor,4,5\nminor,6,7\n',encoding='utf-8')
        data,labels = read_data(path,target='label')
        assert data.shape==(4,2) and np.isnan(data[0,1])
        assert labels.tolist()==['minor','major','major','minor']
    print("PASS: deterministic generation, train immutability, label integrity, interpolation, filtering, edge cases, all baseline fits")
    casb_self_test()
    for clean in (False,True):
        s=CASBEdited(clean=clean)
        xr,yr=s.fit_resample(X,y)
        repeat=CASBEdited(clean=clean).fit_resample(X,y)
        assert np.array_equal(xr,repeat[0]) and np.array_equal(yr,repeat[1])
        assert all(np.any(np.all(xr[yr==1]==point,axis=1)) for point in X[y==1])
        assert all(y[i]==0 for i in s.removed_indices_)
        for point,a in zip(s.generator_.generated_,s.audit_):
            i,j=a['source_id'],a['neighbor_id']
            assert y[i]==y[j]==1 and np.allclose(point,X[i]+a['lambda_value']*(X[j]-X[i]))
    for clf in ('svm','knn'):
        for method in ('smote','bsmote','ros','rus','casb','casb_relaxed','casb_edited'):
            result=evaluate_fit(X[:120],y[:120],X[120:],y[120:],method,clf,42,.75)
            assert 0<=result['auc']<=1 and 0<=result['gmean']<=1
    print('PASS: CASB revisions and SVM/KNN execution')


def revision_summary(report=Path('progress.md')):
    import pandas as pd
    rows=saved_records(report,'casb_revision_svm_knn')
    df=pd.DataFrame(rows)
    if len(df)!=1080 or ('error' in df and df.error.notna().any()):
        raise ValueError('Expected 1080 complete successful outer fits; inspect raw evidence')
    df['composite']=(df.gmean+df.f1)/2
    metrics=['composite','gmean','f1','precision','recall','ba','ap']
    means=df.groupby(['dataset','classifier','method'])[metrics].mean()
    avg=means.groupby('method')[metrics].mean().sort_values('composite',ascending=False)
    text='\n## Revised CASB: SVM/KNN outcome\n\n1080 outer fits; two seeds, five folds, six datasets, two classifiers, nine methods. Each resampler has three training-only ratio trials per outer fit.\n\n'
    text+='| Method | Composite | G-mean | F1 | Precision | Recall |\n|---|---:|---:|---:|---:|---:|\n'
    for method,row in avg.iterrows():
        text+=f'| {method} | '+' | '.join(f'{row[m]:.4f}' for m in metrics[:5])+' |\n'
    core=['smote','bsmote','ros','rus']
    for candidate in ['casb_relaxed','casb_edited']:
        text+=f'\n### {candidate}: dataset-level differences\n\nComposite differences in percentage points, equally averaging SVM/KNN.\n\n'
        text+='| Dataset | vs SMOTE | vs BSMOTE | vs ROS | vs RUS | Beats all four | Both metrics beat all four |\n|---|---:|---:|---:|---:|---|---|\n'
        ds=means.groupby(['dataset','method'])[metrics].mean()
        wins=strict=0
        for dataset in CASB_DATASETS:
            c=ds.loc[(dataset,candidate)]
            diffs=[c-ds.loc[(dataset,b)] for b in core]
            win=all(d['composite']>1e-12 for d in diffs)
            both=all(d['gmean']>1e-12 and d['f1']>1e-12 for d in diffs)
            wins+=win; strict+=both
            text+=f'| {dataset} | '+' | '.join(f"{100*d['composite']:+.3f}" for d in diffs)+f' | {win} | {both} |\n'
        text+=f'\nJoint wins: **{wins}/6 datasets** on composite; **{strict}/6** on both G-mean and F1. Majority target (at least 4/6): {"met in this development run" if wins>=4 else "NOT met"}.\n'
        for clf in ['svm','knn']:
            subset=means.xs(clf,level='classifier')
            count=sum(all(subset.loc[(d,candidate),'composite']>subset.loc[(d,b),'composite']+1e-12 for b in core) for d in CASB_DATASETS)
            text+=f'- {clf.upper()} separately: beats all four on composite in {count}/6 datasets.\n'
        text+='\nPairwise dataset wins on composite: '
        text+=', '.join(f"{b} {sum(ds.loc[(d,candidate),'composite']>ds.loc[(d,b),'composite']+1e-12 for d in CASB_DATASETS)}/6" for b in core)+'.\n'
    text+='\nDevelopment results after method design; not an independent superiority or novelty claim. Ratio tuning used a single inner holdout, not nested multi-fold tuning. Different classifier choices can change rankings. All original results remain preserved.\n'
    append(report,text)
    print(text)
    return df


def revision_notebooks():
    """Execute only the added analysis cells; preserve prior executed LR/RF cells."""
    import sys,nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    for index,dataset in enumerate(CASB_DATASETS,1):
        path=Path(f'{index:02d}_{dataset}_CASB.ipynb')
        original=nbformat.read(path,as_version=4)
        marker='# Revised CASB with SVM and KNN'
        if any(c.cell_type=='markdown' and c.source.startswith(marker) for c in original.cells):
            raise ValueError(f'Revision section already exists in {path}')
        cells=[nbformat.v4.new_markdown_cell(f'''{marker}

Dataset: **{dataset}**. The previous LR/RF comparison is preserved above.
This new study uses RBF SVM (C=1, gamma=scale) and distance-weighted KNN (k=7).
Two seeds (42,137), original five folds, identical training-only ratio search
(0.50/0.75/1.00) for all samplers, selection by (G-mean + F1)/2. No test tuning.

`casb_relaxed`: 70/20/10 coverage/border/rare allocation, unrestricted interpolation.
`casb_edited`: same generation plus majority-only ENN with 3-neighbor majority vote.
All original minority observations are preserved. These are experimental hybrids,
not claims of new ENN algorithms. Ratio denotes pre-cleaning counts.

The cells below execute analysis of persisted **actual model-fitting results** from
`progress.md`; they do not retrain models. Full rerun command (PowerShell environment
setup is documented in progress.md):

```powershell
python -B binary_oversampling.py --tag casb_revision_svm_knn --datasets pima haberman yeast3 ecoli4 vowel0 segment0 --classifiers svm knn --methods original ros rus smote bsmote smoteenn casb casb_relaxed casb_edited --seeds 42 137 --ratios 0.5 0.75 1.0 --selection gmean_f1
```

Results are development evidence on reused datasets. Composite wins and gains on
both individual outcomes are distinguished; SD is not a confidence interval.
'''),nbformat.v4.new_code_cell(f'''import sys, os
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(os.environ.get('TEMP','/tmp'))/'ml_project_runtime'))
sys.path.insert(0,str(Path.cwd()))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from binary_oversampling import saved_records
revision=pd.DataFrame(saved_records(Path('progress.md'),'casb_revision_svm_knn'))
revision=revision[revision.dataset=={dataset!r}].copy()
assert len(revision)==180 and ('error' not in revision or revision.error.isna().all())
revision['composite']=(revision.gmean+revision.f1)/2
metrics=['composite','gmean','f1','precision','recall','ba','ap']
summary=revision.groupby(['classifier','method'])[metrics].mean()
for clf in ['svm','knn']:
    display(Markdown(f'### {{clf.upper()}}: mean over folds/seeds'))
    display(summary.loc[clf].sort_values('composite',ascending=False).round(4))
print('Source: 180 completed outer fits; each resampler used three training-only ratio candidates.')
'''),nbformat.v4.new_code_cell('''comparisons=[]
for candidate in ['casb_relaxed','casb_edited']:
    for clf in ['svm','knn']:
        for baseline in ['smote','bsmote','ros','rus','casb','smoteenn']:
            delta=summary.loc[(clf,candidate)]-summary.loc[(clf,baseline)]
            comparisons.append({'candidate':candidate,'classifier':clf,'baseline':baseline,
                **{m+'_delta_pp':100*delta[m] for m in ['composite','gmean','f1','precision','recall']}})
display(pd.DataFrame(comparisons).round(3))
dataset_mean=summary.groupby('method').mean()
for candidate in ['casb_relaxed','casb_edited']:
    deltas=[dataset_mean.loc[candidate]-dataset_mean.loc[b] for b in ['smote','bsmote','ros','rus']]
    print(candidate,'beats all four on composite:',all(d.composite>0 for d in deltas),
          '; improves BOTH metrics against all four:',all(d.gmean>0 and d.f1>0 for d in deltas))
display(Markdown('Selected ratio counts (training-only selection):'))
display(pd.crosstab([revision.classifier,revision.method],revision.ratio))
'''),nbformat.v4.new_code_cell(f'''fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
for ax,clf in zip(axes,['svm','knn']):
    table=summary.loc[clf][['gmean','f1']].sort_values('gmean')
    table.plot.barh(ax=ax,color=['#176b87','#9c3fb2'])
    ax.set_xlim(0,1.02)
    ax.set_title({dataset!r}+' / '+clf.upper())
    ax.set_xlabel('Mean score (higher is better)')
    ax.grid(axis='x',alpha=.2)
plt.show()
print('Dataset reuse and single inner holdout limit claims. No significance test on correlated folds.')
''')]
        added=nbformat.v4.new_notebook(cells=cells)
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
        NotebookClient(added,km=km,timeout=180,resources={'metadata':{'path':str(Path.cwd())}}).execute(cleanup_kc=True)
        original.cells.extend(added.cells)
        nbformat.validate(original)
        nbformat.write(original,path)
        print(f'Updated {path} with executed SVM/KNN comparisons',flush=True)


REVISION_METHODS=['original','ros','rus','smote','bsmote','smoteenn','casb','casb_relaxed','casb_edited']


def numeric_inventory(root):
    supported,excluded=[],{}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir(): continue
        try:
            raw=[p for p in directory.glob('*.dat') if not re.search(r'-\d+-\d+(tra|tst)\.dat$',p.name)]
            if len(raw)!=1: raise ValueError('Expected one original dataset file')
            read_data(raw[0])
            supported.append(directory.name)
        except ValueError as error: excluded[directory.name]=str(error)
    return supported,excluded


def remaining_study(root,report):
    """Frozen remaining-dataset run, resumable at dataset boundaries."""
    datasets,excluded=numeric_inventory(root)
    remaining=[d for d in datasets if d not in CASB_DATASETS]
    print(f'Remaining: {len(remaining)}; excluded: {excluded}',flush=True)
    for dataset in remaining:
        tag=f'casb_remaining_{dataset}'
        try: previous=saved_records(report,tag)
        except ValueError: previous=[]
        if len(previous)==180:
            print(f'SKIP completed {dataset} ({sum("error" in r for r in previous)} errors)',flush=True)
            continue
        if previous: append(report,'\n```\n</details>\nPartial earlier run superseded by retry below.\n')
        config=dict(dataset=dataset,methods=REVISION_METHODS,classifiers=['svm','knn'],
            seeds=[42,137],outer_folds=5,ratios=[.5,.75,1.],selection='mean G-mean/F1 on stratified 25% inner holdout',
            classifier_settings={'svm':{'kernel':'rbf','C':1.,'gamma':'scale'},'knn':{'weights':'distance','k':'min(7,resampled_count)'}},
            sampler_frozen=True,excluded=excluded,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','imbalanced-learn']})
        append(report,f'\n## Run: {tag}\n\n```json\n{json.dumps(config,indent=2)}\n```\n\n<details><summary>Raw remaining-dataset evidence</summary>\n\n```jsonl\n')
        rows,hashes=[],{}
        for seed in [42,137]:
            for fold,X,y,Xt,yt,paths in data_splits(root,dataset,5,seed):
                for path in paths: hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                shared=len(set(map(tuple,np.nan_to_num(X,nan=-1e99)))&set(map(tuple,np.nan_to_num(Xt,nan=-1e99))))
                for clf in ['svm','knn']:
                    for method in REVISION_METHODS:
                        row=dict(dataset=dataset,seed=seed,fold=fold,classifier=clf,method=method,
                            train_counts=np.bincount(y).tolist(),test_counts=np.bincount(yt).tolist(),
                            train_test_shared_feature_vectors=shared)
                        try:
                            ratio,validation=choose_ratio(X,y,method,clf,seed,[.5,.75,1.],'gmean_f1')
                            row.update(ratio=ratio,validation=validation)
                            row.update(evaluate_fit(X,y,Xt,yt,method,clf,seed,ratio))
                        except ValueError as error: row['error']=str(error)
                        rows.append(row)
                        append(report,json.dumps(row,separators=(',',':'))+'\n')
                print(f'{dataset}: seed={seed} fold={fold}/5 done',flush=True)
        append(report,'```\n</details>\n\n<details><summary>Input hashes</summary>\n\n```json\n'+json.dumps(hashes,indent=2)+'\n```\n</details>\n'+summarize(rows))
        print(f'FINISHED {dataset}: {len(rows)} fits, {sum("error" in r for r in rows)} errors',flush=True)


def full_study_summary(root,report):
    """Matched complete-dataset means; distinguish remaining from reused data."""
    import pandas as pd
    supported,excluded=numeric_inventory(root)
    original=saved_records(report,'casb_revision_svm_knn')
    remaining=[]
    for dataset in supported:
        if dataset not in CASB_DATASETS:
            rows=saved_records(report,f'casb_remaining_{dataset}')
            if len(rows)!=180: raise ValueError(f'{dataset}: incomplete ({len(rows)}/180)')
            remaining.extend(rows)
    text='\n## Final full-collection SVM/KNN comparison\n\n'
    text+=f'{len(supported)} numeric datasets; {len(supported)-6} remaining plus six development datasets. Exclusions: {json.dumps(excluded)}.\n\n'
    for name,rows in [('Remaining numeric datasets',remaining),('All numeric datasets',original+remaining)]:
        f=pd.DataFrame(rows)
        if 'error' not in f: f['error']=None
        f['composite']=(f.gmean+f.f1)/2
        errors=f[f.error.notna()]
        bad=set(errors.dataset)
        # A method must have ten successful fold/seed scores per model. Do not
        # compare optimistic available-case averages with different coverage.
        counts=f[f.error.isna()].groupby(['dataset','classifier','method']).size()
        for dataset in f.dataset.unique():
            for clf in ['svm','knn']:
                for method in REVISION_METHODS:
                    if counts.get((dataset,clf,method),0)!=10: bad.add(dataset)
        valid=f[(f.error.isna())&(~f.dataset.isin(bad))]
        metrics=['composite','gmean','f1','precision','recall','ba','ap','auc']
        mean=valid.groupby(['dataset','classifier','method'])[metrics].mean()
        ds=mean.groupby(['dataset','method'])[metrics].mean()
        aggregate=ds.groupby('method')[metrics].mean().sort_values('composite',ascending=False)
        text+=f'### {name}\n\nAttempted outer fits: {len(f)}; errors: {len(errors)}. Matched complete datasets: {valid.dataset.nunique()}/{f.dataset.nunique()}. Excluded from matched means: {sorted(bad)}. Equal weight per dataset and classifier.\n\n'
        text+='| Method | G-mean | Minority F1 | Precision | Recall | Balanced accuracy | AP | ROC AUC | Composite |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for method,row in aggregate.iterrows():
            text+=f'| {method} | '+' | '.join(f'{100*row[m]:.2f}' for m in ['gmean','f1','precision','recall','ba','ap','auc','composite'])+' |\n'
        text+='\n| Baseline | Edited wins / ties / losses (composite) | Wins on BOTH G-mean and F1 |\n|---|---:|---:|\n'
        core=['smote','bsmote','ros','rus']
        datasets=sorted(valid.dataset.unique())
        for baseline in core+['smoteenn','casb']:
            deltas=[ds.loc[(d,'casb_edited')]-ds.loc[(d,baseline)] for d in datasets]
            wins=sum(v.composite>1e-12 for v in deltas)
            losses=sum(v.composite < -1e-12 for v in deltas)
            both=sum(v.gmean>1e-12 and v.f1>1e-12 for v in deltas)
            text+=f'| {baseline} | {wins} / {len(deltas)-wins-losses} / {losses} | {both}/{len(deltas)} |\n'
        joint=sum(all(ds.loc[(d,'casb_edited'),'composite']>ds.loc[(d,b),'composite']+1e-12 for b in core) for d in datasets)
        strict=sum(all(ds.loc[(d,'casb_edited'),m]>ds.loc[(d,b),m]+1e-12 for b in core for m in ['gmean','f1']) for d in datasets)
        text+=f'\nEdited CASB beats all four core baselines on composite on {joint}/{len(datasets)} datasets; on both metrics on {strict}/{len(datasets)}.\n'
        text+='\n| Classifier | Method | G-mean | F1 | Composite |\n|---|---|---:|---:|---:|\n'
        for (clf,method),row in mean.groupby(['classifier','method'])[metrics].mean().iterrows():
            text+=f'| {clf} | {method} | {100*row.gmean:.2f} | {100*row.f1:.2f} | {100*row.composite:.2f} |\n'
        text+='\n<details><summary>Per-dataset average of SVM/KNN (G-mean / F1 in percent)</summary>\n\n| Dataset | '+' | '.join(REVISION_METHODS)+' |\n|---|'+'---:|'*len(REVISION_METHODS)+'\n'
        for d in datasets:
            text+=f'| {d} | '+' | '.join(f'{100*ds.loc[(d,m),"gmean"]:.2f} / {100*ds.loc[(d,m),"f1"]:.2f}' for m in REVISION_METHODS)+' |\n'
        text+='\n</details>\n'
        if len(errors): text+='\nErrors (never silently replaced):\n```json\n'+errors[['dataset','classifier','method','seed','fold','error']].to_json(orient='records',indent=2)+'\n```\n'
    text+='\nProtocol limitations: related dataset families, duplicate-feature overlap, reused development datasets, one inner holdout, and two seeds. Seeds/folds are not independent scientific replications. Performance comparisons do not establish algorithmic novelty. Categorical Abalone inputs require a separate mixed-type extension; this is a 42-dataset numeric study, not a complete 44-dataset claim.\n'
    append(report,text)
    print(text)



def notebook_records(path, dataset, root=Path('All'), rerun=False):
    """Resume actual fits; checkpoint ONLY in this dataset notebook."""
    import nbformat
    path=Path(path)
    nb=nbformat.read(path,as_version=4)
    evidence=dict(nb.metadata.get('benchmark',{}))
    rows=[] if rerun else list(evidence.get('records',[]))
    keys={(r['seed'],r['fold'],r['classifier'],r['method']) for r in rows}
    evidence.update(dataset=dataset,protocol='casb_svm_knn_v1',seeds=[42,137],
        methods=REVISION_METHODS,classifiers=['svm','knn'],ratios=[.5,.75,1.],
        selection='gmean_f1',encoding='training-only one-hot nominal; scaled numeric',
        versions={p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','imbalanced-learn']})
    hashes=dict(evidence.get('input_sha256',{}))
    with threadpool_limits(limits=1):
        for seed in [42,137]:
            for fold,X,y,Xt,yt,paths in data_splits(root,dataset,5,seed,allow_nominal=True):
                for source in paths: hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
                shared=len(set(map(tuple,X)) & set(map(tuple,Xt)))
                for clf in ['svm','knn']:
                    for method in REVISION_METHODS:
                        if (seed,fold,clf,method) in keys: continue
                        row=dict(dataset=dataset,seed=seed,fold=fold,classifier=clf,method=method,
                            train_counts=np.bincount(y).tolist(),test_counts=np.bincount(yt).tolist(),
                            train_test_shared_feature_vectors=shared,
                            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                        try:
                            ratio,validation=choose_ratio(X,y,method,clf,seed,[.5,.75,1.],'gmean_f1')
                            row.update(ratio=ratio,validation=validation)
                            row.update(evaluate_fit(X,y,Xt,yt,method,clf,seed,ratio))
                        except ValueError as error: row['error']=str(error)
                        rows.append(row)
                        evidence.update(records=rows,input_sha256=hashes,status=f'{len(rows)}/180 outer fits')
                        nb.metadata['benchmark']=evidence
                        nbformat.write(nb,path)
                print(f'{dataset}: seed {seed}, fold {fold}: {len(rows)}/180',flush=True)
    return rows


def dataset_notebook_study(root=Path('All'), execute=True):
    """Create all dataset notebooks, migrate old evidence once, then execute/resume."""
    import sys,nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    datasets=CASB_DATASETS+[p.name for p in sorted(root.iterdir()) if p.is_dir() and p.name not in CASB_DATASETS]
    paths=[]
    for index,dataset in enumerate(datasets,1):
        path=Path(f'{index:02d}_{dataset}_CASB.ipynb')
        paths.append(path)
        old=nbformat.read(path,as_version=4) if path.exists() else None
        if old and old.metadata.get('benchmark',{}).get('protocol')=='casb_svm_knn_v1': continue
        try:
            rows=saved_records(Path('progress.md'),'casb_revision_svm_knn' if dataset in CASB_DATASETS else f'casb_remaining_{dataset}')
            rows=[r for r in rows if r['dataset']==dataset]
        except ValueError: rows=[]
        # Preserve historical notebook evidence without displaying LR/RF as part of this study.
        history=[dict(c) for c in old.cells] if old else []
        note=('Sex is encoded with a training-fitted one-hot encoder. Numeric columns are median-imputed and scaled on training data only. '
              'Interpolation can produce fractional dummy columns: this is an encoded-feature-space extension, not categorical-value synthesis.'
              if dataset.startswith('abalone') else 'Numeric columns are median-imputed and scaled on training data only.')
        cells=[nbformat.v4.new_markdown_cell(f'# {dataset}: CASB comparison with SVM and KNN\n\n'
            'Nine methods; supplied five folds; seeds 42 and 137. Each resampler uses the same training-only 25% validation split and ratio grid 0.5/0.75/1.0, selected by mean G-mean/F1.\n\n'
            'RBF SVM: C=1, gamma=scale. KNN: distance weighting, k=min(7, resampled training size). '
            'CASB-relaxed uses 70/20/10 allocation and unrestricted interpolation; CASB-edited adds majority-only ENN.\n\n'+note+'\n\n'
            'Raw records, validation scores, and file hashes are stored in this notebook metadata. Re-running resumes saved fits; set rerun=True to retrain. '
            'Prior completed results were imported once from the historical log. This notebook has no runtime dependency on progress.md. '
            'Historical LR/RF cells, where present, are retained only in metadata. Seeds/folds are correlated; performance alone does not establish novelty.'),
            nbformat.v4.new_code_cell(f"import sys, os\nfrom pathlib import Path\nsys.dont_write_bytecode=True\nsys.path.insert(0,str(Path(os.environ.get('TEMP','/tmp'))/'ml_project_runtime'))\nsys.path.insert(0,str(Path.cwd()))\nfrom binary_oversampling import notebook_records\nimport pandas as pd\nfrom IPython.display import display, Markdown\nrecords=notebook_records({str(path)!r}, {dataset!r}, root=Path({str(root)!r}), rerun=False)\nframe=pd.DataFrame(records)\nprint('Outer fits:',len(frame),'Errors:',sum('error' in r for r in records))"),
            nbformat.v4.new_code_cell("errors=frame[frame['error'].notna()] if 'error' in frame else frame.iloc[:0]\nif len(errors): display(errors[['classifier','method','seed','fold','error']])\nassert len(frame)==180 and len(errors)==0, 'Incomplete comparison: inspect errors; no available-case averages'\nframe['composite']=(frame.gmean+frame.f1)/2\nmetrics=['composite','gmean','f1','precision','recall','specificity','ba','ap','auc','macro_f1','weighted_f1']\nmeans=frame.groupby(['classifier','method'])[metrics].mean()\nfor clf in ['svm','knn']:\n    display(Markdown('## '+clf.upper()+' average (%)'))\n    display((100*means.loc[clf]).round(2).sort_values('composite',ascending=False))\ndisplay(Markdown('## Final comparison: equal average of SVM and KNN (%)'))\ncombined=means.groupby('method').mean().sort_values('composite',ascending=False)\ndisplay((100*combined).round(2))"),
            nbformat.v4.new_code_cell("deltas=pd.DataFrame({b:100*(combined.loc['casb_edited']-combined.loc[b]) for b in ['smote','bsmote','ros','rus','smoteenn','casb']}).T\ndisplay(Markdown('## CASB-edited gain over baselines (percentage points)'))\ndisplay(deltas.round(3))\nprint('Beats all four core baselines on composite:', bool((deltas.loc[['smote','bsmote','ros','rus'],'composite']>1e-10).all()))\ndisplay(Markdown('## Fold-level evidence'))\ndisplay(frame[['seed','fold','classifier','method','ratio']+metrics])\ndisplay(Markdown('## Standard deviations across folds/seeds (not confidence intervals)'))\ndisplay(frame.groupby(['classifier','method'])[metrics].std().round(4))")]
        nb=nbformat.v4.new_notebook(cells=cells)
        nb.metadata['benchmark']=dict(protocol='casb_svm_knn_v1',dataset=dataset,records=rows,status=f'{len(rows)}/180 outer fits',legacy_cells=history)
        nb.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
        nbformat.write(nb,path)
        print(f'Prepared {path}: {len(rows)} imported fits',flush=True)
    if not execute: return
    for path in paths:
        nb=nbformat.read(path,as_version=4)
        if nb.metadata['benchmark'].get('executed_complete'): continue
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[sys.executable,'-B','-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(nb,km=km,timeout=None,resources={'metadata':{'path':str(Path.cwd())}})
        try:
            client.execute(cleanup_kc=True)
        finally:
            saved=nbformat.read(path,as_version=4)
            nb.metadata['benchmark']=saved.metadata['benchmark']
            nb.metadata['benchmark']['executed_complete']=all(c.get('execution_count') is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
            nbformat.write(nb,path)
        print(f'EXECUTED {path}',flush=True)



def generator_extension_study(method="casb_ros"):
    if method not in ("casb_ros", "adasyn"): raise ValueError(method)
    metadata_key="generator_extension" if method=="casb_ros" else "adasyn_extension"
    """CASB-ROS extension; all evidence checkpointed in dataset notebooks."""
    import nbformat
    import pandas as pd
    for path in sorted(Path('.').glob('[0-9][0-9]_*_CASB.ipynb')):
        nb=nbformat.read(path,as_version=4)
        dataset=nb.metadata.benchmark.dataset
        study=dict(nb.metadata.get(metadata_key,{}))
        rows=list(study.get('records',[]))
        hashes=dict(study.get('input_sha256',{}))
        done={(r['seed'],r['fold'],r['classifier']) for r in rows}
        for seed in [42,137]:
            for fold,X,y,Xt,yt,paths in data_splits(Path('All'),dataset,5,seed,allow_nominal=True):
                for p in paths:
                    hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                for clf in ['svm','knn']:
                    if (seed,fold,clf) in done: continue
                    row=dict(dataset=dataset,seed=seed,fold=fold,classifier=clf,method=method)
                    try:
                        ratio,validation=choose_ratio(X,y,method,clf,seed,[.5,.75,1.],'gmean_f1')
                        row.update(ratio=ratio,validation=validation)
                        row.update(evaluate_fit(X,y,Xt,yt,method,clf,seed,ratio))
                    except (ValueError,RuntimeError) as error:
                        row['error']=str(error)
                    rows.append(row)
                    study.update(records=rows,protocol='CASB source quotas .5/.3/.2; ROS duplicates selected source; same existing outer/inner protocol',
                        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        input_sha256=hashes)
                    nb.metadata[metadata_key]=study
                    nbformat.write(nb,path)
        assert len(rows)==20
        if method=='adasyn':
            study.update(complete=True,protocol='Official imbalanced-learn ADASYN; same training-only three-ratio search; errors retained without fallback')
            nb.metadata[metadata_key]=study
            nbformat.write(nb,path)
            print('ADASYN complete:',dataset,'errors:',sum('error' in r for r in rows),flush=True)
            continue
        f=pd.DataFrame(rows);f['composite']=(f.gmean+f.f1)/2
        metric=['gmean','f1','precision','recall','ba','composite']
        table=f.groupby('classifier')[metric].mean()
        text='# CASB generator extension: ROS\n\nActual completed fits: 20 (SVM/KNN, five folds, two seeds). Three training-only ratio candidates per fit. Raw evidence is in `metadata.generator_extension.records`.\n\n'
        text+='CASB selects source samples with its original 50/30/20 quotas; ROS duplicates them. This swaps the generator, rather than running two balancing operations in sequence. It is inspired by the modular experiment design of the uploaded UGO paper; it does not reproduce UGO. MCSMOTE is excluded at user request.\n\n'
        text+='| Classifier | '+' | '.join(metric)+' |\n|---|'+'---:|'*len(metric)+'\n'
        for clf,r in table.iterrows(): text+='| '+clf+' | '+' | '.join(f'{100*r[m]:.3f}' for m in metric)+' |\n'
        text+='\nValues are percentages. To resume/reproduce the extension through the shared implementation, use `generator_extension_study()`; completed records are reused.\n'
        nb.cells=[c for c in nb.cells if not (c.cell_type=='markdown' and c.source.startswith('# CASB generator extension: ROS'))]
        nb.cells.append(nbformat.v4.new_markdown_cell(text))
        study['complete']=True;nb.metadata[metadata_key]=study
        nbformat.write(nb,path)
        print('CASB-ROS complete:',dataset,flush=True)


CURRENT_METHODS = ['original','smote','bsmote','ros','adasyn','casb','casb_relaxed','casb_edited','casb_ros']


def current_dataset_records(path, dataset, rerun=False):
    """Resume/retrain current methods with notebook-only checkpoints."""
    import nbformat
    path=Path(path);nb=nbformat.read(path,as_version=4)
    prior=[]
    if not rerun:
        for key in ['benchmark','generator_extension','adasyn_extension','current_comparison']:
            prior.extend(nb.metadata.get(key,{}).get('records',[]))
    cached={(r['seed'],r['fold'],r['classifier'],r['method']):r for r in prior if r['method'] in CURRENT_METHODS}
    rows=[];hashes={}
    with threadpool_limits(limits=1):
        for seed in [42,137]:
            for fold,X,y,Xt,yt,paths in data_splits(Path('All'),dataset,5,seed,allow_nominal=True):
                for source in paths: hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
                for clf in ['svm','knn']:
                    for method in CURRENT_METHODS:
                        key=(seed,fold,clf,method)
                        row=cached.get(key)
                        if row is None:
                            row=dict(dataset=dataset,seed=seed,fold=fold,classifier=clf,method=method)
                            try:
                                ratio,validation=choose_ratio(X,y,method,clf,seed,[.5,.75,1.],'gmean_f1')
                                row.update(ratio=ratio,validation=validation)
                                row.update(evaluate_fit(X,y,Xt,yt,method,clf,seed,ratio))
                            except (ValueError,RuntimeError) as error: row['error']=str(error)
                            row['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                        rows.append(row)
                        nb.metadata['current_comparison']=dict(records=rows,methods=CURRENT_METHODS,
                            input_sha256=hashes,complete=len(rows)==180,
                            protocol='5 supplied folds x 2 seeds; SVM/KNN; train-only ratio .5/.75/1; gmean_f1')
                        nbformat.write(nb,path)
    return rows



def classifier_expansion(report=Path('progress.md'), workers=3):
    """Resume both classifier studies; persist all evidence only in the report."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import pandas as pd
    tag='classifier_expansion_v1'
    root=Path(__file__).resolve().parent
    report=root/report
    methods=CURRENT_METHODS
    clfs=['rf','lr','gnb']
    existing=[]
    for line in report.read_text(encoding='utf8').splitlines():
        if line.startswith('{"expansion":'):
            obj=json.loads(line)
            if obj['expansion']==tag: existing.append(obj)
    key=lambda r:(r['study'],r['dataset'],r['seed'],r['fold'],r['classifier'],r['method'])
    done={key(r) for r in existing}
    rows=list(existing)
    assert len(done)==len(rows)
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if rows: assert all(r['code_sha256']==digest for r in rows),'Implementation changed; do not mix runs'
    notebooks=sorted(root.glob('[0-9][0-9]_*_CASB.ipynb'))
    assert len(notebooks)==44
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in notebooks}
    append(report,'\n```\n</details>\n\n## Classifier expansion: RF, LR, GaussianNB\n\n'
        'Both 44-dataset benchmark and 20-seed overlap simulation. Fixed existing sampler settings. '
        'Five supplied folds and seeds 42/137 for benchmark; separations 5 to 0 for overlap. '
        'Train-only 25% holdout selects ratio .5/.75/1 by (G-mean + F1)/2; lower ratio breaks ties. '
        'RF: 100 trees, min leaf 2; LR: default C=1, max_iter=3000; GaussianNB: var_smoothing=1e-9. '
        'No classifier tuning, no test-set selection. All failures retained. No notebook or figure changes.\n\n'
        'Code SHA256: '+digest+'\n\n<details><summary>Resumable raw classifier evidence</summary>\n\n```jsonl\n')
    def evaluate_task(study,dataset,seed,fold,X,y,T,yt,provenance):
        result=[]
        for clf in clfs:
            for method in methods:
                r=dict(expansion=tag,study=study,dataset=dataset,seed=seed,fold=fold,classifier=clf,method=method,code_sha256=digest)
                if key(r) in done: continue
                r.update(train_counts=np.bincount(y).tolist(),test_counts=np.bincount(yt).tolist(),provenance=provenance)
                try:
                    ratio,trials=choose_ratio(X,y,method,clf,seed,[.5,.75,1.],'gmean_f1')
                    r.update(ratio=ratio,validation=trials)
                    r.update(evaluate_fit(X,y,T,yt,method,clf,seed,ratio))
                    tn,fp,fn,tp=r['confusion'];C=np.array([[tn,fp],[fn,tp]],float)
                    rr=C.sum(1);cc=C.sum(0);dd=np.diag(C)
                    r['iam']=float(np.mean((dd-np.maximum(rr-dd,cc-dd))/np.maximum(rr,cc)))
                    den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
                    r['mcc']=float((tp*tn-fp*fn)/den) if den else 0.
                except (ValueError,RuntimeError) as e:r['error']=type(e).__name__+': '+str(e)
                result.append(r)
        return result
    def tasks():
        for nbpath in notebooks:
            meta=json.loads(nbpath.read_text(encoding='utf8'))['metadata']
            dataset=meta['benchmark']['dataset']
            for seed in [42,137]:
                for fold,X,y,T,yt,paths in data_splits(root/'output'/'All',dataset,5,seed,allow_nominal=True):
                    provenance={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                    for p in paths:
                        oldkey=str(Path('All')/dataset/p.name)
                        assert meta['current_comparison']['input_sha256'][oldkey]==provenance[str(p.relative_to(root))]
                    yield ('benchmark',dataset,seed,fold,X,y,T,yt,provenance)
        for seed in range(20260,20280):
            tr_rng,te_rng=[np.random.default_rng(v) for v in np.random.SeedSequence(seed).spawn(2)]
            tr0=tr_rng.normal(size=(400,2));tr1=tr_rng.normal(size=(40,2))
            te0=te_rng.normal(size=(1000,2));te1=te_rng.normal(size=(100,2))
            y=np.r_[np.zeros(400,int),np.ones(40,int)];yt=np.r_[np.zeros(1000,int),np.ones(100,int)]
            for d in np.arange(5,-.01,-.5):
                X=np.vstack([tr0,tr1+[d,0]]);T=np.vstack([te0,te1+[d,0]])
                provenance=dict(train=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),test=hashlib.sha256(T.tobytes()+yt.tobytes()).hexdigest())
                yield ('overlap',f'd={d:g}',seed,0,X,y,T,yt,provenance)
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded submissions keep large dataset matrices out of the pending queue.
        from itertools import islice
        iterator=iter(tasks())
        while True:
            batch=list(islice(iterator,workers*2))
            if not batch:break
            for future in as_completed([pool.submit(evaluate_task,*t) for t in batch]):
                part=future.result()
                for r in part:append(report,json.dumps(r,separators=(',',':'))+'\n')
                rows.extend(part)
                if part: print(f"{len(rows)}/17820 saved; {part[-1]['study']} {part[-1]['dataset']}",flush=True)
    assert len(rows)==17820 and len({key(r) for r in rows})==17820
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    append(report,'```\n</details>\n\n### Completed classifier expansion\n\n')
    frame=pd.DataFrame(rows);metrics=['gmean','f1','precision','recall','mcc','iam']
    core=['smote','bsmote','ros','adasyn'];family=['casb','casb_relaxed','casb_edited','casb_ros']
    for study in ['benchmark','overlap']:
        f=frame[frame.study==study].copy()
        failed=f[f.error.notna()] if 'error' in f else f.iloc[:0]
        append(report,f'\n#### {study}\n\nAttempts: {len(f)}; failures: {len(failed)}.\n\n')
        for clf in clfs:
            z=f[f.classifier==clf];bad=set(failed.loc[failed.classifier==clf,'dataset'])
            z=z[~z.dataset.isin(bad)]
            means=z.groupby(['dataset','method'])[metrics].mean()
            avg=means.groupby('method').mean()
            append(report,f'\n**{clf}: {z.dataset.nunique()} complete matched datasets/separations**; excluded: {sorted(bad)}\n\n'+(avg*100).round(3).to_markdown()+'\n\n')
            summary=[]
            for metric in metrics:
                w=means[metric].unstack('method');delta=w[family].max(axis=1)-w[core].max(axis=1)
                ours=avg.loc[family,metric];base=avg.loc[core,metric]
                summary.append(dict(metric=metric,best_family=ours.idxmax(),family_score=100*ours.max(),best_baseline=base.idxmax(),baseline_score=100*base.max(),delta_pp=100*(ours.max()-base.max()),wins=int((delta>1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),losses=int((delta < -1e-12).sum())))
            append(report,pd.DataFrame(summary).round(3).to_markdown(index=False)+'\n\n')
        append(report,'Failure counts:\n\n'+failed.groupby(['classifier','method','dataset']).size().to_markdown()+'\n\n' if len(failed) else 'No failed fits.\n\n')
    append(report,'Family maxima are retrospective per-metric summaries, not a deployable selection method. '
        'Dataset win counts compare the best family variant against the strongest of all four baselines on each dataset. '
        'Means weight datasets/separations equally. No composite score or statistical significance is inferred. '
        'Notebooks and figures remain unchanged pending review.\n')
    print('Both studies completed; final tables appended to progress.md',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("All"))
    parser.add_argument("--datasets", nargs="+", default=["pima", "haberman", "glass1", "yeast3"])
    parser.add_argument("--methods", nargs="+", default=REVISION_METHODS)
    parser.add_argument("--classifiers", nargs="+", choices=["lr", "rf", "svm", "knn"], default=["svm", "knn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0], help="Multiple ratios enable identical train-only holdout selection for each method")
    parser.add_argument("--target", help="Target column for a standalone CSV")
    parser.add_argument("--report", type=Path, default=Path("progress.md"))
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--analyze-run", help="Append paired analysis of a saved run tag without retraining")
    parser.add_argument("--candidate", default="margin")
    parser.add_argument("--verify-run", help="Refit first saved setting and verify metrics")
    parser.add_argument('--create-notebooks', action='store_true')
    parser.add_argument('--execute-notebooks', action='store_true')
    parser.add_argument('--summarize-casb', action='store_true')
    parser.add_argument('--selection',choices=['ba','gmean_f1'],default='ba',help='Training-only ratio selection objective; validation ba field stores this objective')
    parser.add_argument('--revision-summary',action='store_true')
    parser.add_argument('--revision-notebooks',action='store_true')
    parser.add_argument('--remaining-study',action='store_true')
    parser.add_argument('--full-study-summary',action='store_true')
    parser.add_argument('--dataset-notebooks',action='store_true')
    parser.add_argument('--prepare-dataset-notebooks',action='store_true')
    parser.add_argument('--generator-extension',action='store_true')
    parser.add_argument('--adasyn-extension',action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.adasyn_extension: return generator_extension_study('adasyn')
        if args.generator_extension: return generator_extension_study()
        if args.dataset_notebooks or args.prepare_dataset_notebooks: return dataset_notebook_study(args.root,execute=args.dataset_notebooks)
        if args.inventory: return inventory(args.root)
        if args.self_test: return self_test()
        if args.analyze_run: return analyze_run(args.report,args.analyze_run,args.candidate)
        if args.verify_run: return verify_saved(args.report,args.verify_run,args.root,args.candidate,args.folds)
        if args.create_notebooks: return create_notebooks()
        if args.execute_notebooks: return execute_notebooks()
        if args.summarize_casb: return summarize_casb(args.report)
        if args.revision_summary: return revision_summary(args.report)
        if args.revision_notebooks: return revision_notebooks()
        if args.remaining_study: return remaining_study(args.root,args.report)
        if args.full_study_summary: return full_study_summary(args.root,args.report)
        if any(not 0 < r <= 1 for r in args.ratios): parser.error("Ratios must lie in (0,1]")
        config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        config["python"] = platform.python_version()
        config["versions"] = {p:importlib.metadata.version(p) for p in ("numpy","scipy","scikit-learn","imbalanced-learn")}
        config["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        append(args.report, f"\n## Run: {args.tag}\n\nStarted: {datetime.now().isoformat(timespec='seconds')} (host local time)\n\n```json\n{json.dumps(config,indent=2)}\n```\n\n")
        records, hashes = [], {}
        append(args.report, "<details><summary>Raw fold records (JSON lines, reproducible evidence)</summary>\n\n```jsonl\n")
        for dataset in args.datasets:
            for seed in args.seeds:
                for fold, X, y, Xt, yt, paths in data_splits(args.root, dataset, args.folds, seed, args.target):
                    for path in paths:
                        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    overlap = len(set(map(tuple,np.nan_to_num(X,nan=-1e99))) & set(map(tuple,np.nan_to_num(Xt,nan=-1e99))))
                    for clf in args.classifiers:
                        for method in args.methods:
                            row = dict(dataset=dataset, seed=seed, fold=fold, classifier=clf, method=method,
                                train_counts=np.bincount(y).tolist(), test_counts=np.bincount(yt).tolist(),
                                train_test_shared_feature_vectors=overlap)
                            try:
                                ratio, validation = choose_ratio(X,y,method,clf,seed,args.ratios,args.selection)
                                row.update(ratio=ratio, validation=validation)
                                row.update(evaluate_fit(X,y,Xt,yt,method,clf,seed,ratio))
                            except ValueError as error:
                                row["error"] = str(error)
                            records.append(row)
                            append(args.report, json.dumps(row,separators=(',',':'))+"\n")
                    print(f"{args.tag}: {dataset} seed={seed} fold={fold}/{args.folds} complete",flush=True)
        append(args.report, "```\n</details>\n\n<details><summary>Input hashes</summary>\n\n```json\n"+json.dumps(hashes,indent=2)+"\n```\n</details>\n")
        summary = summarize(records)
        append(args.report, summary)
        print(summary[summary.index("### Equal-weight"):], flush=True)


if __name__ == "__main__":
    main()
