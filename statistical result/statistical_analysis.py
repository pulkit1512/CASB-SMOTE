"""Reproduce CASB statistics from saved records only; never import training code.

PowerShell: python -B 'statistical result/statistical_analysis.py'
All generated artifacts stay beside this file. Dataset scores require all ten
supplied-fold/seed evaluations. Primary tests use datasets as paired units.
"""
from pathlib import Path
from itertools import combinations
from io import BytesIO
import base64
import hashlib
import importlib.metadata
import json
import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata, friedmanchisquare, binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
ALPHA, TOL = .05, 1e-12
METRICS = ['gmean', 'f1', 'precision', 'recall']
EXTRA = ['ba', 'specificity', 'auc', 'ap']
FAMILY = ['casb', 'casb_relaxed', 'casb_edited', 'casb_ros']
CORE = ['original', 'smote', 'bsmote', 'ros', 'adasyn']
CLFS = ['svm', 'knn', 'rf', 'lr', 'gnb']
SOURCES = ['abalone', 'ecoli', 'glass', 'haberman', 'iris', 'new-thyroid',
           'page-blocks', 'pima', 'segment', 'shuttle', 'vehicle', 'vowel', 'wisconsin', 'yeast']
NAMES = {'original':'Original','smote':'SMOTE','bsmote':'Borderline-SMOTE',
         'ros':'ROS','rus':'RUS','adasyn':'ADASYN','smoteenn':'SMOTE-ENN',
         'casb':'CASB','casb_relaxed':'CASB-relaxed','casb_edited':'CASB-edited','casb_ros':'CASB-ROS'}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_family(dataset):
    found = [s for s in SOURCES if dataset.startswith(s)]
    assert len(found) == 1, (dataset, found)
    return found[0]


def holm(values):
    p = np.asarray(values, float)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    order = np.argsort(p, kind='stable')
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    return result


def paired(a, b):
    # Scores/means remain full precision; only near-zero test differences are
    # set to zero, and rank differences rounded to avoid numerical pseudo-ties.
    delta = np.asarray(a, float) - np.asarray(b, float)
    d = np.round(delta, 12)
    d[np.abs(delta) <= TOL] = 0
    nz = d[d != 0]
    ranks = rankdata(np.abs(nz))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        p = float(wilcoxon(d, zero_method='wilcox', correction=False,
                          alternative='two-sided', method='approx').pvalue) if len(nz) else 1.
    w, l = int((d > 0).sum()), int((d < 0).sum())
    effect = float((ranks*np.sign(nz)).sum()/ranks.sum()) if len(nz) else 0.
    return dict(n=len(d), casb_mean=float(np.mean(a)), baseline_mean=float(np.mean(b)),
                mean_difference=float(delta.mean()), median_difference=float(np.median(delta)),
                p_raw=p, rank_biserial=effect, wins=w, ties=int((d == 0).sum()), losses=l,
                sign_p_raw=float(binomtest(w, w+l).pvalue) if w+l else 1.)


def extract():
    paths = sorted(ROOT.glob('[0-9][0-9]_*_CASB.ipynb'))
    assert len(paths) == 44
    protected = paths+[ROOT/'binary_oversampling.py', ROOT/'progress.md']
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in protected}
    raw, audit, seen = [], [], {}
    original_count = original_failures = 0
    for p in paths:
        n = json.loads(p.read_text(encoding='utf8'))
        meta = n['metadata']; current = meta['current_comparison']
        dataset = meta['benchmark']['dataset']
        executed = all(c.get('execution_count') is not None and
                       not any(o.get('output_type') == 'error' for o in c.get('outputs', []))
                       for c in n['cells'] if c['cell_type'] == 'code')
        assert executed, p
        for src, h in current['input_sha256'].items():
            q = ROOT/src
            if not q.exists(): q = ROOT/'output'/src
            assert digest(q) == h, q
            hashes[str(q.relative_to(ROOT))] = h
        chosen = list(current['records'])
        svm_knn = [r for r in chosen if r['classifier'] in ['svm','knn']]
        original_count += len(svm_knn)
        original_failures += sum('error' in r for r in svm_knn)
        # Legacy RUS/ENN are compatible only if every shared record agrees.
        lookup = {(r['seed'],r['fold'],r['classifier'],r['method']):r for r in chosen}
        old = meta['benchmark']['records']
        overlap = 0
        for r in old:
            key = (r['seed'],r['fold'],r['classifier'],r['method'])
            if key in lookup:
                overlap += 1
                for k in ['ratio','confusion','gmean','f1','precision','recall','error']:
                    assert r.get(k) == lookup[key].get(k), (p,key,k)
        assert overlap >= 140
        chosen += [r for r in old if r['method'] in ['rus','smoteenn']]
        audit.append(dict(notebook=p.name,dataset=dataset,executed=executed,
                          current_records=len(current['records']),legacy_compatible_shared=overlap,
                          legacy_rus_enn=sum(r['method'] in ['rus','smoteenn'] for r in chosen)))
        for r in chosen:
            k = (dataset,r['classifier'],r['method'],r['seed'],r['fold'])
            assert k not in seen, k
            seen[k] = True
            raw.append(dict(r,dataset=dataset,origin=p.name,protocol='supplied5_seed42_137_inner_gmean_f1'))
    # Additional classifiers may reside in progress.md rather than notebook metadata.
    for line in (ROOT/'progress.md').read_text(encoding='utf8').splitlines():
        if not line.startswith('{"expansion":'): continue
        r = json.loads(line)
        if r['expansion'] != 'classifier_expansion_v1' or r['study'] != 'benchmark': continue
        k = (r['dataset'],r['classifier'],r['method'],r['seed'],r['fold'])
        if k in seen: continue
        seen[k] = True
        raw.append(dict(r,origin='progress.md classifier_expansion_v1',protocol='supplied5_seed42_137_inner_gmean_f1'))
    assert original_count == 7920 and original_failures == 26
    expected = {(s,f) for s in [42,137] for f in range(1,6)}
    f = pd.DataFrame(raw)
    if 'error' not in f: f['error'] = None
    for r in raw:
        if 'error' in r: continue
        tn,fp,fn,tp = r['confusion']
        assert sum(r['confusion']) > 0
        computed = dict(gmean=math.sqrt(tp/max(1,tp+fn)*tn/max(1,tn+fp)),
                        f1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),
                        recall=tp/max(1,tp+fn),specificity=tn/max(1,tn+fp),
                        ba=.5*(tp/max(1,tp+fn)+tn/max(1,tn+fp)))
        for m,v in computed.items(): assert abs(r[m]-v) <= 1e-12, (r['dataset'],m)
        trials = [t for t in r.get('validation',[]) if 'ba' in t]
        if trials: assert r['ratio'] == max(trials,key=lambda t:(t['ba'],-t['ratio']))['ratio']
    scores = []
    for (d,c,m), g in f.groupby(['dataset','classifier','method']):
        complete = len(g) == 10 and set(zip(g.seed,g.fold)) == expected and g.error.isna().all()
        z = dict(dataset=d,dataset_family=source_family(d),classifier=c,method=m,
                 n_expected=10,n_attempted=len(g),n_success=int(g.error.isna().sum()),
                 n_failed=int(g.error.notna().sum()),status='complete' if complete else 'incomplete',
                 protocol='supplied5_seed42_137_inner_gmean_f1',seeds='42;137',folds='1;2;3;4;5',
                 origins=';'.join(sorted(set(g.origin))))
        z.update({metric:float(g[metric].mean()) if complete and g[metric].notna().all() else np.nan for metric in METRICS+EXTRA})
        scores.append(z)
    scores = pd.DataFrame(scores)
    assert len(f) == 21560, len(f)  # 5*44*9*10 plus 2*44*2*10
    return f, scores, pd.DataFrame(audit), hashes


def make_tables(scores):
    alltests, friedman, averages, family_rows, individuals = [], [], [], [], []
    posthoc = []
    for clf in CLFS:
        q = scores[(scores.classifier == clf) & (scores.status == 'complete')]
        baselines = CORE+(['rus','smoteenn'] if clf in ['svm','knn'] else [])
        methods = baselines+FAMILY
        complete = q.pivot(index='dataset',columns='method',values='gmean').reindex(columns=methods).dropna().index
        for population in ['strict_all_methods','available_method']:
            z = q[q.dataset.isin(complete)] if population == 'strict_all_methods' else q
            for m,g in z.groupby('method'):
                averages.append(dict(classifier=clf,population=population,method=m,n_datasets=len(g),
                                     **{k:float(g[k].mean()) for k in METRICS+EXTRA}))
        # Primary family: all fixed CASB x available compatible baselines x four metrics.
        tests = []
        representative = scores[scores.classifier == clf].sort_values('dataset').groupby('dataset_family').dataset.first().to_dict()
        for analysis in ['dataset_pairs','source_family_mean','one_task_per_family']:
            start = len(tests)
            for met in METRICS:
                w = q.pivot(index='dataset',columns='method',values=met)
                for a in FAMILY:
                    for b in baselines:
                        v = w[[a,b]].dropna()
                        if analysis == 'source_family_mean':
                            v = v.assign(source=[source_family(d) for d in v.index]).groupby('source')[[a,b]].mean()
                        elif analysis == 'one_task_per_family':
                            v = v.loc[v.index.intersection(list(representative.values()))]
                        result = paired(v[a],v[b])
                        tests.append(dict(analysis=analysis,classifier=clf,metric=met,method=a,baseline=b,
                                          paired_units=';'.join(v.index),**result))
            sub = tests[start:]
            adjusted = holm([r['p_raw'] for r in sub]); signs = holm([r['sign_p_raw'] for r in sub])
            for r,p,sp in zip(sub,adjusted,signs):
                r.update(p_holm=float(p),sign_p_holm=float(sp),correction_n=len(sub),
                         significant=bool(p<ALPHA),direction='CASB higher' if p<ALPHA and r['rank_biserial']>0 else 'CASB lower' if p<ALPHA else 'not significant')
        for r in tests:
            r['p_holm_core80'] = np.nan
        # Reproduce the original five-baseline scope separately, never pick smaller p.
        for analysis in ['dataset_pairs','source_family_mean','one_task_per_family']:
            sub = [r for r in tests if r['analysis']==analysis and r['baseline'] in CORE]
            assert len(sub)==80
            for r,p in zip(sub,holm([r['p_raw'] for r in sub])): r['p_holm_core80']=float(p)
        alltests.extend(tests)
        individuals.extend([dict(r) for r in tests if r['analysis']=='dataset_pairs'])
        for met in METRICS:
            w = q[q.dataset.isin(complete)].pivot(index='dataset',columns='method',values=met)[methods]
            fr = friedmanchisquare(*[w[m].to_numpy() for m in methods])
            ranks = w.rank(axis=1,ascending=False,method='average').mean()
            friedman.append(dict(classifier=clf,metric=met,n_datasets=len(w),n_methods=len(methods),
                                 statistic=float(fr.statistic),p_raw=float(fr.pvalue),
                                 mean_ranks_json=json.dumps(ranks.to_dict(),sort_keys=True),
                                 dataset_units=';'.join(w.index),methods=';'.join(methods)))
            # All-pairs Wilcoxon-Holm post hoc within each classifier, four metrics.
            for a,b in combinations(methods,2):
                posthoc.append(dict(analysis='friedman_posthoc',classifier=clf,metric=met,
                                    method=a,baseline=b,**paired(w[a],w[b])))
            # Explicitly label baseline populations. all_available includes original.
            sets={'all_available':baselines,'original_four_resamplers':['smote','bsmote','ros','adasyn']}
            for label,bs in sets.items():
                valid = q.pivot(index='dataset',columns='method',values=met)[FAMILY+bs].dropna()
                for d,values in valid.iterrows():
                    ours=values[FAMILY];other=values[bs];delta=float(ours.max()-other.max())
                    family_rows.append(dict(row_type='dataset',population=label,classifier=clf,metric=met,dataset=d,
                        dataset_family=source_family(d),best_variant=ours.idxmax(),best_baseline=other.idxmax(),
                        family_score=float(ours.max()),baseline_score=float(other.max()),winning_margin=delta,
                        result='win' if delta>TOL else 'loss' if delta < -TOL else 'tie'))
            for a in FAMILY:
                delta=w[a]-w[baselines].max(axis=1)
                individuals.append(dict(analysis='variant_vs_strongest_baseline',classifier=clf,metric=met,method=a,
                    baseline='strongest_available',n=len(w),wins=int((delta>TOL).sum()),ties=int((delta.abs()<=TOL).sum()),
                    losses=int((delta < -TOL).sum()),mean_difference=float(delta.mean()),p_raw=np.nan,p_holm=np.nan))
        # Joint means: same variant must beat each metric's strongest baseline.
        gm=q.pivot(index='dataset',columns='method',values='gmean')[methods].dropna()
        ff=q.pivot(index='dataset',columns='method',values='f1')[methods].dropna()
        for label,bs in [('all_available',baselines),('original_four_resamplers',['smote','bsmote','ros','adasyn'])]:
            for d in complete:
                margins=np.minimum(gm.loc[d,FAMILY]-gm.loc[d,bs].max(),ff.loc[d,FAMILY]-ff.loc[d,bs].max())
                best=margins.idxmax();delta=float(margins.max())
                family_rows.append(dict(row_type='dataset',population=label,classifier=clf,metric='joint_gmean_f1_same_variant',
                    dataset=d,dataset_family=source_family(d),best_variant=best,winning_margin=delta,
                    result='win' if delta>TOL else 'loss' if delta < -TOL else 'tie'))
    fr=pd.DataFrame(friedman)
    for c,ix in fr.groupby('classifier').groups.items():fr.loc[ix,'p_holm_four_metrics']=holm(fr.loc[ix,'p_raw'])
    fr['significant']=fr.p_holm_four_metrics<ALPHA
    ph=pd.DataFrame(posthoc)
    for c,ix in ph.groupby('classifier').groups.items():
        ph.loc[ix,'p_holm']=holm(ph.loc[ix,'p_raw']);ph.loc[ix,'correction_n']=len(ix)
    ph['significant']=ph.p_holm<ALPHA
    ph['direction']=np.where(~ph.significant,'not significant',np.where(ph.rank_biserial>0,'first method higher','first method lower'))
    ph['omnibus_significant']=[bool(fr[(fr.classifier==r.classifier)&(fr.metric==r.metric)].significant.iloc[0]) for r in ph.itertuples()]
    pairs=pd.concat([pd.DataFrame(alltests),ph],ignore_index=True)
    fam=pd.DataFrame(family_rows)
    summary=[]
    for key,g in fam.groupby(['population','classifier','metric']):
        count=g.result.value_counts();n=len(g)
        summary.append(dict(row_type='summary',population=key[0],classifier=key[1],metric=key[2],
                            n=n,wins=int(count.get('win',0)),ties=int(count.get('tie',0)),losses=int(count.get('loss',0)),
                            win_percent=100*count.get('win',0)/n,majority=bool(count.get('win',0)>n/2)))
    fam=pd.concat([fam,pd.DataFrame(summary)],ignore_index=True)
    return pd.DataFrame(averages),pairs,fr,pd.DataFrame(individuals),fam


def plot_figures(avg,pairs,fr,fam):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    figures=[]
    def keep(fig,title):
        fig.suptitle(title,fontsize=15);fig.tight_layout(rect=(0,0,1,.95));figures.append((title,fig))
    primary=avg[(avg.population=='strict_all_methods') & (avg.classifier!='gnb')]
    for clf in CLFS[:4]:
        g=primary[primary.classifier==clf].set_index('method');methods=list(g.index)
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,m in zip(axes.flat,METRICS):
            ax.bar(range(len(methods)),100*g[m],color=['#8055a0' if x in FAMILY else '#2679a5' for x in methods])
            ax.set_xticks(range(len(methods)),[NAMES[x] for x in methods],rotation=60,ha='right',fontsize=8)
            ax.set_ylabel(m+' (%)');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.15)
        keep(fig,clf.upper()+': equal-dataset averages, strict matched population')
    for met,title in [('gmean','G-mean averages'),('f1','F1 averages')]:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,c in zip(axes.flat,CLFS[:4]):
            q=primary[primary.classifier==c].sort_values(met);ax.barh([NAMES[m] for m in q.method],100*q[met],color=['#8055a0' if m in FAMILY else '#2679a5' for m in q.method]);ax.set(xlim=(0,100),title=c.upper(),xlabel=met+' (%)')
        keep(fig,title)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    all_methods=list(primary.method.unique())
    palette=plt.get_cmap('tab20')
    for ax,c in zip(axes.flat,CLFS[:4]):
        q=primary[primary.classifier==c]
        for r in q.itertuples():ax.scatter(100*r.recall,100*r.precision,color=palette(all_methods.index(r.method)),marker='s' if r.method in FAMILY else 'o',label=NAMES[r.method],s=45,edgecolors='white',linewidths=.4)
        ax.set(title=c.upper(),xlabel='Recall (%)',ylabel='Precision (%)');ax.margins(.2);ax.grid(alpha=.15)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,-.035),ncol=6,fontsize=8,frameon=False)
    keep(fig,'Precision-recall trade-offs')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,met in zip(axes.flat,METRICS):
        for m in FAMILY:
            q=primary[primary.method==m].set_index('classifier').reindex(CLFS[:4]);ax.plot(CLFS[:4],100*q[met],marker='o',label=NAMES[m])
        ax.set(title=met,ylabel='Score (%)');ax.legend(fontsize=8);ax.grid(alpha=.15)
    keep(fig,'CASB variants by classifier')
    f=fam[(fam.row_type=='summary')&(fam.population=='all_available')&(fam.metric.isin(METRICS))&(fam.classifier!='gnb')]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,c in zip(axes.flat,CLFS[:4]):
        q=f[f.classifier==c].set_index('metric').reindex(METRICS);bottom=np.zeros(4)
        for col,color in [('wins','#208b63'),('ties','#bbbbbb'),('losses','#c65757')]:ax.bar(METRICS,q[col],bottom=bottom,label=col,color=color);bottom+=q[col].to_numpy()
        ax.set(title=c.upper(),ylabel='Datasets');ax.legend(fontsize=8)
    keep(fig,'Retrospective family win / tie / loss against all available baselines')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,c in zip(axes.flat,CLFS[:4]):
        for met in METRICS:
            r=fr[(fr.classifier==c)&(fr.metric==met)].iloc[0];ranks=json.loads(r.mean_ranks_json);ax.plot([NAMES[x] for x in ranks],list(ranks.values()),marker='o',label=met)
        ax.set_title(c.upper());ax.set_ylabel('Mean rank (lower is better)');ax.tick_params(axis='x',rotation=65,labelsize=7);ax.legend(fontsize=7)
    keep(fig,'Friedman mean ranks (no critical-distance claim)')
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for ax,c in zip(axes.flat,CLFS[:4]):
        q=pairs[(pairs.analysis=='dataset_pairs')&(pairs.classifier==c)];cols=sorted(set(q.baseline));matrix=np.zeros((16,len(cols)));labels=[]
        for i,(met,m) in enumerate((met,m) for met in METRICS for m in FAMILY):
            labels.append(met+' / '+NAMES[m])
            for j,b in enumerate(cols):
                r=q[(q.metric==met)&(q.method==m)&(q.baseline==b)].iloc[0]
                matrix[i,j]=r.rank_biserial if r.p_holm<ALPHA else 0
                ax.text(j,i,'+' if matrix[i,j]>0 else '-' if matrix[i,j]<0 else '.',ha='center',va='center',fontsize=8)
        ax.imshow(matrix,cmap='RdBu',vmin=-1,vmax=1,aspect='auto');ax.set_xticks(range(len(cols)),[NAMES[b] for b in cols],rotation=45,ha='right',fontsize=8);ax.set_yticks(range(16),labels,fontsize=7);ax.set_title(c.upper())
    keep(fig,'Wilcoxon-Holm: + significant CASB benefit, - significant harm, . non-significant')
    fig,ax=plt.subplots(figsize=(11,5))
    for i,met in enumerate(METRICS):
        q=f[f.metric==met].set_index('classifier').reindex(CLFS[:4]);ax.bar(np.arange(4)+(i-1.5)*.18,q.win_percent,width=.18,label=met)
    ax.axhline(50,color='#444444',ls='--',lw=1);ax.set_xticks(range(4),[c.upper() for c in CLFS[:4]]);ax.set_ylabel('Family wins (%)');ax.set_ylim(0,100);ax.legend()
    keep(fig,'Family win rates against all available baselines (retrospective)')
    return figures


def markdown_report(raw,scores,audit,avg,pairs,fr,fam,hashes):
    lines=['# CASB statistical analysis and paper results','',
           '## A. Statistical methodology','',
           'Analysis uses saved evaluation records only; no model was trained. Each method/classifier/dataset score averages all ten supplied fold/seed evaluations. Incomplete groups are withheld. Primary tests keep classifiers separate. G-mean and minority F1 are primary metrics; precision and recall are secondary. Scores retain full precision; presentation tables show percentages. Numerical tie tolerance is 1e-12.', '',
           'Two-sided Wilcoxon signed-rank tests compare each fixed CASB variant against every compatible baseline. Zeros are discarded; differences are rounded to 12 decimals for ranks, and the tie-corrected asymptotic p-value is used. The null concerns a paired-difference distribution symmetric about zero; asymmetry and related datasets limit interpretation. Rank-biserial effects, raw and adjusted p-values, exact sign-test sensitivity, paired units, and win/tie/loss counts are saved.', '',
           'Holm families are fixed within classifier and analysis population across all four metrics: 4 variants x 7 baselines x 4 metrics = 112 for SVM/KNN; 4 x 5 x 4 = 80 for RF/LR/GaussianNB. The baseline set includes no resampling. RUS and SMOTE-ENN are unavailable under the RF/LR/GaussianNB protocol. The original five-baseline 80-test correction is also supplied in p_holm_core80, as a separate scope rather than a favourable replacement. No primary correction pools classifiers. Sensitivity populations have their own identical-size correction families.', '',
           'Friedman tests use the same strict complete datasets and method set within each classifier, with Holm correction over four metrics. Significant omnibus tests establish differences somewhere among methods. Post hoc all-pairs Wilcoxon-Holm results use the same complete cohort, with a separate family covering all method pairs and four metrics for each classifier. Primary fixed-variant tests use all valid pairs, so comparisons without ADASYN can use 44 datasets while ADASYN uses 40.', '',
           '## Extraction and protocol audit','',
           f'All {len(audit)} original dataset notebooks were read and verified executed. Original SVM/KNN current records: 7,920, including 26 failed evaluations. Expanded current RF/LR/GaussianNB records and compatible RUS/SMOTE-ENN evidence yield {len(raw):,} unique saved evaluations. Successful: {raw.error.isna().sum():,}; failed: {raw.error.notna().sum():,}. All failures remain visible in the executed notebook.', '',
           'Training-only preprocessing and a 25% stratified inner holdout selected ratios 0.5/0.75/1.0 by (G-mean + F1)/2, ties preferring the lower ratio. Seeds were 42/137, using the supplied five train/test folds. The two seeds reuse the same supplied folds and are averaged, never counted as independent datasets. Legacy RUS/ENN compatibility was checked by equality of every shared current/legacy record (scores, confusion matrices, ratios and errors). Historical protocols outside these metadata sections were not merged. Abalone uses training-fitted one-hot inputs; interpolation may yield fractional dummy values.', '',
           'RF and LR were selected for presentation after examining results; GaussianNB is reported separately. None of these are independent confirmatory evaluations. Overlap-simulation records are excluded from the real-dataset statistical population.','']
    failures=raw[raw.error.notna()].groupby(['classifier','method','dataset']).size().rename('failed').reset_index()
    lines += [failures.to_markdown(index=False),'','## B. Overall performance','']
    strict=avg[avg.population=='strict_all_methods']
    for c in CLFS:
        q=strict[strict.classifier==c].set_index('method');formatted=(100*q[METRICS]).map(lambda x:f'{x:.3f}')
        for m in METRICS:
            mx=q[m].max()
            for method in q.index:
                if abs(q.loc[method,m]-mx)<=TOL:formatted.loc[method,m]='**'+formatted.loc[method,m]+'**'
        lines += [f'### {c.upper()}: {int(q.n_datasets.iloc[0])} strict matched datasets', '',formatted.to_markdown(),'']
    envelope=[]
    for c in CLFS:
        q=strict[strict.classifier==c].set_index('method')
        baselines=CORE+(['rus','smoteenn'] if c in ['svm','knn'] else [])
        for metric in METRICS:
            a=q.loc[FAMILY,metric];b=q.loc[baselines,metric]
            envelope.append(dict(classifier=c,metric=metric,best_variant=a.idxmax(),family_average_percent=100*a.max(),best_baseline=b.idxmax(),baseline_average_percent=100*b.max(),difference_pp=100*(a.max()-b.max())))
    lines += ['### Best observed variant averages versus all available baselines','', 'Each row selects the highest collection-wide variant average for a metric. This differs from selecting a different variant within every dataset. Both summaries are descriptive.','',pd.DataFrame(envelope).to_markdown(index=False,floatfmt='.3f'),'']
    lines += ['## C. Friedman results','',fr[['classifier','metric','n_datasets','n_methods','statistic','p_raw','p_holm_four_metrics','significant']].to_markdown(index=False,floatfmt='.6g'),'',
              'A significant Friedman result does not identify a superior CASB variant. Mean ranks and the complete method/dataset lists are stored in friedman_results.csv. All-pairs post hoc results are rows labelled friedman_posthoc in wilcoxon_holm_results.csv.','',
              '## D. Wilcoxon results','']
    primary=pairs[pairs.analysis=='dataset_pairs'];significant=primary[primary.significant.fillna(False)]
    lines += [primary.groupby(['classifier','direction']).size().unstack(fill_value=0).to_markdown(),'',
              'All significant improvements and decreases (scores/differences below remain fractions, not percentages):','',
              significant[['classifier','metric','method','baseline','n','mean_difference','rank_biserial','p_raw','p_holm','direction']].to_markdown(index=False,floatfmt='.6g'),'']
    edited=primary[(primary.method=='casb_edited')&(primary.metric.isin(['recall','gmean','precision','f1']))]
    lines += ['### CASB-edited against each baseline','',edited[['classifier','metric','baseline','n','mean_difference','p_holm','direction']].to_markdown(index=False,floatfmt='.6g'),'']
    lines += ['## E. Family-level performance','',
              'The following totals include all compatible baselines, including no resampling and, for SVM/KNN, RUS and SMOTE-ENN. The CSV also reports the original four-resampler population separately. A family win selects the best variant retrospectively; it measures an upper envelope of observed scores rather than expected deployable selector performance. No significance test is applied to this oracle. Joint wins require the same variant to beat the best baseline separately on both G-mean and F1.','',
              fam[(fam.row_type=='summary')&(fam.population=='all_available')][['classifier','metric','n','wins','ties','losses','win_percent','majority']].to_markdown(index=False,floatfmt='.3f'),'']
    lines += ['## Dataset dependence and sensitivity','',
              'Named source families are assigned by dataset identity (Abalone, Ecoli, Glass, Haberman, Iris, New-thyroid, Page-blocks, Pima, Segment, Shuttle, Vehicle, Vowel, Wisconsin, Yeast). Two sensitivity views are prespecified: equal means of valid paired differences within each source family; and one lexicographically first named task per source family, chosen without consulting performance. A representative task with a failed method remains missing for that pair and is not replaced with a favourable task. Exact sign tests are also Holm-corrected. These analyses reduce over-weighting of related variants but cannot prove family independence.','']
    for analysis in ['source_family_mean','one_task_per_family']:
        q=pairs[pairs.analysis==analysis]
        lines += [f'### {analysis}', '',q.groupby(['classifier','direction']).size().unstack(fill_value=0).to_markdown(),'',
                  f'Corrected exact sign-test rejections: {int((q.sign_p_holm<ALPHA).sum())}.','']
    lines += ['## F. Discussion','',
              'CASB-edited can trade precision for recall because its unrestricted generation and majority-only ENN alter the training geometry and class balance. These are plausible interpretations of the implementation, not a demonstrated causal explanation. A matched generation/cleaning ablation with identical quotas, ratios and classifier tuning is needed to attribute a gain to a particular component. A higher collection mean can coexist with non-significant pairwise results or a minority of dataset wins.','',
              '## G. Limitations and discrepancies from earlier summaries','',
              'Earlier averages compared the CASB family with four resamplers on 40 complete datasets. This report adds the no-resampling reference and compatible RUS/SMOTE-ENN baselines and uses 44 valid pairs where possible. Stronger baseline maxima can reduce family wins. Holm correction now has 112 or 80 hypotheses within classifier rather than a global 192/320-view family; exact p-values therefore need not match older reports. No pooled classifier view is a primary test. The 80-scope column permits an explicit scope comparison.', '',
              'Non-significant differences do not establish equivalence. G-mean/F1 consistency, precision-recall trade-offs, ADASYN failures, related KEEL variants, reused development datasets, retrospective variant/classifier selection, absent independent confirmation and unmatched cleaning ablations restrict claims. GaussianNB has the same stored protocol but is a separate additional classifier analysis. Holm control is within each declared family, not across every result in the report. Friedman uses the asymptotic chi-square approximation; the complete cohorts exceed ten datasets and compare more than six methods.', '',
              '## H. Future work and paper-ready conclusion','',
              'All four classifiers have significant Friedman differences for each metric after correcting four omnibus tests. Against all available baselines, RF alone has majority family wins for both G-mean (27/40) and F1 (26/40); its same-variant joint count is 19/40, below a majority. Against only the original four resamplers, RF wins 28/40 for G-mean, 26/40 for F1, and 20/40 jointly (exactly half). These are not counts out of 44.','',
              'CASB-edited with RF has corrected G-mean gains over Borderline-SMOTE and ROS and corrected recall gains over SMOTE, Borderline-SMOTE and ROS. It also has significant precision losses against those three methods. No fixed-variant Wilcoxon comparison survives either source-family sensitivity analysis after Holm correction.','',
              'The experiments support describing CASB as an evaluated SMOTE-based family with classifier- and metric-dependent gains and losses. They do not support universal superiority or establish novelty. A paper should present all corrected beneficial and harmful comparisons, disclose the post hoc family envelope and classifier selection, and distinguish performance from mechanism. Future work should test independent datasets, prespecify variant selection using training data, compare matched cleaning ablations, and improve G-mean/F1 consistency without claiming these changes are already achieved.','',
              '## Verification and reproducibility','',
              'Confusion-matrix metrics and training-only ratio choices were verified. Original notebook records were deduplicated by dataset/classifier/method/seed/fold. All 440 train/test file hashes matched notebook provenance. CSV means were cross-checked with independent arithmetic aggregation; Holm monotonicity, all-tie handling, paired-unit counts, Friedman ranks and family-win denominators are asserted by the analysis. Existing notebooks, code, progress report and datasets are SHA256-checked before/after analysis; hashes are embedded in the executed notebook. No training is imported or executed.','',
              f'Runtime: '+json.dumps({p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib']})+'.','',
              'Methods references: [Demsar (2006)](https://jmlr.org/papers/v7/demsar06a.html); [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html); [SciPy Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html).','']
    return '\n'.join(lines)


def report_pdf(text,figures):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape,A4
    from xml.sax.saxutils import escape
    styles=getSampleStyleSheet();styles['BodyText'].fontSize=9;styles['BodyText'].leading=12
    styles['Heading1'].fontSize=16;styles['Heading2'].fontSize=12
    story=[];lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[c.strip() for c in lines[i].strip('|').split('|')]
                if not all(set(c)<=set('-: ') for c in cells):table.append(cells)
                i+=1
            if table:
                width=760/len(table[0]);sty=styles['BodyText'].clone('cell');sty.fontSize=6.5;sty.leading=8
                import re
                data=[[Paragraph(re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',escape(c)),sty) for c in row] for row in table]
                t=Table(data,colWidths=[width]*len(table[0]),repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce7f1')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#cccccc')),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3)]))
                story.extend([t,Spacer(1,9)])
            continue
        if line.startswith('# '):story.append(Paragraph(escape(line[2:]),styles['Title']))
        elif line.startswith('## '):story.append(Paragraph(escape(line[3:]),styles['Heading1']))
        elif line.startswith('### '):story.append(Paragraph(escape(line[4:]),styles['Heading2']))
        elif line.strip():story.append(Paragraph(escape(line).replace('**',''),styles['BodyText']));story.append(Spacer(1,5))
        i+=1
    for title,fig in figures:
        buf=BytesIO();fig.savefig(buf,format='png',dpi=150,bbox_inches='tight');buf.seek(0)
        from PIL import Image as PILImage
        im=PILImage.open(buf);w,h=im.size;ratio=min(760/w,470/h)
        story += [PageBreak(),Paragraph(escape(title),styles['Heading1']),Image(buf,width=w*ratio,height=h*ratio)]
    doc=SimpleDocTemplate(str(OUT/'statistical_report.pdf'),pagesize=landscape(A4),rightMargin=40,leftMargin=40,topMargin=35,bottomMargin=35,title='CASB statistical evidence',author='')
    def footer(c,d):c.setFont('Helvetica',8);c.drawRightString(800,18,str(d.page))
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def run():
    OUT.mkdir(exist_ok=True)
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    assert paired([1,1],[1,1])['p_raw']==1 and paired([1,1],[1,1])['ties']==2
    raw,scores,audit,hashes=extract()
    avg,pairs,fr,wtl,fam=make_tables(scores)
    for metric in METRICS+EXTRA:
        for r in avg.itertuples():
            q=scores[(scores.classifier==r.classifier)&(scores.method==r.method)&(scores.status=='complete')]
            if r.population=='strict_all_methods':
                methods=CORE+FAMILY+(['rus','smoteenn'] if r.classifier in ['svm','knn'] else [])
                complete=scores[(scores.classifier==r.classifier)&(scores.status=='complete')].pivot(index='dataset',columns='method',values='gmean')[methods].dropna().index
                q=q[q.dataset.isin(complete)]
            assert math.isclose(getattr(r,metric),sum(q[metric])/len(q),abs_tol=1e-14)
    summaries=fam[fam.row_type=='summary']
    assert ((summaries.wins+summaries.ties+summaries.losses)==summaries.n).all()
    for r in pairs.itertuples():
        assert r.n==r.wins+r.ties+r.losses
        assert r.p_holm+1e-14>=r.p_raw
    outputs={'dataset_scores.csv':scores,'average_metric_scores.csv':avg,'wilcoxon_holm_results.csv':pairs,
             'friedman_results.csv':fr,'win_tie_loss_results.csv':wtl,'family_win_analysis.csv':fam}
    for name,table in outputs.items():table.to_csv(OUT/name,index=False,float_format='%.17g')
    text=markdown_report(raw,scores,audit,avg,pairs,fr,fam,hashes)
    (OUT/'statistical_summary.md').write_text(text,encoding='utf8')
    figures=plot_figures(avg,pairs,fr,fam)
    report_pdf(text,figures)
    # One self-contained executed analysis notebook, with actual computed tables,
    # embedded charts and source code; execute via nbclient when requested.
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# CASB saved-result statistical analysis\n\nNo model training. The source file beside this notebook reproduces all outputs. Raw failures, provenance, strict and pairwise populations are retained. RF/LR were added to the original notebooks before this analysis at the user\'s request.')])
    n.cells.append(nbformat.v4.new_code_cell("from pathlib import Path\nimport sys\nHERE=Path.cwd()\nif HERE.name!='statistical result':HERE=HERE/'statistical result'\nsys.path.insert(0,str(HERE))\nfrom statistical_analysis import run\nresult=run()\nfrom IPython.display import display,Markdown\nfor name in ['audit','averages','friedman','pairs','family','failures']:\n    display(Markdown('## '+name));display(result[name])\nfor title,png in result['charts']:\n    from IPython.display import Image\n    display(Markdown('## '+title));display(Image(data=png))\n"))
    n.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
    n.metadata['input_sha256']=hashes
    n.metadata['statistical_analysis_source']=Path(__file__).read_text(encoding='utf8')
    # Do not overwrite a notebook being executed by nbclient.
    if not (OUT/'statistical_analysis.ipynb').exists():nbformat.write(n,OUT/'statistical_analysis.ipynb')
    charts=[]
    for title,fig in figures:
        buf=BytesIO();fig.savefig(buf,format='png',dpi=160,bbox_inches='tight');charts.append((title,buf.getvalue()));plt.close(fig)
    for p,h in hashes.items():assert digest(ROOT/p)==h, 'Input changed during analysis: '+p
    print(f'Verified {len(raw)} records; {len(scores)} dataset-level rows; all {len(hashes)} protected input hashes unchanged.')
    print(fr[['classifier','metric','p_holm_four_metrics','significant']].to_string(index=False))
    print(summaries[summaries.population=='all_available'][['classifier','metric','n','wins','ties','losses']].to_string(index=False))
    return dict(audit=audit,averages=avg,friedman=fr,pairs=pairs,family=fam,failures=raw[raw.error.notna()],charts=charts)


if __name__=='__main__':run()
