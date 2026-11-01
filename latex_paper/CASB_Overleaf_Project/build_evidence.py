"""Rebuild manuscript tables from immutable executed notebook records; never train."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'tables'; OUT.mkdir(exist_ok=True)
FIG=HERE/'figures'; FIG.mkdir(exist_ok=True)
CLFS=['svm','knn','rf','lr']; MET=['gmean','f1','precision','recall']
METHODS=['original','smote','bsmote','ros','adasyn','casb','casb_relaxed','casb_edited','casb_ros']
FAM=METHODS[5:]; BASE=METHODS[1:5]
NAME={'original':'No resampling','smote':'SMOTE','bsmote':'B-SMOTE','ros':'ROS','adasyn':'ADASYN','casb':'CASB','casb_relaxed':'CASB-relaxed','casb_edited':'CASB-edited','casb_ros':'CASB-ROS','rus':'RUS','smoteenn':'SMOTE-ENN','gmean':'G-mean','f1':'F1','precision':'Precision','recall':'Recall','gnb':'GNB'}
def esc(x): return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def name(x): return NAME.get(x,str(x).upper())
def table(file,caption,headers,rows,fmt=None):
    fmt=fmt or 'l'+'r'*(len(headers)-1)
    s='\\begin{table}[pos=!htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{tab:'+file+'}\n'
    s+='\\begin{tabular}{'+fmt+'}\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    s+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n'
    (OUT/(file+'.tex')).write_text(s,encoding='utf-8')

inputs={}; records=[]; inventory=[]
for p in sorted(ROOT.glob('[0-9][0-9]_*_CASB.ipynb')):
    inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    n=json.loads(p.read_text(encoding='utf-8')); meta=n['metadata']['current_comparison']; rr=meta['records']
    assert len(rr)==360 and set(meta['classifiers'])==set(CLFS)
    assert not any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))
    records.extend(rr)
    r=next(r for r in rr if 'error' not in r)
    tc=np.array(r['train_counts'])+np.array(r['test_counts'])
    datafile=next(iter(meta['input_sha256'])).replace('\\','/')
    dp=ROOT/'output'/datafile
    lines=dp.read_text(encoding='utf-8-sig').splitlines()
    dims=sum(l.lower().startswith('@attribute') for l in lines)-1
    inventory.append([esc(r['dataset']),sum(tc),dims,int(tc[0]),int(tc[1]),f'{tc[0]/tc[1]:.2f}'])
df=pd.DataFrame(records)
counts_by_fold={(r['dataset'],r['fold'],r['seed']):r['train_counts'] for r in records if isinstance(r.get('train_counts'),list)}
df['train_counts']=[counts_by_fold[(r.dataset,r.fold,r.seed)] for r in df.itertuples()]
assert len(df)==15840 and df.error.notna().sum()==52
assert not df.duplicated(['dataset','classifier','method','fold','seed']).any()
scores=pd.read_csv(ROOT/'statistical result/dataset_scores.csv')
averages=pd.read_csv(ROOT/'statistical result/average_metric_scores.csv')
av=averages[averages.population=='strict_all_methods']
strict=set(scores.query("classifier=='svm' and method=='adasyn' and status=='complete'").dataset)
assert len(strict)==40
means=df[df.dataset.isin(strict)].groupby(['classifier','method','dataset'])[MET].mean().groupby(['classifier','method']).mean()
for c in CLFS:
    rows=[]
    for m in METHODS:
        a=av[(av.classifier==c)&(av.method==m)].iloc[0]
        np.testing.assert_allclose(means.loc[(c,m),MET],a[MET].astype(float),rtol=0,atol=1e-12)
        rows.append([name(m)]+[f'{a[v]*100:.3f}' for v in MET])
    table('scores_'+c,f'{c.upper()}: equal-weight means on 40 matched datasets (percent). Each dataset score averages five supplied folds and two seeds.', ['Method']+[name(v) for v in MET],rows)
for cs,tag in [(CLFS,'combined'),(['svm','knn'],'svm_knn')]:
    a=means.loc[cs].groupby('method').mean()
    table('scores_'+tag,'Equal-classifier and equal-dataset means: '+', '.join(c.upper() for c in cs)+'; 40 matched datasets (percent).',['Method']+[name(v) for v in MET],[[name(m)]+[f'{a.loc[m,v]*100:.3f}' for v in MET] for m in METHODS])
table('datasets','Dataset inventory from the first supplied train/test pair. Counts refer to original data; dimensions precede one-hot expansion.',['Dataset','$n$','$p$','$n_{maj}$','$n_{min}$','IR'],inventory)
compact=[]
for i in range(22):
    compact.append([inventory[i][j] for j in [0,1,2,5]]+[inventory[i+22][j] for j in [0,1,2,5]])
table('datasets','All 44 tasks: original observation count $n$, pre-encoding dimension $p$, and majority/minority imbalance ratio IR, from the first supplied train/test pair. Full class counts are retained in the evidence inventory.',['Dataset','$n$','$p$','IR','Dataset','$n$','$p$','IR'],compact,'lrrrlrrr')
evidence=HERE/'evidence'; evidence.mkdir(exist_ok=True)
pd.DataFrame(inventory,columns=['dataset','n','p','n_majority','n_minority','imbalance_ratio']).to_csv(evidence/'dataset_inventory.csv',index=False)

env=[]
for c in CLFS:
    a=means.loc[c]
    for v in MET:
        f=a.loc[FAM,v].idxmax(); b=a.loc[BASE,v].idxmax()
        env.append([c.upper(),name(v),name(f),f'{a.loc[f,v]*100:.3f}',name(b),f'{a.loc[b,v]*100:.3f}',f'{(a.loc[f,v]-a.loc[b,v])*100:+.3f}'])
table('envelope','Best collection-wide fixed-variant averages versus the four core resamplers. Selection is retrospective, separately for each metric; differences are percentage points.',['Clf.','Metric','Family member','Score','Baseline','Score','$\Delta$'],env,'lllrlrr')
fw=pd.read_csv(ROOT/'statistical result/family_win_analysis.csv')
fr=[]
for pop in ['original_four_resamplers','all_available']:
    for c in CLFS:
        x=fw[(fw.row_type=='summary')&(fw.population==pop)&(fw.classifier==c)]
        fr.append(['Core four' if pop.startswith('original') else 'All available',c.upper()]+[str(int(x.loc[x.metric==v,'wins'].iloc[0])) for v in MET+['joint_gmean_f1_same_variant']])
table('family','Dataset-level retrospective family wins out of 40. Joint requires the same variant to beat every baseline on both G-mean and F1.',['Scope','Clf.','G-mean','F1','Prec.','Recall','Joint'],fr,'llrrrrr')
w=pd.read_csv(ROOT/'statistical result/wilcoxon_holm_results.csv')
rows=[]
for c in CLFS:
    for v in MET:
        for m,b in [('casb','smote'),('casb_ros','ros')]:
            r=w[(w.analysis=='dataset_pairs')&(w.classifier==c)&(w.metric==v)&(w.method==m)&(w.baseline==b)].iloc[0]
            rows.append([c.upper(),name(v),name(m)+' / '+name(b),f'{int(r.wins)}/{int(r.ties)}/{int(r.losses)}',f'{r.mean_difference*100:+.3f}',f'{r.p_holm:.4g}'])
table('paired','Fixed CASB versus SMOTE and CASB-ROS versus ROS. All 44 paired datasets; W/T/L counts, mean difference in percentage points, classifier-specific Holm-adjusted p-values.',['Clf.','Metric','Pair','W/T/L','$\Delta$','$p_H$'],rows,'llllrr')
fried=pd.read_csv(ROOT/'statistical result/friedman_results.csv')
table('friedman','Omnibus Friedman tests on the 40 complete datasets. Holm correction over four metrics within classifier. Rejection does not identify a CASB winner.',['Clf.','Metric','$k$','$\chi_F^2$','$p_H$'],[[r.classifier.upper(),name(r.metric),r.n_methods,f'{r.statistic:.3f}',f'{r.p_holm_four_metrics:.3g}'] for r in fried.itertuples() if r.classifier in CLFS])
rank_rows=[]
for m in METHODS+['rus','smoteenn']:
    row=[name(m)]
    for c in CLFS:
        for v in MET[:2]:
            rr=json.loads(fried[(fried.classifier==c)&(fried.metric==v)].iloc[0].mean_ranks_json)
            row.append(f'{rr[m]:.3f}' if m in rr else '--')
    rank_rows.append(row)
table('ranks','Mean ranks for G-mean (G) and F1 (F). Lower is better; SVM/KNN rank eleven methods, RF/LR nine. Ranks cannot be compared across unequal method sets.',['Method']+[c.upper()+' '+v for c in CLFS for v in ['G','F']],rank_rows)
sig=[]
for c in CLFS:
    x=w[(w.analysis=='dataset_pairs')&(w.classifier==c)]
    sig.append([c.upper(),len(x),sum((x.p_holm<.05)&(x.mean_difference>0)),sum((x.p_holm<.05)&(x.mean_difference<0)),sum(x.p_holm>=.05)])
table('significance','Fixed-variant Wilcoxon-Holm decisions, including no resampling and compatible legacy controls. Counts span all four metrics.',['Classifier','Tests','CASB higher','CASB lower','Not significant'],sig)

# Confusion rates and majority removal: average within dataset, then across datasets.
good=df[df.dataset.isin(strict)&df.error.isna()].copy()
good['fpr']=good.confusion.map(lambda x:x[1]/(x[0]+x[1]))
good['fnr']=good.confusion.map(lambda x:x[2]/(x[2]+x[3]))
good['removed_frac']=good.apply(lambda r:(r.audit if isinstance(r.audit,dict) else {}).get('majority_enn_removed',0)/r.train_counts[0],axis=1)
for r in good[good.method=='casb_edited'].itertuples():
    removed=r.train_counts[0]-r.resampled_counts[0]
    if isinstance(r.audit,dict): assert removed==r.audit['majority_enn_removed']
    else: assert removed==0 and r.train_counts==r.resampled_counts  # wrapper no-op
cm=good.groupby(['classifier','method','dataset'])[['fpr','fnr','removed_frac']].mean().groupby(['classifier','method']).mean()
table('mechanisms','Observed error-rate changes (percentage points) for edited minus original CASB, on 40 datasets. Removal is the mean percentage of original majority training observations removed by edited. These paired pipeline comparisons are not component ablations.',['Clf.','$\Delta$ FP rate','$\Delta$ FN rate','Majority removed (\%)'],[[c.upper(),f'{100*(cm.loc[(c,"casb_edited"),"fpr"]-cm.loc[(c,"casb"),"fpr"]):+.3f}',f'{100*(cm.loc[(c,"casb_edited"),"fnr"]-cm.loc[(c,"casb"),"fnr"]):+.3f}',f'{100*cm.loc[(c,"casb_edited"),"removed_frac"]:.3f}'] for c in CLFS])

# Historical three-variant selector: recompute decisions using saved validation only.
selected=[]
for key,g in df[df.classifier.isin(['svm','knn'])].groupby(['dataset','classifier','fold','seed']):
    for label,methods in [('CASB selector',['casb','casb_relaxed','casb_edited'])]:
        cand=[]
        for j,m in enumerate(methods):
            r=g[g.method==m].iloc[0]
            for trial in r.validation:
                if 'error' not in trial: cand.append((trial['ba'],-trial['ratio'],-j,m,trial['ratio']))
        if cand:
            chosen=max(cand); r=g[g.method==chosen[3]].iloc[0]; assert r.ratio==chosen[4]
            selected.append(dict(dataset=key[0],classifier=key[1],method=label,**{v:r[v] for v in MET}))
sel=pd.DataFrame(selected); assert len(sel)==880
sm=sel.groupby(['classifier','dataset'])[MET].mean().groupby('classifier').mean()
table('selector','Historical training-only selector over three CASB variants and three ratios (44 datasets). CASB-ROS is not a candidate; RF/LR selector experiments were not run. Scores are percent.',['Classifier']+[name(v) for v in MET],[[c.upper()]+[f'{sm.loc[c,v]*100:.3f}' for v in MET] for c in ['svm','knn']]+[['Equal SVM/KNN']+[f'{sm[v].mean()*100:.3f}' for v in MET]])

on=json.loads((ROOT/'paper_material/CASB_Overlap_Experiment.ipynb').read_text(encoding='utf-8'))
od=pd.DataFrame(on['metadata']['overlap_experiment']['records']+on['metadata']['classifier_extension']['records'])
assert len(od)==7920 and od.error.notna().sum()==8
om=od[od.distance!=5].groupby(['classifier','method','seed'])[MET].mean().groupby(['classifier','method']).mean()
table('overlap','Simulation averages over 20 independent seeds and ten complete distances (0 to 4.5); distance 5 is omitted for every method due to ADASYN failures. Scores are percent.',['Clf.','Method']+[name(v) for v in MET],[[c.upper(),name(m)]+[f'{om.loc[(c,m),v]*100:.2f}' for v in MET] for c in CLFS for m in METHODS],'llrrrr')

for p in sorted((ROOT/'paper_material').glob('[0-9][0-9]_*.pdf'))+sorted((ROOT/'paper plot').glob('*.pdf')):
    shutil.copy2(p,FIG/p.name)
    inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in (ROOT/'statistical result').glob('*.csv'):
    inputs[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    evidence=HERE/'evidence'; evidence.mkdir(exist_ok=True)
    shutil.copy2(p,evidence/p.name)
inputs['binary_oversampling.py']=hashlib.sha256((ROOT/'binary_oversampling.py').read_bytes()).hexdigest()
(HERE/'evidence_manifest.json').write_text(json.dumps({'input_sha256':inputs,'evaluations':len(df),'failures':52,'matched_datasets':40,'notebooks':44,'figures':len(list(FIG.glob('*.pdf'))),'selector':sm.to_dict(),'confusion_analysis':cm.reset_index().to_dict('records')},indent=2),encoding='utf-8')
print('Verified 44 notebooks, 15840 records, 52 failures; tables and 23 figure assets generated.')

# UGO-style comparison layouts: metrics grouped down rows, methods across columns.
SHORT={'original':'NORES','ros':'ROS','smote':'SMOTE','bsmote':'B-SMOTE','adasyn':'ADASYN','casb':'CASB','casb_relaxed':r'\shortstack{CASB\\relaxed}','casb_edited':r'\shortstack{CASB\\edited}','casb_ros':r'\shortstack{CASB\\ROS}'}
ORDER=['original','ros','smote','bsmote','adasyn']+FAM
def wide_score(tag,caption,metrics,groups,lookup):
    rows=[]
    for metric in metrics:
        for j,(label,code) in enumerate(groups):
            vals=np.array([lookup(code,m,metric) for m in ORDER],float)
            cells=[(r'\textbf{'+f'{v:.3f}'+'}') if abs(v-vals.max())<=1e-12 else f'{v:.3f}' for v in vals]
            rows.append(' & '.join([name(metric) if j==0 else '',label]+cells)+r' \\')
        rows.append(r'\addlinespace[3pt]')
    s=r'\begin{table}[pos=!htbp]'+'\n'+r'\centering\small\setlength{\tabcolsep}{3pt}'+'\n'
    s+=r'\caption{'+caption+r'}\label{tab:'+tag+'}\n'
    s+=r'\resizebox{\linewidth}{!}{\begin{tabular}{llrrrrrrrrr}\toprule'+'\n'
    s+='Metric & Classifier & '+' & '.join(SHORT[m] for m in ORDER)+r' \\\midrule'+'\n'
    s+='\n'.join(rows)+r'\bottomrule\end{tabular}}'+'\n'+r'\end{table}'+'\n'
    (OUT/(tag+'.tex')).write_text(s,encoding='utf-8')
def bench_lookup(c,m,v):
    if c=='overall': return means.loc[CLFS].groupby('method').mean().loc[m,v]
    if c=='svmknn': return means.loc[['svm','knn']].groupby('method').mean().loc[m,v]
    return means.loc[(c,m),v]
wide_score('benchmark_comparison','Benchmark performance on 40 matched datasets. Each metric groups SVM, KNN, RF, LR and equal-classifier summaries; methods are columns. Scores are fractions, not percentages. Bold identifies the largest unrounded value in each row (ties within $10^{-12}$). NORES denotes no resampling. Different rows can have different winners.',MET,[(c.upper(),c) for c in CLFS]+[('SVM/KNN','svmknn'),('Overall','overall')],bench_lookup)
simmetrics=['iam','mcc']+MET
sim=od[od.distance!=5].groupby(['classifier','method','seed'])[simmetrics].mean().groupby(['classifier','method']).mean()
wide_score('simulation_comparison','Gaussian overlap simulation performance, averaged equally over 20 seeds and ten complete distances (0--4.5). Distance 5 is excluded for every method because of eight ADASYN failures. Metric/classifier groups and method columns follow the reference-paper layout. Bold marks the largest unrounded row score; these descriptive averages do not establish significance.',simmetrics,[(c.upper(),c) for c in CLFS],lambda c,m,v:sim.loc[(c,m),v])
rankrows=[]; rankaudit=[]
for metric in MET:
    for j,c in enumerate(['overall']+CLFS):
        x=scores[(scores.dataset.isin(strict))&scores.classifier.isin(CLFS)&scores.method.isin(ORDER)]
        if c=='overall': x=x.groupby(['dataset','method'])[metric].mean().unstack('method')[ORDER]
        else: x=x[x.classifier==c].pivot(index='dataset',columns='method',values=metric)[ORDER]
        ranks=x.rank(axis=1,ascending=False,method='average').mean()
        counts=x.sub(x.max(axis=1),axis=0).abs().le(1e-12).sum()
        cells=[r'\textbf{'+f'{v:.3f}'+'}' if abs(v-ranks.min())<=1e-12 else f'{v:.3f}' for v in ranks]
        wins=[r'\textbf{'+str(v)+'}' if v==counts.max() else str(v) for v in counts]
        rankrows.append(' & '.join([name(metric) if j==0 else '',c.upper() if c!='overall' else 'Overall']+cells+wins)+r' \\')
        for m in ORDER: rankaudit.append(dict(metric=metric,classifier=c,method=m,average_rank=ranks[m],wins=int(counts[m]),n=40))
    rankrows.append(r'\addlinespace[4pt]')
s=r'\begin{table}[pos=p]\centering\small\setlength{\tabcolsep}{3pt}'+'\n'
s+=r'\caption{Average ranks and dataset win counts, on the same 40 tasks and nine methods throughout. Overall first averages the four classifiers within each dataset, then ranks methods. Lower average rank is better; wins count every method tied for the highest score within $10^{-12}$, so row totals can exceed 40. Bold marks the best rank or largest win count separately. These are individual-method wins, not retrospective CASB-family wins.}\label{tab:rank_wins}'+'\n'
for panel, start in [('A. Average ranks', 2), ('B. Dataset win counts', 11)]:
    s+=r'\resizebox{.86\linewidth}{!}{\begin{tabular}{llrrrrrrrrr}\toprule'+'\n'
    s+=r'\multicolumn{11}{l}{\textbf{'+panel+r'}} \\\midrule'+'\n'
    s+='Metric & Group & '+' & '.join(SHORT[m] for m in ORDER)+r' \\\midrule'+'\n'
    for row in rankrows:
        if row.startswith(r'\addlinespace'):
            s+=r'\addlinespace[2pt]'+'\n'
        else:
            cells=row.removesuffix(r' \\').split(' & ')
            s+=' & '.join(cells[:2]+cells[start:start+9])+r' \\'+'\n'
    s+=r'\bottomrule\end{tabular}}\par\vspace{5pt}'+'\n'
s+=r'\end{table}'+'\n'
(OUT/'rank_wins.tex').write_text(s,encoding='utf-8')
pd.DataFrame(rankaudit).to_csv(HERE/'evidence/paper_rank_win_counts.csv',index=False,float_format='%.17g')
print('UGO-style benchmark, simulation and horizontal stacked rank/win tables generated.')



