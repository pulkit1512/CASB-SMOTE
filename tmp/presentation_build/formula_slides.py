def revise(slides):
    removed={'Training and evaluation protocol','Statistical testing protocol','Corrected fixed-variant comparisons'}
    slides[:]=[s for s in slides if s['title'] not in removed]
    for s in slides:
        if s['section'].startswith('06 '): s['section']='06  Dataset win counts'
    extra=[]
    def table(title,rows,notes,take):
        extra.append(dict(kind='table',title=title,section='04  Proposed method',source='latex_paper/manuscript.tex: Proposed method',headers=['Symbol / rule','Formula or value','Meaning'],rows=rows,widths=[.20,.45,.35],notes=notes,take=take))
    table('CASB notation: data and neighbours',[
        ['M / J','M = {i : yᵢ = 1}; J = {i : yᵢ = 0}','Minority / majority indices'],
        ['nₘ / nⱼ','nₘ = |M|; nⱼ = |J|','Class sizes'],
        ['Nᵢ','Up to 10 nearest training neighbours','Exclude the source index'],
        ['Kᵢ','Up to 5 nearest minority neighbours','Sorted by distance'],
        ['Distance','dist(xᵢ, xⱼ) = ‖xᵢ − xⱼ‖₂','Euclidean distance']],
        'M is the list of minority points, and J is the list of majority points. The symbols n m and n j count them. For each minority point, N i contains up to ten nearest training neighbours from both classes. We exclude the point itself by index, so a different point with the same coordinates can remain. K i contains up to five nearest minority neighbours. We measure distance after training-only feature preparation. Each point has one class label, not two.',
        'Minority label = 1; majority label = 0. Distances use preprocessed training features.')
    table('Local measures: oᵢ, sᵢ, dᵢ and qᵢ',[
        ['oᵢ','oᵢ = Σⱼ∈Nᵢ 𝟙(yⱼ = 0) / |Nᵢ|','Local majority fraction'],
        ['sᵢ','sᵢ = Σⱼ∈Nᵢ 𝟙(yⱼ = 1)','Minority neighbour count'],
        ['dᵢ','dᵢ = minⱼ∈J ‖xᵢ − xⱼ‖₂','Nearest majority distance'],
        ['qᵢ','qᵢ = ‖xᵢ − xKᵢ[min(3,|Kᵢ|)]‖₂','Third-minority-neighbour distance']],
        'Read o i as the share of nearby points that belong to the majority class. Read s i as the number of nearby minority points. The indicator symbol means count one when the condition is true, and zero otherwise. d i is the distance to the closest majority point. q i measures how far away the third minority neighbour is. If fewer than three exist, use the last available neighbour. A large q i suggests a sparse minority area. In a generation step, we use the same values at source s, written o s, s s and d s.',
        'Example: 4 majority + 6 minority neighbours gives oᵢ = 0.4 and sᵢ = 6.')
    table('Exact conditions for the three source pools',[
        ['Coverage C','C = M','Every minority point'],
        ['Border B','B = {i ∈ M : 0.2 < oᵢ ≤ 0.7,\nsᵢ ≥ 2, dᵢ > 0}','Mixed neighbourhood\nwith minority support'],
        ['Rare R','R = {i ∈ M : qᵢ ≥ Q₀.₇₅(q),\nsᵢ ≥ 2, dᵢ > 0}','Sparse neighbourhood\nwith minority support']],
        'Coverage has no extra filter: every minority point is eligible. Border requires more than twenty percent and at most seventy percent majority neighbours. It also needs at least two minority neighbours and a positive distance to the nearest majority point. Rare uses the seventy-fifth percentile of q as its threshold, with the same support and distance checks. Points exactly at that threshold are included. The groups overlap; they are not three separate classes. Coverage can still include noisy minority points.',
        'Q₀.₇₅ is the 75th percentile. C includes B and R; B and R can overlap.')
    table('Budget, weights and exact integer allocation',[
        ['Added samples','G = max(0, ⌈ρ nⱼ⌉ − nₘ)','ρ = target minority / majority ratio'],
        ['Original / ROS','(wC, wB, wR) = (0.50, 0.30, 0.20)','Coverage / border / rare'],
        ['Relaxed / edited','(wC, wB, wR) = (0.70, 0.20, 0.10)','More coverage allocation'],
        ['Initial quota','aᵦ = ⌊wᵦG⌋; remainder = G − Σᵦ aᵦ','Add remaining units by largest fractions']],
        'The target ratio rho tells us how many minority points we want compared with majority points. Multiply rho by the majority count, round up, then subtract the current minority count. Never add a negative number. Multiply this budget by each branch weight and round down first. Give leftover units to the largest fractional remainders. Ties use coverage, border, rare order. Training validation chooses rho from 0.5, 0.75 and 1.0. These branch weights and thresholds are fixed design choices.',
        'ρ candidates: 0.50, 0.75, 1.00. Rounding preserves exactly G additions.')
    table('Source selection and overlapping membership',[
        ['Effective pool Pᵦ','Pᵦ = branch pool; if empty, use C','Keep the allocated branch rule'],
        ['Source probability','πᵢ = Σᵦ (aᵦ/G) 𝟙(i ∈ Pᵦ) / |Pᵦ|','For a randomly chosen addition; G > 0'],
        ['Expected uses','E[uses of i] = Gπᵢ','Overlapping membership adds influence'],
        ['Never selected','P(unused i) = ∏ᵦ (1 − 𝟙(i ∈ Pᵦ)/|Pᵦ|)ᵃᵦ','Eligibility is not guaranteed use']],
        'Within a branch, every eligible source has the same chance, and we sample with replacement. A point can therefore be chosen many times or not at all. If a point belongs to several pools, it has several routes to being chosen. Pi i adds those routes. Multiplying by G gives its expected number of uses. If a border or rare pool is empty, choose its source from coverage, but keep that branch’s neighbour and step rules. These formulas explain why broad eligibility is different from guaranteed coverage.',
        'Majority points provide context; they are never generation sources.')
    table('Generate a point: formula and branch limits',[
        ['Neighbour t','t ∈ Kₛ; rare uses first min(2, |Kₛ|)','Choose uniformly'],
        ['New point','L = ‖xₜ − xₛ‖₂; z = xₛ + λ(xₜ − xₛ)','λ ∼ Uniform(0, λmax)'],
        ['Zero edge','λmax = 0 if L = 0','Repeat source coordinates'],
        ['Coverage exception','λmax = 1 if b = C and oₛ ≤ 0.2','Full segment allowed'],
        ['Rare / other','Rare: min(1, 0.35 dₛ/L)\nOtherwise: min(1, 0.50 dₛ/L)','Apply after the zero-edge case']],
        'Choose a source s and a minority neighbour t. L is the distance between them. Lambda is a random fraction of the allowed move, and z is the new point. A zero-length edge copies the source. Coverage with at most twenty percent majority neighbours allows the whole line. A rare step is limited by 0.35 times the nearest-majority distance. Other capped steps use 0.50. Both limits stop at the selected neighbour. For L equal to four and d s equal to two, the ordinary maximum lambda is 0.25 and the rare maximum is 0.175.',
        'Branch limits use dₛ and oₛ of the selected source, including after pool fallback.')
    table('What the distance cap guarantees—and does not',[
        ['Capped movement','r = ‖z − xₛ‖₂ ≤ c dₛ; c ≤ 0.5','c = 0.50 or 0.35'],
        ['Observed majority u','‖z − u‖₂ ≥ dₛ − r ≥ r','z is no farther from its source'],
        ['Scope','Only the capped branches','Not unrestricted coverage / relaxed'],
        ['Not guaranteed','Correct class label of z','Missing points and label noise remain']],
        'A short geometric argument explains the cap. Call the movement from the source r. If r is no more than half the nearest-majority distance, the new point is at least as close to its source as to any observed majority point. This follows from the triangle inequality. It is a distance property, not proof of the true class. It does not apply when we allow the full line, and it does not protect us from a wrongly labelled source.',
        'Geometric closeness to a source does not certify a synthetic label.')
    table('Variant formulas: relaxed, edited and ROS',[
        ['Relaxed','λmax = 1 for L > 0; weights 70/20/10','Same pools; full interpolation'],
        ['Edited removal','Remove j ∈ J if mode(yN₃(j)) ≠ yⱼ','After relaxed generation; majority only'],
        ['After h removals','Final ratio = (nₘ + G) / (nⱼ − h)','May exceed the target ratio'],
        ['CASB-ROS','z = xₛ; weights 50/30/20','Copy pool-selected sources']],
        'Relaxed allows the whole nonzero segment and changes the weights to seventy, twenty and ten percent. Edited starts with relaxed generation. For each targeted majority point, look at its three nearest neighbours and take their most common label. Remove that majority point if the vote disagrees with its label. Minority points are kept. If h majority points are removed, the final ratio can be higher than the requested one. CASB-ROS copies the source, using the original fifty, thirty and twenty percent weights.',
        'Edited uses 3-neighbour mode ENN; it cannot remove mislabeled minority points.')
    index=next(i for i,s in enumerate(slides) if s['title']=='The three extensions: what changes?')
    slides[index:index]=extra
    # Reorder by titles so the user's original slide numbers remain unambiguous.
    def move_after(title, anchor):
        item=next(s for s in slides if s['title']==title)
        slides.remove(item)
        slides.insert(next(i for i,s in enumerate(slides) if s['title']==anchor)+1,item)
    move_after('Exact conditions for the three source pools','Step 1: identify useful minority sources')
    move_after('Local measures: oᵢ, sᵢ, dᵢ and qᵢ','Exact conditions for the three source pools')
    item=next(s for s in slides if s['title']=='Budget, weights and exact integer allocation')
    slides.remove(item)
    slides.insert(next(i for i,s in enumerate(slides) if s['title']=='Step 2: divide the generation budget'),item)
    slides[:]=[s for s in slides if s['title']!='What the distance cap guarantees—and does not']
    labels={
        'CASB notation: data and neighbours':'xᵢ = feature vector; yᵢ = class label; |A| = number of elements in A; ‖·‖₂ = Euclidean length.',
        'Local measures: oᵢ, sᵢ, dᵢ and qᵢ':'𝟙 = 1 if true, else 0; Σ = sum; Nᵢ = local neighbours; Kᵢ = ordered minority neighbours; J = majority indices.',
        'Exact conditions for the three source pools':'M = minority indices; oᵢ = majority fraction; sᵢ = minority support; dᵢ = majority distance; qᵢ = sparsity distance. Q₀.₇₅ = 75th percentile.',
        'Budget, weights and exact integer allocation':'G = additions; ρ = target ratio; nₘ / nⱼ = minority / majority counts; b = branch; wᵦ = weight; aᵦ = quota. ⌈ ⌉ rounds up; ⌊ ⌋ rounds down.',
        'Source selection and overlapping membership':'b ∈ {C, B, R}; Pᵦ = effective pool; aᵦ = additions; G = total; πᵢ = source probability. 𝟙 = indicator; Σ = sum; ∏ = product.',
        'Generate a point: formula and branch limits':'s = source; t = neighbour; L = source-to-neighbour distance; z = new point; λ = step fraction; λmax = upper limit; dₛ = nearest-majority distance; oₛ = majority fraction; b = branch.',
        'Variant formulas: relaxed, edited and ROS':'N₃(j) = 3 nearest neighbours after generation; mode = most common label; h = majority points removed. nₘ / nⱼ = original class counts; G = additions.',
    }
    for s in slides:
        if s['title'] in labels:
            s['take']=labels[s['title']]
            s['notes']+=' Symbol guide: '+labels[s['title']]
            s['take_size']=14
        if s['title']=='Generate a point: formula and branch limits':
            s['rows'][1][2]='L = source-to-neighbour distance\nλ ∼ Uniform(0, λmax)'
        if s['title']=='The interpolation rule':
            s['take']='z = new point; xₛ = source; xₜ = neighbour; λ = step fraction. L = neighbour distance; d = nearest-majority distance; c = cap factor.'
            s['take_size']=14
    slides[:]=[s for s in slides if s['title'] not in {'CASB notation: data and neighbours','Source selection and overlapping membership'}]
    references=[
        ('Core oversampling methods',[
            ['Chawla et al. (2002)','SMOTE: Synthetic Minority Over-sampling Technique','JAIR 16, 321–357\n10.1613/jair.953'],
            ['Han, Wang & Mao (2005)','Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning','LNCS 3644, 878–887\n10.1007/11538059_91'],
            ['He et al. (2008)','ADASYN: Adaptive synthetic sampling approach for imbalanced learning','IJCNN, 1322–1328\n10.1109/IJCNN.2008.4633969']]),
        ('Local structure and generation geometry',[
            ['Bunkhumpornpat et al. (2009)','Safe-Level-SMOTE: Safe-Level-Synthetic Minority Over-Sampling TEchnique for Handling the Class Imbalanced Problem','PAKDD, 475–482\n10.1007/978-3-642-01307-2_43'],
            ['Douzas & Bacao (2019)','Geometric SMOTE a geometrically enhanced drop-in replacement for SMOTE','Information Sciences 501, 118–135\n10.1016/j.ins.2019.06.007'],
            ['Napierala & Stefanowski (2016)','Types of minority class examples and their influence on learning classifiers from imbalanced data','JIIS 46, 563–597\n10.1007/s10844-015-0368-1']]),
        ('Related research',[
            ['Skryjomski & Krawczyk (2017)','Influence of minority class instance types on SMOTE imbalanced data oversampling','PMLR 74, 7–21\nproceedings.mlr.press/v74/skryjomski17a.html'],
            ['Koziarski & Woźniak (2024)','Local neighborhood encodings for imbalanced data classification','Machine Learning 113, 7421–7449\n10.1007/s10994-024-06563-6'],
            ['Sağlam & Cengiz (2026)','Uncertainty guided oversampling: A classifier-specific approach for effective handling of class imbalance in machine learning','Knowledge-Based Systems 342, 115886\n10.1016/j.knosys.2026.115886']]),
        ('Datasets and software',[
            ['Alcalá-Fdez et al. (2009)','KEEL: A software tool to assess evolutionary algorithms for data mining problems','Soft Computing 13, 307–318\n10.1007/s00500-008-0323-y'],
            ['Lemaître, Nogueira & Aridas (2017)','Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning','JMLR 18(17), 1–5\njmlr.org/papers/v18/16-365.html'],
            ['Pedregosa et al. (2011)','Scikit-learn: Machine Learning in Python','JMLR 12, 2825–2830\njmlr.org/papers/v12/pedregosa11a.html']])
    ]
    entries=[row for _,rows in references for row in rows]
    index=next(i for i,s in enumerate(slides) if s['kind']=='closing')
    for i in range(2):
        slide=dict(kind='references',title=f'References {i+1}/2',section='References',source='latex_paper/references.bib',entries=entries[i*6:(i+1)*6],notes='These are the sources used in our paper for the methods, related work, data and software. There is no need to read the full list aloud. The source details are here for anyone who wants to look up a study.',take='References from the research paper bibliography.')
        slides.insert(index+i,slide)

    slides[:]=[s for s in slides if s['title'] not in {'Variant formulas: relaxed, edited and ROS','The three extensions: what changes?'}]

    slides[:]=[s for s in slides if s['title'] != 'Local measures: oᵢ, sᵢ, dᵢ and qᵢ']
