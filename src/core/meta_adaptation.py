"""Fit the surrogate ensemble with a bounded GP training set."""
import numpy as np

def build_with_cap(data,prior,seed=0,cap=256):
    """GP-only thinning within original scopes; full-prefix standardizer."""
    import cs2fs
    data=np.asarray(data,float)
    if int(cap)>=len(data):
        models,audit=cs2fs.build_heterogeneous_pool(data,prior,seed)
        audit.update(gpr_training_cap=int(cap),training_rows={
            n:int(m.scope["fit_sample_count"]) for n,m in models.items()})
        return models,audit
    x,y=data[:,:-1],data[:,-1]
    xm,xs=np.mean(x,axis=0),np.maximum(np.std(x,axis=0),1e-6)
    ym,ys=float(np.mean(y)),max(float(np.std(y)),1e-6)
    xn,yn=(x-xm)/xs,(y-ym)/ys
    models,scopes,omitted,seen,counts={},[],[],set(),{}
    for name,family,limit in cs2fs.CANONICAL_POOL:
        count=len(x) if limit is None else min(limit,len(x))
        start=len(x)-count
        signature=(family,start,count)
        if signature in seen:
            omitted.append({"model":name,"reason":"duplicate_family_and_data_scope"})
            continue
        seen.add(signature)
        rows=np.arange(start,len(x))
        if family=="GPR" and len(rows)>int(cap):
            rows=rows[np.linspace(0,len(rows)-1,int(cap),dtype=int)]
        if family=="GPR":
            p=prior["gpr_prior"]
            model=cs2fs.GPRModel(p["length_multiplier"],p["noise"]).fit(xn[rows],yn[rows])
        else:
            p=prior["rbf_prior"]
            model=cs2fs.RBFModel(np.asarray(prior["rbf_centers"]),p["width"],p["ridge"],
                                 np.asarray(p["weight_prior"])).fit(xn[rows],yn[rows])
        scope={"model_name":name,"data_scope":"all_visible" if limit is None else f"recent_{limit}",
               "fit_start_index":start,"fit_sample_count":len(rows),"scope_sample_count":count,
               "training_row_indices":rows.tolist(),"causally_visible_only":True}
        models[name]=cs2fs.ScaleAdapter(model,xm,xs,ym,ys,scope)
        scopes.append(scope)
        counts[name]=len(rows)
    return models,{"active_pool":sorted(models),"active_pool_size":len(models),
        "canonical_pool":[v[0] for v in cs2fs.CANONICAL_POOL],"fit_scopes":scopes,
        "omitted_duplicate_slots":omitted,"training_rows":counts,"gpr_training_cap":int(cap),
        "standardizer_sample_count":len(data),"future_values_used":False}
