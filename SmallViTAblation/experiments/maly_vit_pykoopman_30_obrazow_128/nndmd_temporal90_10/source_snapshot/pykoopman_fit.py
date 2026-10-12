"""Weighted affine dynamics in original channels, solved by PyKoopman EDMD."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
from threadpoolctl import threadpool_limits


def build_pairs(latent, coverage, tool, indices, last):
    """Explicit same-image, same-token pairs, endpoints no later than last."""
    assert last > 0 and last < len(latent)
    xs, ys, ws, counts = [], [], [], []
    for image in indices:
        token_ids = np.flatnonzero(tool[image] & (coverage[image] > 0))
        assert len(token_ids) > 0
        cw = np.asarray(coverage[image, token_ids], dtype=np.float64)
        cw /= cw.sum()
        xs.append(np.asarray(latent[:last, image][:, token_ids], dtype=np.float64).reshape(-1, latent.shape[-1]))
        ys.append(np.asarray(latent[1:last+1, image][:, token_ids], dtype=np.float64).reshape(-1, latent.shape[-1]))
        ws.append(np.tile(cw, last) / (last * len(indices)))
        counts.append({"image": int(image), "tokens": len(token_ids), "pairs": last*len(token_ids), "total_weight": 1/len(indices)})
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()
    assert np.all(w > 0) and abs(w.sum()-1) < 1e-12
    return x, y, w, {"indices": list(map(int, indices)), "last_checkpoint_index": last,
                     "last_source_step": last*10, "image_time_transitions": last*len(indices),
                     "real_token_pairs": len(x), "per_image": counts, "total_weight": float(w.sum())}


def fit_model(x, y, w, lam):
    """All coefficients are learned by the real package EDMD, full rank."""
    import pykoopman as pk
    assert x.shape == y.shape and len(w) == len(x) and lam >= 0
    assert abs(w.sum()-1) < 1e-10 and np.all(w > 0)
    d = x.shape[1]
    scale = np.sqrt(w[:, None])
    xx = np.c_[x, np.ones(len(x))] * scale
    yy = np.c_[y, np.ones(len(y))] * scale
    alpha = float(lam * np.sum(w[:,None]*x*x) / d)
    if alpha > 0:
        pseudo = np.c_[np.eye(d)*np.sqrt(alpha), np.zeros(d)]
        xx = np.vstack([xx, pseudo]); yy = np.vstack([yy, pseudo])
    package = pk.Koopman(observables=pk.observables.Identity(),
                         regressor=pk.regression.EDMD(svd_rank=-1, tlsq_rank=0))
    with threadpool_limits(limits=4):
        package.fit(xx, y=yy, dt=10)
        fitted = package._pipeline.steps[-1][1]
        basis = np.asarray(fitted.ur)
        internal = np.asarray(fitted.state_matrix_)
        operator = np.real_if_close(basis @ internal @ basis.T)
        eigenvalues = np.linalg.eigvals(operator[:d,:d])
        singular_values = np.linalg.svd(xx, compute_uv=False)
        # Independent orientation check uses the package predict, never a solver.
        verify = np.c_[x[:min(100,len(x))], np.ones(min(100,len(x)))]
        error = float(np.max(np.abs(package.predict(verify)-verify@operator.T)))
    assert operator.shape == (d+1,d+1) and internal.shape == (d+1,d+1)
    assert not np.iscomplexobj(operator) and np.isfinite(operator).all()
    assert error < 1e-9
    constant_error = float(np.max(np.abs(operator[-1]-np.r_[np.zeros(d),1])))
    assert constant_error < 1e-8, f"Constant feature not preserved: {constant_error}"
    affine = operator.copy(); affine[-1] = np.r_[np.zeros(d),1]
    return {"A": operator[:d,:d].copy(), "b":operator[:d,-1].copy(),
            "operator_raw":operator, "operator_affine":affine,
            "internal_basis":basis, "internal_operator":internal, "eigenvalues_A":eigenvalues,
            "singular_values_design": singular_values,
            "diagnostics": {"lambda":float(lam), "alpha":alpha, "features_original":d,
                            "features_augmented":d+1, "rank_retained":len(singular_values),
                            "real_pairs":len(x), "synthetic_pairs":d if alpha>0 else 0,
                            "constant_row_maxabs_error":constant_error,
                            "package_predict_orientation_maxabs":error,
                            "spectral_radius_A":float(np.max(np.abs(eigenvalues))),
                            "condition_number_design":float(singular_values[0]/singular_values[-1]),
                            "regression_class":"pykoopman.regression.EDMD", "svd_rank":-1,
                            "tlsq_rank":0, "observable":"Identity on explicitly weighted [z,1]"}}


def forecast(model, start, n_steps):
    """Open loop: start is the only observed state accepted by this function."""
    z = np.asarray(start, dtype=np.float64).copy()
    result = np.empty((n_steps,)+z.shape, dtype=np.float64)
    with threadpool_limits(limits=4):
        for step in range(n_steps):
            z = z @ model["A"].T + model["b"]
            result[step] = z
    assert np.isfinite(result).all()
    return result


def metrics_one(origin, truth, prediction, token_weights, majority):
    """Coverage-weighted metrics on cumulative motion, one image/horizon."""
    dp = prediction-origin; dt = truth-origin
    d = dt.shape[-1]; w = token_weights[:,None] / d
    error = prediction-truth
    mse = float(np.sum(w*error*error))
    baseline = float(np.sum(w*dt*dt))
    result = {"mse_state":mse, "mse_motion":float(np.sum(w*(dp-dt)**2)),
              "mse_persistence":baseline,
              "mse_ratio_vs_persistence":mse/baseline if baseline>0 else None,
              "mse_skill_vs_persistence":1-mse/baseline if baseline>0 else None}
    for eps, suffix in [(1e-8,""), (1e-3,"_eps1e_3")]:
        active = np.abs(dt)>eps
        active_mass = float(np.sum(w*active))
        correct = (np.sign(dp)==np.sign(dt)) & active
        basecorrect = (majority[None,:]==np.sign(dt)) & active
        result["active_component_mass"+suffix] = active_mass
        result["sign_accuracy"+suffix] = float(np.sum(w*correct))/active_mass if active_mass>0 else None
        result["majority_sign_accuracy"+suffix] = float(np.sum(w*basecorrect))/active_mass if active_mass>0 else None
    nt = np.linalg.norm(dt,axis=1); npred = np.linalg.norm(dp,axis=1)
    valid = (nt>1e-12)&(npred>1e-12)
    cmass = float(token_weights[valid].sum())
    cosine = np.sum(dp[valid]*dt[valid],axis=1)/(nt[valid]*npred[valid])
    result["cosine_defined_mass"] = cmass
    result["cosine_motion"] = float(np.sum(token_weights[valid]*cosine)/cmass) if cmass>0 else None
    norm_valid = nt>1e-12; nmass = float(token_weights[norm_valid].sum())
    result["norm_defined_mass"] = nmass
    result["norm_relative_error"] = float(np.sum(token_weights[norm_valid]*np.abs(npred[norm_valid]-nt[norm_valid])/nt[norm_valid])/nmass) if nmass>0 else None
    result["motion_true_rms"] = float(np.sqrt(baseline))
    result["motion_pred_rms"] = float(np.sqrt(np.sum(w*dp*dp)))
    return result


def evaluate_group(latent, coverage, tool, indices, origin, end, model, majority, save_path=None):
    """Future observations occur only here, after a model is selected/frozen."""
    horizons = end-origin
    rows = []
    saved = np.empty((horizons,len(indices),latent.shape[2],latent.shape[3]),dtype=np.float32) if save_path else None
    for position, image in enumerate(indices):
        start_all = np.asarray(latent[origin,image], dtype=np.float64)
        pred_all = forecast(model, start_all, horizons)
        if saved is not None: saved[:,position] = pred_all.astype(np.float32)
        ids = np.flatnonzero(tool[image] & (coverage[image]>0))
        w = coverage[image,ids].astype(np.float64); w /= w.sum()
        for h in range(1,horizons+1):
            truth = np.asarray(latent[origin+h,image,ids],dtype=np.float64)
            values = metrics_one(start_all[ids],truth,pred_all[h-1,ids],w,majority)
            rows.append({"image_index":int(image),"horizon_updates":h*10,"origin_updates":origin*10,**values})
    aggregate=[]
    for h in range(1,horizons+1):
        part=[r for r in rows if r["horizon_updates"]==h*10]
        av={key:float(np.mean([r[key] for r in part if r[key] is not None]))
            if any(r[key] is not None for r in part) else None
            for key in part[0] if key not in ["image_index","horizon_updates","origin_updates"]}
        # Ratio of equal-image weighted errors is the model-selection objective.
        av["mse_ratio_vs_persistence"] = av["mse_state"]/av["mse_persistence"] if av["mse_persistence"]>0 else None
        av["mse_skill_vs_persistence"] = 1-av["mse_ratio_vs_persistence"] if av["mse_ratio_vs_persistence"] is not None else None
        aggregate.append({"horizon_updates":h*10,"origin_updates":origin*10,"images":len(indices),**av})
    if saved is not None:
        np.savez_compressed(save_path,prediction=saved,image_indices=np.asarray(indices),
                            checkpoint_steps=np.arange(origin+1,end+1)*10)
    return {"image_indices":list(map(int,indices)),"origin_updates":origin*10,
            "end_updates":end*10,"per_image_horizon":rows,"aggregate":aggregate,
            "mean_mse_ratio":float(np.mean([r["mse_ratio_vs_persistence"] for r in aggregate])),
            "mean_sign_accuracy":float(np.mean([r["sign_accuracy"] for r in aggregate]))}


def majority_sign(latent, coverage, tool, indices, origin, end):
    """A channel-wise majority sign baseline learned only from fit prefix."""
    vote=np.zeros(latent.shape[-1],dtype=np.float64)
    for image in indices:
        ids=np.flatnonzero(tool[image]&(coverage[image]>0))
        w=coverage[image,ids].astype(np.float64); w/=w.sum()
        delta=np.asarray(latent[end,image,ids],dtype=np.float64)-np.asarray(latent[origin,image,ids],dtype=np.float64)
        vote+=np.sum(w[:,None]*np.sign(delta),axis=0)/len(indices)
    return np.sign(vote),vote


def write_json(path, value):
    import json
    from pathlib import Path
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def sha256(path):
    import hashlib
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def save_model(folder, filename, model, channels):
    np.savez_compressed(folder/filename,**{k:v for k,v in model.items() if k!="diagnostics"},channels0=channels)
    write_json(folder/(filename.replace(".npz","_DIAGNOSTYKA.json")),model["diagnostics"])


def now():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()


def run(stage, root):
    import json, time, sys, importlib.metadata as metadata
    from pathlib import Path
    folder=Path(root).resolve()
    started=time.perf_counter()
    dane=json.loads((folder/"DANE.json").read_text(encoding="utf-8"))
    manifest=json.loads((folder/"MANIFEST.json").read_text(encoding="utf-8"))
    assert dane["status"]=="complete" and dane["all_finite"]
    assert manifest["fit_indices"]==list(range(8))+list(range(12,34))
    assert manifest["holdout_indices"]==[8,9,10,11]
    latent=np.load(folder/"LATENT_J128.npy",mmap_mode="r")
    assert latent.shape==(101,34,600,128) and latent.dtype==np.float32
    assert dane["checkpoint_steps"]==list(range(0,1001,10))
    masks=np.load(folder/"MASKI.npz"); coverage=masks["coverage"]; tool=masks["tool"]
    channels=np.load(folder/"KANALY_J128.npz")["channels0"]
    assert channels.shape==(128,) and len(np.unique(channels))==128
    fitids=manifest["fit_indices"]; holdids=manifest["holdout_indices"]
    internalfit=fitids[:24]; internalval=fitids[24:]
    protocol_hash=sha256(folder/"PROTOKOL_MODELU.md")
    status={"stage":stage,"started":now(),"script_sha256":sha256(__file__),"protocol_sha256":protocol_hash}
    write_json(folder/"MODEL_POSTEP.json",status)
    if stage=="fit":
        assert not (folder/"WYNIKI.json").exists(), "Completed test exists; refuse retuning. Use a new experiment."
        versions={name:metadata.version(name) for name in ["pykoopman","pydmd","numpy","scipy","scikit-learn","torch","setuptools"]}
        write_json(folder/"RUNTIME.json",{"python":sys.version,"executable":sys.executable,"versions":versions,
                                        "isolated":sys.prefix!=sys.base_prefix,"system_site_packages":False,
                                        "package_source":str(Path(metadata.distribution("pykoopman").locate_file("pykoopman"))),
                                        "sources":["https://github.com/dynamicslab/pykoopman/releases/tag/v1.2.1",
                                                   "https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmd_with_rbf.html"],
                                        "installation_repair":"setuptools80.9.0 supplies pkg_resources required by pykoopman1.2.1"})
        x,y,w,pairs=build_pairs(latent,coverage,tool,internalfit,30)
        majority,_=majority_sign(latent,coverage,tool,internalfit,0,30)
        candidates=[]
        for number,lam in enumerate([0,1e-6,1e-4,1e-2]):
            print(f"Fit candidate lambda={lam:g}; {len(x)} particle pairs",flush=True)
            candidate=fit_model(x,y,w,lam)
            val=evaluate_group(latent,coverage,tool,internalval,30,50,candidate,majority)
            write_json(folder/f"WALIDACJA_LAMBDA_{number}.json",val)
            save_model(folder,f"MODEL_WALIDACJA_{number}.npz",candidate,channels)
            candidates.append({"lambda":lam,"validation_mean_mse_ratio":val["mean_mse_ratio"],
                               "validation_mean_sign_accuracy":val["mean_sign_accuracy"],
                               "diagnostics":candidate["diagnostics"]})
            print(f"Validation ratio={val['mean_mse_ratio']:.8f}, sign={val['mean_sign_accuracy']:.6f}",flush=True)
            write_json(folder/"MODEL_POSTEP.json",{**status,"completed_candidates":len(candidates),"candidates":candidates})
        selected=min(candidates,key=lambda q:(q["validation_mean_mse_ratio"],q["lambda"]))
        selection={"selected_at":now(),"selected_lambda":selected["lambda"],"candidates":candidates,
                   "internal_fit_indices":internalfit,"internal_validation_indices":internalval,
                   "internal_fit_pairs":pairs,"fit_max_updates":300,"validation_origin_updates":300,
                   "validation_end_updates":500,"test_images_not_used":holdids,
                   "protocol_sha256":protocol_hash,"script_sha256":sha256(__file__)}
        write_json(folder/"WYBOR_MODELU.json",selection)
        del x,y,w
        x,y,w,pairs=build_pairs(latent,coverage,tool,fitids,50)
        print(f"Refit lambda={selected['lambda']:g} on30 images, prefix500, {len(x)} particle pairs",flush=True)
        model=fit_model(x,y,w,selected["lambda"])
        maj,vote=majority_sign(latent,coverage,tool,fitids,0,50)
        np.savez_compressed(folder/"BAZA_WIEKSZOSCI_ZNAKU.npz",sign=maj,vote=vote,fit_indices=fitids,
                            fit_max_updates=500,channels0=channels)
        save_model(folder,"MODEL_PREFIX_500.npz",model,channels)
        write_json(folder/"ZAMROZENIE_MODELU.json",{"frozen_at":now(),"model_file":"MODEL_PREFIX_500.npz",
                    "model_sha256":sha256(folder/"MODEL_PREFIX_500.npz"),"protocol_sha256":protocol_hash,
                    "script_sha256":sha256(__file__),"selected_lambda":selected["lambda"],"fit_pairs":pairs,
                    "forecast_origin_updates":500,"test_end_updates":1000,"test_images":holdids,
                    "no_test_future_states_read_for_fitting":True,"no_test_images_used_for_selection":True,
                    "latent_sha256_from_extraction":dane["latent_sha256"],
                    "masks_sha256":sha256(folder/"MASKI.npz"),"channels_sha256":sha256(folder/"KANALY_J128.npz")})
        print("Model frozen before final test; MODEL_PREFIX_500.npz saved",flush=True)
    elif stage=="evaluate":
        frozen=json.loads((folder/"ZAMROZENIE_MODELU.json").read_text(encoding="utf-8"))
        assert frozen["model_sha256"]==sha256(folder/frozen["model_file"])
        assert frozen["protocol_sha256"]==protocol_hash
        assert frozen["script_sha256"]==sha256(__file__), "Script changed after freezing"
        assert not (folder/"WYNIKI.json").exists(), "Final test already exists; refuse another exploratory evaluation."
        stored=np.load(folder/frozen["model_file"]); model={key:stored[key] for key in ["A","b"]}
        majority=np.load(folder/"BAZA_WIEKSZOSCI_ZNAKU.npz")["sign"]
        print("Final test on4 frozen holdout images, open-loop500..1000",flush=True)
        hold=evaluate_group(latent,coverage,tool,holdids,50,100,model,majority,folder/"PROGNOZA_HOLDOUT.npz")
        write_json(folder/"METRYKI_HOLDOUT.json",hold)
        train=evaluate_group(latent,coverage,tool,fitids,50,100,model,majority,folder/"PROGNOZA_FIT_DIAGNOSTYKA.npz")
        write_json(folder/"METRYKI_FIT_DIAGNOSTYKA.json",train)
        contiguous=0
        for row in hold["aggregate"]:
            if row["sign_accuracy"]>=.85: contiguous=row["horizon_updates"]
            else: break
        diag=json.loads((folder/"MODEL_PREFIX_500_DIAGNOSTYKA.json").read_text(encoding="utf-8"))
        result={"status":"complete","completed":now(),"model_frozen_at":frozen["frozen_at"],
                "selected_lambda":frozen["selected_lambda"],"model_shape":[128,128],"bias_shape":[128],
                "operator_augmented_shape":[129,129],"operator_time_updates":10,
                "fit_images":30,"holdout_images":4,"fit_max_updates":500,
                "fit_image_time_transitions":1500,"fit_token_pairs":frozen["fit_pairs"]["real_token_pairs"],
                "holdout_origin_updates":500,"holdout_end_updates":1000,
                "holdout_H10":hold["aggregate"][0],"holdout_H100":hold["aggregate"][9],
                "holdout_H200":hold["aggregate"][19],"holdout_H500":hold["aggregate"][-1],
                "holdout_mean_sign_accuracy":hold["mean_sign_accuracy"],
                "holdout_mean_mse_ratio":hold["mean_mse_ratio"],
                "contiguous_sign85_horizon_updates":contiguous,
                "any_horizons_sign85":[r["horizon_updates"] for r in hold["aggregate"] if r["sign_accuracy"]>=.85],
                "fit_diagnostic_H500":train["aggregate"][-1],"operator_diagnostics":diag,
                "model_sha256":frozen["model_sha256"],"protocol_sha256":protocol_hash,
                "limitations":["J128 chosen using checkpoint500; inner validation is retrospective in this basis",
                               "Holdout is only for dynamics identification, all34 images belong to detector train",
                               "Four old holdout images were used in previous investigations",
                               "Only25of30 fit images belong to exact source base1500 (3old+22new)",
                               "Single shared ViT training trajectory; correlated tokens and times are not independent replicates",
                               "No measurement between checkpoints; no per-update forecast validation"]}
        write_json(folder/"WYNIKI.json",result)
        print(json.dumps({"H10_sign":result["holdout_H10"]["sign_accuracy"],
                          "H500_sign":result["holdout_H500"]["sign_accuracy"],
                          "H500_MSE_skill":result["holdout_H500"]["mse_skill_vs_persistence"],
                          "sign85_contiguous_horizon":contiguous},indent=2),flush=True)
    elif stage=="descriptive":
        assert (folder/"WYNIKI.json").exists(), "Descriptive future fit permitted only after final test"
        selection=json.loads((folder/"WYBOR_MODELU.json").read_text(encoding="utf-8"))
        x,y,w,pairs=build_pairs(latent,coverage,tool,fitids,100)
        model=fit_model(x,y,w,selection["selected_lambda"])
        save_model(folder,"MODEL_OPISOWY_0_1000.npz",model,channels)
        write_json(folder/"MODEL_OPISOWY_INFO.json",{"purpose":"Descriptive only. NEVER used in forecasts from500.",
                     "computed_after_final_test":True,"created":now(),"pairs":pairs,
                     "lambda_inherited_from_internal_validation":selection["selected_lambda"],
                     "model_sha256":sha256(folder/"MODEL_OPISOWY_0_1000.npz")})
    else: raise ValueError(stage)
    write_json(folder/"MODEL_POSTEP.json",{**status,"completed":now(),"status":"complete","seconds":time.perf_counter()-started})


if __name__=="__main__":
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["fit","evaluate","descriptive"],required=True)
    parser.add_argument("--root",default=str(Path(__file__).parent))
    args=parser.parse_args()
    run(args.stage,args.root)
