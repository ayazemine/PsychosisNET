"""Stage 3: turn every MNI-space run into network inputs.

Per run and atlas: denoised ROI time series at the common TR, a frame mask
(False = motion-censored) and a Fisher-z functional connectivity matrix.
Denoising is done on ROI means, which is equivalent to voxel-wise denoising
followed by averaging because every step is linear.
"""
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, signal, sparse
from scipy.interpolate import CubicSpline

from .config import per_dataset, rel


# ---------------------------------------------------------------- atlas
SUBCORTICAL = ("Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala", "Accumbens")


def load_atlas(name):
    """schaefer<N>: cortical parcels; schaefer<N>s: plus 14 Harvard-Oxford subcortical structures."""
    from nilearn import datasets

    if not name.startswith("schaefer"):
        raise ValueError(f"unknown atlas {name!r}")
    with_sub = name.endswith("s")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=int(name[8:].rstrip("s")), yeo_networks=7, resolution_mm=2, verbose=0)
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]
    labels = [l for l in labels if l != "Background"]
    table = pd.DataFrame({"roi": range(len(labels)), "name": labels})
    table["hemi"] = table["name"].str.split("_").str[1]
    table["network"] = table["name"].str.split("_").str[2]
    img = nib.load(atlas.maps)
    if not with_sub:
        return img, table
    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", verbose=0)
    ho_img = ho.maps if isinstance(ho.maps, nib.Nifti1Image) else nib.load(ho.maps)
    lab, ho_lab = np.asarray(img.dataobj).astype(int), np.asarray(ho_img.dataobj).astype(int)
    if ho_img.affine[0, 0] * img.affine[0, 0] < 0:              # same grid, opposite storage order along x
        ho_lab, flip = ho_lab[::-1], np.diag([-1, 1, 1, 1.0])
        flip[0, 3] = ho_lab.shape[0] - 1
        assert np.allclose(ho_img.affine @ flip, img.affine)
    else:
        assert np.allclose(ho_img.affine, img.affine)
    assert ho_lab.shape == lab.shape
    rows = []
    for value, label in enumerate(ho.labels):                   # label index = voxel value
        if label.split(" ", 1)[-1] in SUBCORTICAL:
            lab[(ho_lab == value) & (lab == 0)] = len(table) + len(rows) + 1
            rows.append(dict(roi=len(table) + len(rows), name=label.replace(" ", "_"),
                             hemi="LH" if label.startswith("Left") else "RH", network="Subcortex"))
    return nib.Nifti1Image(lab.astype(np.int16), img.affine), pd.concat([table, pd.DataFrame(rows)], ignore_index=True)


class Parcellator:
    """Area-weighted parcel averaging on grids coarser than the atlas.

    W[r, v] counts the atlas voxels of parcel r that fall inside data voxel v,
    so partially covered voxels (6 mm COBRE grid) contribute proportionally.
    """

    def __init__(self, name):
        self.img, self.table = load_atlas(name)
        self.lab = np.asarray(self.img.dataobj).astype(int)
        self.n_roi = len(self.table)
        self._cache = {}

    def weights(self, shape, affine):
        key = (tuple(shape), affine.tobytes())
        if key not in self._cache:
            ijk = np.argwhere(self.lab > 0)
            xyz = nib.affines.apply_affine(self.img.affine, ijk)
            tgt = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), xyz)).astype(int)
            ok = ((tgt >= 0) & (tgt < shape)).all(1)
            flat = np.ravel_multi_index(tgt[ok].T, shape)
            rois = self.lab[tuple(ijk[ok].T)] - 1
            self._cache[key] = sparse.coo_matrix((np.ones(ok.sum()), (rois, flat)),
                                                 shape=(self.n_roi, int(np.prod(shape)))).tocsr()
        return self._cache[key]

    def __call__(self, data, affine, valid):
        """data (X,Y,Z,T), valid (X,Y,Z) bool -> ts (T,R), coverage (R,)"""
        w = self.weights(data.shape[:3], affine)
        wv = w.multiply(valid.ravel()[None, :]).tocsr()
        total, covered = np.asarray(w.sum(1)).ravel(), np.asarray(wv.sum(1)).ravel()
        ts = (wv @ data.reshape(-1, data.shape[-1])).T / np.maximum(covered, 1e-9)
        return ts, covered / np.maximum(total, 1e-9)


def brain_voxels(data):
    """Voxels with usable signal: bright enough and not constant."""
    mean, std = data.mean(-1), data.std(-1)
    head = mean > 0.25 * np.percentile(mean, 98)
    head = ndimage.binary_fill_holes(head)
    return head & (std > 0) & (mean > 0.4 * np.median(mean[head]))


# ---------------------------------------------------------------- confounds
def feature_sets(cfg):
    """{feature set name: (atlas, denoise settings)}; variants are written to <atlas>_<variant>."""
    sets = {a: (a, cfg["denoise"]) for a in cfg["atlases"]}
    for variant, override in (cfg.get("variants") or {}).items():
        sets.update({f"{a}_{variant}": (a, {**cfg["denoise"], **override}) for a in cfg["atlases"]})
    return sets


def load_confounds(run, cfg, dn=None):
    """Returns (regressors (T,K), bad frames (T,) bool, fd (T,)) for one manifest row."""
    dn = dn or cfg["denoise"]
    thr = per_dataset(dn["fd_threshold"], run["dataset"])
    if run["dataset"] == "cobre":
        c = pd.read_csv(cfg["root"] / run["confounds"], sep="\t")
        motion = c[[f"motion_{a}" for a in ("tx", "ty", "tz", "rx", "ry", "rz")]].to_numpy()
        tissue = c[["wm_avg", "vent_avg"]].to_numpy()
        gs = c[["global_signal_pca"]].to_numpy()
        fd = c["FD"].to_numpy()
        bad = c["scrub"].to_numpy() > 0 if thr is None else fd > thr
    else:
        c = pd.read_csv(cfg["root"] / run["confounds"], sep="\t")
        motion = c[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]].to_numpy()
        tissue = c[["white_matter", "csf"]].to_numpy()
        gs = c[["global_signal"]].to_numpy()
        fd = c["framewise_displacement_filtered"].to_numpy()
        bad = fd > thr
    parts = [motion]
    if dn["motion"] >= 12:
        parts.append(np.diff(motion, axis=0, prepend=motion[:1]))
    if dn["motion"] >= 24:
        parts = parts + [p ** 2 for p in parts]
    if dn["wm_csf"]:
        parts.append(tissue)
    if dn["global_signal"]:
        parts.append(gs)
    return np.hstack(parts), bad, fd


# ---------------------------------------------------------------- signal
def _fill_bad(x, bad):
    """Linear interpolation over censored frames so the filter does not smear spikes."""
    if not bad.any():
        return x
    t = np.arange(len(x))
    out = x.copy()
    for j in range(x.shape[1]):
        out[bad, j] = np.interp(t[bad], t[~bad], x[~bad, j])
    return out


def denoise(ts, conf, bad, tr, band):
    """Censor-aware band-pass + nuisance regression. Returns z-scored (T,R)."""
    good = ~bad
    sos = signal.butter(2, band, btype="band", fs=1 / tr, output="sos")

    def prep(x):
        x = x - x[good].mean(0)
        x = signal.detrend(_fill_bad(x, bad), axis=0)
        return signal.sosfiltfilt(sos, x, axis=0)

    y, c = prep(ts), prep(conf)
    # Same filter on data and regressors (Lindquist et al. 2019); fit on clean frames only.
    c = c / np.maximum(c[good].std(0), 1e-12)
    beta = np.linalg.lstsq(c[good], y[good], rcond=None)[0]
    y = y - c @ beta
    y = y - y[good].mean(0)
    return y / np.maximum(y[good].std(0), 1e-12)


def falff(raw, conf, bad, tr, band, f_max=0.25):
    """Fractional amplitude of low-frequency fluctuations per region: spectral amplitude inside
    `band` over the amplitude up to f_max (the Nyquist frequency of the slowest acquisition)."""
    good = ~bad
    x = signal.detrend(_fill_bad(raw - raw[good].mean(0), bad), axis=0)
    c = signal.detrend(_fill_bad(conf - conf[good].mean(0), bad), axis=0)
    c = c / np.maximum(c[good].std(0), 1e-12)
    x = x - c @ np.linalg.lstsq(c[good], x[good], rcond=None)[0]
    amp, f = np.abs(np.fft.rfft(x, axis=0)), np.fft.rfftfreq(len(x), tr)
    return amp[(f >= band[0]) & (f <= band[1])].sum(0) / np.maximum(amp[(f > 0) & (f <= f_max)].sum(0), 1e-12)


def resample_time(ts, bad, tr, target_tr):
    if abs(tr - target_tr) < 1e-3:
        return ts, bad
    t_old = np.arange(len(ts)) * tr
    t_new = np.arange(0, t_old[-1] + 1e-6, target_tr)
    out = CubicSpline(t_old, ts, axis=0)(t_new)
    return out, np.interp(t_new, t_old, bad.astype(float)) > 0.25


def connectivity(ts, good, roi_valid):
    """Fisher-z Pearson matrix over clean frames; rows of invalid ROIs are zero."""
    fc = np.zeros((ts.shape[1],) * 2, dtype=np.float32)
    if good.sum() > 2:
        v = np.flatnonzero(roi_valid)
        r = np.corrcoef(ts[good][:, v].T)
        np.fill_diagonal(r, 0)
        fc[np.ix_(v, v)] = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return fc


# ---------------------------------------------------------------- driver
def attach_preprocessed(runs, cfg):
    """Point raw-dataset rows at the outputs of the preprocess stage (rows without output are dropped)."""
    from .preprocess import SPACE

    runs = runs.copy()
    keep = []
    for i, r in runs.iterrows():
        if r["dataset"] == "cobre":
            keep.append(i)
            continue
        func = cfg["paths"]["preproc"] / r["dataset"] / r["participant_id"] / "func"
        stem = f"{r['participant_id']}_task-{r['task']}_run-{r['run']}"
        bold, conf = func / f"{stem}_space-{SPACE}_desc-preproc_bold.nii.gz", func / f"{stem}_desc-confounds_timeseries.tsv"
        if bold.exists() and conf.exists():
            runs.loc[i, ["bold", "confounds"]] = rel(cfg, bold), rel(cfg, conf)
            keep.append(i)
    return runs.loc[keep]


def roi_signals(run, cfg, parcellators, drop):
    """Undenoised parcel means per atlas, cached in data/features_raw so that denoising variants
    do not have to read the 4-D images again."""
    stem = f"{run['participant_id']}_task-{run['task']}_run-{run['run']}.npz"
    cache = {a: cfg["paths"]["data"] / "features_raw" / a / run["dataset"] / stem for a in parcellators}
    if not all(f.exists() for f in cache.values()):
        img = nib.load(cfg["root"] / run["bold"])
        data = np.asarray(img.dataobj, dtype=np.float32)[..., drop:]
        valid = brain_voxels(data)
        for atlas, parc in parcellators.items():
            raw, coverage = parc(data, img.affine, valid)
            cache[atlas].parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache[atlas], raw=raw.astype(np.float32), coverage=coverage, drop=drop)
    out = {}
    for atlas, f in cache.items():
        z = np.load(f)
        assert int(z["drop"]) == drop, f"{f}: cached with other drop_first_seconds, delete data/features_raw"
        out[atlas] = z["raw"].astype(np.float64), z["coverage"]
    return out


def process_run(run, cfg, parcellators):
    target_tr = cfg["target"]["tr"]
    drop = int(round(per_dataset(cfg["denoise"]["drop_first_seconds"], run["dataset"]) / run["tr"]))
    signals = roi_signals(run, cfg, parcellators, drop)

    rows = []
    for name, (atlas, dn) in feature_sets(cfg).items():
        conf, bad, fd = (a[drop:] for a in load_confounds(run, cfg, dn))
        raw, coverage = signals[atlas]
        raw = raw.copy()
        roi_valid = coverage >= 0.5
        raw[:, ~roi_valid] = np.random.default_rng(0).standard_normal((len(raw), (~roi_valid).sum()))
        ts = denoise(raw, conf, bad, run["tr"], dn["band_hz"])
        amp = np.where(roi_valid, falff(raw, conf, bad, run["tr"], dn["band_hz"]), 0)
        ts, bad_t = resample_time(ts, bad, run["tr"], target_tr)
        ts[:, ~roi_valid] = 0
        good = ~bad_t
        out = cfg["paths"]["features"] / name / run["dataset"]
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{run['participant_id']}_task-{run['task']}_run-{run['run']}.npz"
        np.savez_compressed(f, timeseries=ts.astype(np.float32), frame_mask=good, falff=amp.astype(np.float32),
                            fc=connectivity(ts, good, roi_valid), roi_valid=roi_valid, tr=target_tr)
        clean_s = float(good.sum() * target_tr)
        rows.append(dict(atlas=name, subject=run["subject"], task=run["task"], run=run["run"],
                         file=rel(cfg, f), n_frames=len(ts), n_clean=int(good.sum()), clean_seconds=clean_s,
                         fd_mean=float(fd.mean()), fd_clean_mean=float(fd[~bad].mean()) if (~bad).any() else np.nan,
                         roi_valid_frac=float(roi_valid.mean()),
                         qc_pass=int(clean_s >= dn["min_clean_seconds"]
                                     and roi_valid.mean() >= dn["min_roi_valid_frac"])))
    return rows


def reindex(cfg):
    """Re-attach the participants table to the existing feature index (after organize changed labels)."""
    part = pd.read_csv(cfg["paths"]["manifest"] / "participants.csv")
    for atlas in feature_sets(cfg):
        f = cfg["paths"]["features"] / atlas / "index.csv"
        index = pd.read_csv(f)
        keep = ["subject"] + [c for c in index.columns if c not in part.columns]
        index[keep].merge(part, on="subject").to_csv(f, index=False)
        print(f"{atlas}: {len(index)} rows re-indexed")


_parcellators = {}


def _worker(run, cfg):
    for a in cfg["atlases"]:
        if a not in _parcellators:
            _parcellators[a] = Parcellator(a)
    return process_run(run, cfg, _parcellators)


def run(cfg, datasets=None):
    """All runs, index rewritten. With datasets: only the runs of those datasets are (re)computed and
    their rows replace those datasets' rows in every index.csv; every other file and row stays untouched
    (used to add the external site ds005237 after the classification models were trained)."""
    from concurrent.futures import ProcessPoolExecutor

    from tqdm import tqdm

    man = cfg["paths"]["manifest"]
    part = pd.read_csv(man / "participants.csv")
    runs = attach_preprocessed(pd.read_csv(man / "runs.csv"), cfg)
    runs = runs[runs["confounds"].notna()]
    if datasets:
        runs = runs[runs["dataset"].isin(datasets)]
    tables = {a: load_atlas(a)[1] for a in cfg["atlases"]}      # also downloads the atlases once

    rows = []
    with ProcessPoolExecutor(cfg["preprocess"]["n_workers"]) as pool:
        futures = [pool.submit(_worker, r, cfg) for r in runs.to_dict("records")]
        for fut in tqdm(futures, desc="features"):
            rows += fut.result()
    index = pd.DataFrame(rows).merge(part, on="subject")
    for atlas, (parcellation, _) in feature_sets(cfg).items():
        out = cfg["paths"]["features"] / atlas
        tables[parcellation].to_csv(out / "labels.csv", index=False)
        sub = index[index["atlas"] == atlas].drop(columns="atlas")
        if datasets:
            old = pd.read_csv(out / "index.csv", low_memory=False)
            sub = pd.concat([old[~old["dataset"].isin(datasets)], sub[[c for c in old.columns if c in sub.columns]]],
                            ignore_index=True)
        sub.to_csv(out / "index.csv", index=False)
        print(f"{atlas}: {len(sub)} runs, {sub['qc_pass'].sum()} pass QC")
        print(pd.crosstab([sub["dataset"], sub["diagnosis"]], sub["qc_pass"]).to_string())
