"""Stage 2: minimal volumetric preprocessing of the raw BOLD runs (ds005073, ds000030).

Brings each run into the space COBRE is released in (ICBM152 2009a symmetric):
  motion correction to the reference -> reference-to-T1w (affine) -> T1w-to-MNI (SyN)
Datasets without anatomical images (ds001461): reference straight to the template brain (SyN).
The reference is the SBRef when the dataset has one, otherwise the temporal mean.
and writes the MNI-space BOLD plus a confounds table (motion, FD, WM, CSF, global).
No slice-timing correction and no fieldmap unwarping (needs FSL topup).
"""
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import ndimage, signal
from scipy.spatial.transform import Rotation

from .config import per_dataset

SPACE = "MNI152NLin2009aSym"


def template(cfg):
    """Full-head template, brain mask and the low-resolution output grid (cached on disk)."""
    import ants
    from nilearn import datasets

    tdir = cfg["paths"]["preproc"] / "template"
    mm = cfg["target"]["mni_voxel_mm"]
    files = {k: tdir / f"tpl-{SPACE}_{k}.nii.gz" for k in ("T1w", "brainmask")}
    files.update({k: tdir / f"tpl-{SPACE}_res-{mm:g}mm_{k}.nii.gz" for k in ("grid", "brain", "wm", "csf")})
    if not all(f.exists() for f in files.values()):
        tdir.mkdir(parents=True, exist_ok=True)
        icbm = datasets.fetch_icbm152_2009(verbose=0)
        t1, mask = ants.image_read(icbm["t1"]), ants.image_read(icbm["mask"])
        grid = ants.resample_image(t1, (mm, mm, mm), use_voxels=False, interp_type=0)
        low = lambda key: ants.resample_image_to_target(ants.image_read(icbm[key]), grid, interp_type="linear")
        brain = low("mask").numpy() > 0.5
        # Conservative tissue masks: high prior probability, WM additionally eroded.
        wm = ndimage.binary_erosion(low("wm").numpy() > 0.9) & brain
        csf = (low("csf").numpy() > 0.8) & brain
        ants.image_write(t1, str(files["T1w"]))
        ants.image_write(mask, str(files["brainmask"]))
        ants.image_write(grid, str(files["grid"]))
        for key, arr in (("brain", brain), ("wm", wm), ("csf", csf)):
            ants.image_write(grid.new_image_like(arr.astype("uint8")), str(files[key]))
    return files


def motion_table(transform_files, tr, lowpass_hz, radius):
    """Rigid parameters and Power framewise displacement from per-volume ANTs transforms."""
    import ants

    params = []
    for f in transform_files:
        p = np.asarray(ants.read_transform(f).parameters)
        rot = Rotation.from_matrix(p[:9].reshape(3, 3)).as_euler("xyz")
        params.append(np.r_[p[9:], rot])
    params = np.asarray(params)
    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

    def fd(x):
        d = np.abs(np.diff(x, axis=0, prepend=x[:1]))
        return d[:, :3].sum(1) + radius * d[:, 3:].sum(1)

    df = pd.DataFrame(params, columns=cols)
    df["framewise_displacement"] = fd(params)
    if lowpass_hz and lowpass_hz < 0.5 / tr:
        sos = signal.butter(2, lowpass_hz, btype="low", fs=1 / tr, output="sos")
        df["framewise_displacement_filtered"] = fd(signal.sosfiltfilt(sos, params, axis=0))
    else:
        df["framewise_displacement_filtered"] = df["framewise_displacement"]
    return df


def qc_figure(mean_img, tpl_t1, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting

    fig = plt.figure(figsize=(11, 3.2))
    disp = plotting.plot_epi(str(mean_img), figure=fig, cut_coords=(0, -18, 10), draw_cross=False,
                             title=title, colorbar=False)
    disp.add_edges(str(tpl_t1))
    fig.savefig(out_png, dpi=80)
    plt.close(fig)


def preprocess_subject(cfg, dataset, sub, runs):
    """runs: list of dicts from the manifest (one subject). Returns a list of QC dicts."""
    import ants

    pp, root = cfg["preprocess"], cfg["root"]
    tpl = template(cfg)
    out = cfg["paths"]["preproc"] / dataset / sub
    (out / "anat").mkdir(parents=True, exist_ok=True)
    (out / "func").mkdir(parents=True, exist_ok=True)
    tpl_t1 = ants.image_read(str(tpl["T1w"]))
    grid = ants.image_read(str(tpl["grid"]))
    masks = {k: ants.image_read(str(tpl[k])).numpy() > 0 for k in ("brain", "wm", "csf")}

    has_t1 = isinstance(runs[0].get("t1w"), str)
    if not has_t1:
        tpl_brain = tpl_t1 * ants.image_read(str(tpl["brainmask"]))
    # --- anatomy: T1w (full head, defaced) -> template (full head)
    if has_t1:
        t1 = ants.resample_image(ants.image_read(str(root / runs[0]["t1w"])), (1, 1, 1), use_voxels=False, interp_type=4)
        prefix = str(out / "anat" / f"{sub}_from-T1w_to-{SPACE}_")
        fwd = [prefix + "1Warp.nii.gz", prefix + "0GenericAffine.mat"]
        if not all(os.path.exists(f) for f in fwd):
            ants.registration(fixed=tpl_t1, moving=t1, type_of_transform=pp["t1_to_mni"], outprefix=prefix, random_seed=0)
        inv = [prefix + "0GenericAffine.mat", prefix + "1InverseWarp.nii.gz"]
        brain = ants.apply_transforms(fixed=t1, moving=ants.image_read(str(tpl["brainmask"])), transformlist=inv,
                                      whichtoinvert=[True, False], interpolator="nearestNeighbor")
        t1_brain = t1 * brain

    qc = []
    for r in runs:
        stem = f"{sub}_task-{r['task']}_run-{r['run']}"
        f_bold = out / "func" / f"{stem}_space-{SPACE}_desc-preproc_bold.nii.gz"
        f_conf = out / "func" / f"{stem}_desc-confounds_timeseries.tsv"
        f_qc = out / "func" / f"{stem}_qc.json"
        if f_bold.exists() and f_conf.exists() and f_qc.exists():
            qc.append(json.loads(f_qc.read_text()))
            continue

        bold = ants.image_read(str(root / r["bold"]))
        has_sbref = isinstance(r.get("sbref"), str)
        sbref = ants.image_read(str(root / r["sbref"])) if has_sbref else ants.get_average_of_timeseries(bold)
        tmp = tempfile.mkdtemp(prefix="mc_", dir=out / "func")
        try:
            mc = ants.motion_correction(bold, fixed=sbref, type_of_transform="BOLDRigid",
                                        outprefix=os.path.join(tmp, "v"), random_seed=0)
            conf = motion_table([m[0] for m in mc["motion_parameters"]], r["tr"],
                                per_dataset(pp["fd_lowpass_hz"], dataset), pp["fd_radius_mm"])
            if has_t1:
                coreg = ants.registration(fixed=t1_brain, moving=sbref, type_of_transform=pp["bold_to_t1"],
                                          outprefix=os.path.join(tmp, "coreg_"), random_seed=0)
                chain = fwd + coreg["fwdtransforms"]
                shutil.copy2(coreg["fwdtransforms"][0], out / "func" / f"{stem}_from-boldref_to-T1w_affine.mat")
            else:
                chain = ants.registration(fixed=tpl_brain, moving=sbref, type_of_transform=pp["bold_to_mni"],
                                          outprefix=os.path.join(tmp, "epi_"), random_seed=0)["fwdtransforms"]
            mni = ants.apply_transforms(fixed=grid, moving=mc["motion_corrected"], transformlist=chain,
                                        interpolator="linear", imagetype=3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        data = mni.numpy()
        for name, key in (("white_matter", "wm"), ("csf", "csf"), ("global_signal", "brain")):
            conf[name] = data[masks[key]].mean(0)
        conf.to_csv(f_conf, sep="\t", index=False, float_format="%.6f")

        mean = data.mean(-1)
        covered = mean > 0.25 * np.median(mean[masks["brain"]])
        dice = 2 * (covered & masks["brain"]).sum() / (covered.sum() + masks["brain"].sum())
        f_mean = out / "func" / f"{stem}_space-{SPACE}_boldref.nii.gz"
        ants.image_write(grid.new_image_like(mean.astype("float32")), str(f_mean))
        qc_figure(f_mean, tpl["T1w"], out / "func" / f"{stem}_qc.png", f"{stem}  dice={dice:.2f}")

        np.clip(data, 0, 32767, out=data)
        ants.image_write(mni.new_image_like(np.rint(data).astype("int16")), str(f_bold))
        row = dict(participant_id=sub, task=r["task"], run=r["run"], n_vols=int(data.shape[-1]),
                   fd_mean=float(conf["framewise_displacement"].mean()),
                   fd_filtered_mean=float(conf["framewise_displacement_filtered"].mean()),
                   brainmask_dice=float(dice))
        f_qc.write_text(json.dumps(row, indent=2))
        qc.append(row)
    return qc


def run(cfg, subjects=None):
    pp = cfg["preprocess"]
    runs = pd.read_csv(cfg["paths"]["manifest"] / "runs.csv")
    runs = runs[[t in per_dataset(pp["tasks"], d, []) for d, t in zip(runs["dataset"], runs["task"])]]
    if subjects:
        runs = runs[runs["participant_id"].isin(subjects)]
    groups = {key: g.to_dict("records") for key, g in runs.groupby(["dataset", "participant_id"])}
    template(cfg)  # build once before the workers start
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(pp["threads_per_worker"])

    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=min(pp["n_workers"], len(groups))) as pool:
        futures = {pool.submit(preprocess_subject, cfg, *key, rs): "/".join(key) for key, rs in groups.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            sub = futures[fut]
            try:
                rows += fut.result()
                print(f"[{i}/{len(futures)}] {sub} done", flush=True)
            except Exception as e:  # keep going, report at the end
                failed.append(sub)
                print(f"[{i}/{len(futures)}] {sub} FAILED: {e!r}", flush=True)
    # Rebuild the QC table from everything on disk so partial runs do not truncate it.
    for ddir in cfg["paths"]["preproc"].iterdir():
        all_qc = [json.loads(f.read_text()) for f in sorted(ddir.glob("sub-*/func/*_qc.json"))]
        if all_qc:
            pd.DataFrame(all_qc).to_csv(ddir / "qc.csv", index=False)
    if failed:
        raise SystemExit(f"preprocessing failed for: {failed}")
