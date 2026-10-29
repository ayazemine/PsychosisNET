"""Stage 1: organise both datasets behind one layout and one manifest.

The raw folders are never modified. COBRE (a flat folder) gets a BIDS-derivatives
style view made of hard links; ds005073 and ds000030 are already BIDS and are
indexed in place.
"""
import json
import os
import re
import shutil

import nibabel as nib
import numpy as np
import pandas as pd

from .config import rel

COBRE_SPACE = "MNI152NLin2009aSym"
COBRE_TR = 2.0  # README value; the header of fmri_0040126 wrongly says 1.0


def _link(src, dst):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def organize_cobre(cfg):
    raw, out = cfg["paths"]["cobre_raw"], cfg["paths"]["cobre_bids"]
    pheno = pd.read_csv(raw / "phenotypic_data.tsv.gz", sep="\t", keep_default_na=False)
    pheno["participant_id"] = pheno["ID"].map(lambda i: f"sub-{i:07d}")

    runs = []
    for row in pheno.itertuples():
        sub = row.participant_id
        stem = f"{sub}_task-rest"
        func = out / sub / "func"
        bold = func / f"{stem}_space-{COBRE_SPACE}_desc-preproc_bold.nii.gz"
        conf = func / f"{stem}_desc-confounds_timeseries.tsv.gz"
        _link(raw / f"fmri_{row.ID:07d}.nii.gz", bold)
        _link(raw / f"fmri_{row.ID:07d}.tsv.gz", conf)
        runs.append(dict(dataset="cobre", participant_id=sub, task="rest", run=1,
                         tr=COBRE_TR, n_vols=nib.load(bold).shape[3],
                         bold=rel(cfg, bold), confounds=rel(cfg, conf)))

    for name in ("README.md", "keys_confounds.json", "keys_phenotypic_data.json"):
        _link(raw / name, out / name)
    (out / "dataset_description.json").write_text(json.dumps({
        "Name": "COBRE preprocessed with NIAK 0.17 - lightweight release",
        "BIDSVersion": "1.8.0", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "NIAK", "Version": "0.17"}],
    }, indent=2))
    (out / "task-rest_bold.json").write_text(json.dumps(
        {"TaskName": "rest", "RepetitionTime": COBRE_TR, "SkullStripped": False}, indent=2))
    pheno.drop(columns="ID").set_index("participant_id").to_csv(out / "participants.tsv", sep="\t")

    part = pd.DataFrame({
        "dataset": "cobre",
        "participant_id": pheno["participant_id"],
        "diagnosis": pheno["Subject Type"].map({"Control": "HC", "Patient": "SZ"}),
        "diagnosis_detail": pheno["Diagnosis"],
        "psychosis": (pheno["Subject Type"] == "Patient").astype(int),
        "age": pheno["Current Age"],
        "sex": pheno["Gender"].map({"Male": "M", "Female": "F"}),
        "handedness": pheno["Handedness"].map({"Right": "R", "Left": "L", "Both": "A"}),
    })
    # Flags for rows whose DSM code does not match the group they were released in.
    is_295 = pheno["Diagnosis"].str.startswith("295")
    part["label_clean"] = ((part["diagnosis"] == "SZ") & is_295
                           | (part["diagnosis"] == "HC") & (pheno["Diagnosis"] == "None")).astype(int)
    return part, pd.DataFrame(runs)


def organize_ds005073(cfg):
    raw = cfg["paths"]["ds005073_raw"]
    pheno = pd.read_csv(raw / "participants.tsv", sep="\t", na_values=[" ", "n/a"])
    pheno = pheno.dropna(subset=["participant_id"]).reset_index(drop=True)  # trailing blank rows
    pheno[["groupID", "psychosis", "age"]] = pheno[["groupID", "psychosis", "age"]].astype(int)
    part = pd.DataFrame({
        "dataset": "ds005073",
        "participant_id": pheno["participant_id"],
        "diagnosis": pheno["groupID"].map({1: "BP", 2: "SZ"}),
        "diagnosis_detail": pheno["groupID"].map({1: "Bipolar", 2: "Schizophrenia/Schizoaffective"}),
        "psychosis": pheno["psychosis"],
        "age": pheno["age"],
        "sex": pheno["gender"],
        "handedness": pheno["handedness"],
        "label_clean": 1,
    })
    clinical = pheno.drop(columns=["groupID", "psychosis", "age", "gender", "handedness", "hasrestdata"])
    part = part.merge(clinical, on="participant_id")

    runs = []
    for sub in part["participant_id"]:
        for bold in sorted((raw / sub / "func").glob("*_bold.nii.gz")):
            m = re.search(r"task-(\w+?)_run-(\d+)_bold", bold.name)
            img = nib.load(bold)
            runs.append(dict(dataset="ds005073", participant_id=sub, task=m[1], run=int(m[2]),
                             tr=round(float(img.header.get_zooms()[3]), 4), n_vols=img.shape[3],
                             bold=rel(cfg, bold),
                             sbref=rel(cfg, bold.with_name(bold.name.replace("_bold", "_sbref"))),
                             t1w=rel(cfg, raw / sub / "anat" / f"{sub}_rec-2_T1w.nii.gz")))
    return part, pd.DataFrame(runs)


def organize_ds000030(cfg):
    raw = cfg["paths"]["ds000030_raw"]
    pheno = pd.read_csv(raw / "participants.tsv", sep="\t")
    dx = pheno["diagnosis"].map({"CONTROL": "HC", "SCHZ": "SZ", "BIPOLAR": "BP", "ADHD": "ADHD"})
    part = pd.DataFrame({
        "dataset": "ds000030",
        "participant_id": pheno["participant_id"],
        "diagnosis": dx,
        "diagnosis_detail": pheno["diagnosis"],
        "psychosis": dx.map({"HC": 0, "SZ": 1, "ADHD": 0}),
        "age": pheno["age"],
        "sex": pheno["gender"],
        "label_clean": 1,
    })
    # Bipolar group: psychosis = 1 when the SCID lists a psychotic diagnosis / specifier or the SAPS
    # global delusion or hallucination rating is at least mild (>= 2). Lifetime history is not
    # released, so bipolar participants without such evidence stay missing rather than 0.
    scid = pd.read_csv(raw / "phenotype" / "scid.tsv", sep="\t").set_index("participant_id")
    codes = scid[[f"scid_dx{i}" for i in range(1, 9)]].astype("string")
    psychotic_dx = codes.apply(lambda c: c.str.contains(r"^29[578]|With Psychotic|Psychotic Disorder", na=False)).any(axis=1)
    saps = pd.read_csv(raw / "phenotype" / "saps.tsv", sep="\t").set_index("participant_id")
    symptoms = (saps[["global_delusions", "global_hallucinations"]] >= 2).any(axis=1)
    evidence = part["participant_id"].map(psychotic_dx | symptoms).fillna(False).astype(bool)
    part.loc[(dx == "BP") & evidence, "psychosis"] = 1
    runs = []
    for sub in part["participant_id"]:
        bold = raw / sub / "func" / f"{sub}_task-rest_bold.nii.gz"
        t1w = raw / sub / "anat" / f"{sub}_T1w.nii.gz"
        if bold.exists() and t1w.exists():
            img = nib.load(bold)
            runs.append(dict(dataset="ds000030", participant_id=sub, task="rest", run=1,
                             tr=round(float(img.header.get_zooms()[3]), 4), n_vols=img.shape[3],
                             bold=rel(cfg, bold), t1w=rel(cfg, t1w)))
    return part, pd.DataFrame(runs)


def organize_ds001461(cfg):
    raw = cfg["paths"]["ds001461_raw"]
    pheno = pd.read_csv(raw / "participants.tsv", sep="	")
    dx = pheno["diagnosis"].map({"schizophrenia": "SZ", "healthy controls": "HC"})
    part = pd.DataFrame({
        "dataset": "ds001461",
        "participant_id": pheno["participant_id"],
        "diagnosis": dx,
        "diagnosis_detail": pheno["diagnosis"],
        "psychosis": (dx == "SZ").astype(int),
        "age": pheno["age"],
        "sex": pheno["sex"],
        "label_clean": 1,
        "smoking": (pheno["smoking_status"] == "smoking").astype(int),
    })
    runs = []
    for sub in part["participant_id"]:
        bold = raw / sub / "func" / f"{sub}_task-rest_bold.nii.gz"
        if bold.exists():                                       # the dataset has no anatomical images
            img = nib.load(bold)
            runs.append(dict(dataset="ds001461", participant_id=sub, task="rest", run=1,
                             tr=round(float(img.header.get_zooms()[3]), 4), n_vols=img.shape[3], bold=rel(cfg, bold)))
    return part, pd.DataFrame(runs)


def organize_ds005237(cfg):
    """Transdiagnostic Connectome Project (Yale + McLean, harmonised Prisma 3T). External test site only.

    Diagnoses come from the SCID-5 primary diagnosis (phenotype/demos.tsv, NDA format: one title row,
    comma separated, latin-1) and its "Active Psychosis" column:
      Group GenPop without any diagnosis         HC, psychosis 0
      SZ / SZA (incl. "SZ or SZA", rule-outs)    SZ, psychosis 1
      BP-I / BP-II                               BP, psychosis 1 when active psychosis is "y" or
                                                 "n (psychotic features)"; otherwise missing, because
                                                 the column does not record lifetime psychosis
      everything else (MDD, anxiety, PTSD, ...)  OTHER (outside the HC / SZ / BP sample)
    Only the first AP rest run is used (task-restAP_run-01, present for all participants).
    """
    raw = cfg["paths"]["ds005237_raw"]
    pheno = pd.read_csv(raw / "participants.tsv", sep="\t")
    demos = pd.read_csv(raw / "phenotype" / "demos.tsv", skiprows=1, encoding="latin-1", low_memory=False)
    demos["participant_id"] = "sub-" + demos["subjectkey"].astype(str).str.strip().str.replace("_", "")  # NDAR_INV... -> NDARINV...
    demos = demos.drop_duplicates("participant_id").set_index("participant_id")
    pheno = pheno.join(demos[["Primary_Dx", "Non-Primary_Dx", "Active Psychosis"]], on="participant_id")
    primary = pheno["Primary_Dx"].astype("string").str.strip()
    active = pheno["Active Psychosis"].astype("string").str.strip()
    upper = primary.str.upper()
    dx = pd.Series("OTHER", index=pheno.index)
    dx[(pheno["Group"] == "GenPop") & primary.isin(["999"])] = "HC"
    dx[(pheno["Group"] == "Patient") & upper.str.match(r"^SZ")] = "SZ"
    dx[(pheno["Group"] == "Patient") & upper.str.match(r"^BP ?(1|I|2|II)\b")] = "BP"
    psychotic = active.isin(["y", "n (psychotic features)"])
    psychosis = pd.Series(np.nan, index=pheno.index)
    psychosis[dx == "HC"] = 0
    psychosis[(dx == "SZ") | psychotic] = 1
    part = pd.DataFrame({
        "dataset": "ds005237",
        "participant_id": pheno["participant_id"],
        "diagnosis": dx,
        "diagnosis_detail": primary.fillna(""),
        "psychosis": psychosis,
        "age": pheno["age"],
        "sex": pheno["sex"],
        "label_clean": 1,
        "tcp_site": pheno["Site"],
        "active_psychosis": active,
    })
    runs = []
    for sub in part["participant_id"]:
        bold = raw / sub / "func" / f"{sub}_task-restAP_run-01_bold.nii.gz"
        t1w = raw / sub / "anat" / f"{sub}_run-01_T1w.nii.gz"
        if bold.exists() and t1w.exists():
            img = nib.load(bold)
            runs.append(dict(dataset="ds005237", participant_id=sub, task="rest", run=1,
                             tr=round(float(img.header.get_zooms()[3]), 4), n_vols=img.shape[3],
                             bold=rel(cfg, bold), t1w=rel(cfg, t1w)))
    return part, pd.DataFrame(runs)


SRPBS_DX = {0: "HC", 1: "ASD", 2: "MDD", 3: "OCD", 4: "SZ", 5: "PAIN", 6: "STROKE", 7: "BP", 8: "DYS", 99: "OTHER"}
SRPBS_TR = {1: 2.5, 2: 2.0, 3: 2.0, 4: 2.7, 5: 2.5, 6: 2.5, 7: 2.0, 8: 2.5, 9: 2.5, 10: 2.5, 11: 2.5, 12: 2.5, 14: 2.5}


def organize_srpbs(cfg):
    """SRPBS Multi-disorder MRI Dataset (unrestricted), unpacked by scripts/unpack_srpbs.py.

    Every imaging site is its own dataset ("srpbs-KUT"), because the scanners and protocols differ.
    The repetition time is not stored in the images and comes from MRI_protocols_rsMRI.tsv.
    Sites and diagnoses named in the config are the classification sample (role "main", with
    T1w). With srpbs.pretrain every other participant is listed as well, with role "pretrain" and
    without T1w: those runs take the quick registration path and are only used without labels.
    """
    raw, sel = cfg["paths"]["srpbs_raw"], cfg["srpbs"]
    pheno = pd.read_csv(raw / "participants.tsv", sep="	")
    dx = pheno["diag"].map(SRPBS_DX)
    part = pd.DataFrame({
        "dataset": "srpbs-" + pheno["site"],
        "participant_id": pheno["participant_id"],
        "diagnosis": dx,
        "diagnosis_detail": dx,
        "psychosis": dx.map({"HC": 0, "SZ": 1}),          # no psychosis information for the other groups
        "age": pheno["age"],
        "sex": pheno["sex"].map({1: "M", 2: "F"}),
        "handedness": pheno["hand"].map({1: "R", 2: "L"}),
        "label_clean": 1,
    })
    use = pheno["site"].isin(sel["sites"]) & dx.isin(sel["diagnoses"])
    part["role"] = np.where(use, "main", "pretrain")
    smoking = pd.concat([pd.read_csv(f, sep="	").set_index("participant_id").filter(like="Smoking").iloc[:, 0]
                         for f in sorted(raw.glob("sup*.tsv")) if "Smoking" in f.read_text(encoding="utf-8").splitlines()[0]])
    part["smoking"] = part["participant_id"].map(pd.to_numeric(smoking, errors="coerce").map({1: 0, 2: 1}))
    runs = []
    for row, main in zip(pheno.itertuples(), use):
        sub = row.participant_id
        bold = raw / sub / "func" / f"{sub}_task-rest_bold.nii.gz"
        t1w = raw / sub / "anat" / f"{sub}_T1w.nii.gz"
        if not bold.exists() or not (main and t1w.exists() or not main and sel.get("pretrain")):
            continue
        run = dict(dataset=f"srpbs-{row.site}", participant_id=sub, task="rest", run=1,
                   tr=SRPBS_TR[row.protocol], n_vols=nib.load(bold).shape[3], bold=rel(cfg, bold))
        if main:
            run["t1w"] = rel(cfg, t1w)
        runs.append(run)
    return part, pd.DataFrame(runs)


def run(cfg):
    out = cfg["paths"]["manifest"]
    out.mkdir(parents=True, exist_ok=True)
    parts, runs = zip(organize_cobre(cfg), organize_ds005073(cfg), organize_ds000030(cfg), organize_ds001461(cfg),
                      organize_srpbs(cfg), organize_ds005237(cfg))
    part = pd.concat(parts, ignore_index=True)
    part["role"] = part["role"].fillna("main")
    part.insert(0, "subject", part["dataset"] + "_" + part["participant_id"])
    runs = pd.concat(runs, ignore_index=True)
    runs.insert(0, "subject", runs["dataset"] + "_" + runs["participant_id"])
    part.to_csv(out / "participants.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    print(f"participants: {len(part)}  runs: {len(runs)}  -> {out}")
    print(pd.crosstab(part["dataset"], part["diagnosis"]).to_string())
    print(runs.groupby(["dataset", "task"]).size().to_string())
    return part, runs
