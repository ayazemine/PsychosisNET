"""Unpack SRPBS_OPEN.tar.gz (one file per fMRI volume) into a BIDS-like layout in one pass.

    python scripts/unpack_srpbs.py

    SRPBS_OPEN/data/sub-0001/rsfmri/vol_001 ...   -> sub-0001/func/sub-0001_task-rest_bold.nii.gz (4-D)
    SRPBS_OPEN/data/sub-0001/t1/defaced_mprage.nii -> sub-0001/anat/sub-0001_T1w.nii.gz
    SRPBS_OPEN/data/sub-0001/fmap/*                -> sub-0001/fmap/*
    everything else (tables, documents)           -> same relative path

The archive is read as a stream and stays in place as the untouched original. Subjects whose
outputs exist are skipped, so the script can be restarted.
"""
import gzip
import re
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "srpbs"
ARCHIVE = ROOT / "SRPBS_OPEN.tar.gz"
MEMBER = re.compile(r"^SRPBS_OPEN/data/(sub-\d+)/(rsfmri|t1|fmap)/(.+)$")
slots = threading.Semaphore(12)          # subjects held in memory at once


def write_subject(sub, files):
    try:
        vols = sorted((n, b) for (kind, n), b in files.items() if kind == "rsfmri")
        if vols:
            imgs = [nib.Nifti1Image.from_bytes(b) for _, b in vols]
            first = imgs[0]
            assert all(i.shape == first.shape and np.allclose(i.affine, first.affine, atol=1e-3) for i in imgs), "volumes differ"
            data = np.stack([np.asanyarray(i.dataobj) for i in imgs], -1)
            if not np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float32)
            out = ROOT / sub / "func" / f"{sub}_task-rest_bold.nii.gz"
            out.parent.mkdir(parents=True, exist_ok=True)
            img = nib.Nifti1Image(data, first.affine)
            img.header.set_zooms(first.header.get_zooms()[:3] + (1.0,))      # TR is not in the files, see organize
            img.set_qform(first.affine, int(first.header["qform_code"]) or 1)
            img.set_sform(first.affine, int(first.header["sform_code"]) or 1)
            tmp = out.with_suffix(".tmp.nii.gz")
            nib.save(img, tmp)
            tmp.replace(out)
        for (kind, name), b in files.items():
            if kind == "t1":
                out = ROOT / sub / "anat" / (f"{sub}_T1w.nii.gz" if name == "defaced_mprage.nii" else name)
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.suffix == ".gz" and not name.endswith(".gz"):
                    with gzip.open(out, "wb", compresslevel=1) as f:
                        f.write(b)
                else:
                    out.write_bytes(b)
            elif kind == "fmap":
                out = ROOT / sub / "fmap" / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b)
        (ROOT / sub / ".done").write_text(str(len(vols)))
        return sub, len(vols), None
    except Exception as e:
        return sub, 0, repr(e)
    finally:
        slots.release()


def main():
    pending, done, failed = [], 0, []
    current, files, seen = None, {}, set()

    def flush(pool):
        nonlocal current, files
        if current is not None and files:
            slots.acquire()
            pending.append(pool.submit(write_subject, current, files))
        current, files = None, {}

    with ThreadPoolExecutor(10) as pool, tarfile.open(ARCHIVE, "r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            hit = MEMBER.match(m.name)
            if not hit:
                out = ROOT / m.name.split("/", 1)[1]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tar.extractfile(m).read())
                print("extra file:", out.relative_to(ROOT), flush=True)
                continue
            sub, kind, name = hit.groups()
            if sub != current:
                flush(pool)
                assert sub not in seen, f"{sub} is not contiguous in the archive"
                seen.add(sub)
                current = sub
                if len(seen) % 25 == 0:
                    for f in [f for f in pending if f.done()]:
                        pending.remove(f)
                        s, n, err = f.result()
                        done += 1
                        if err:
                            failed.append(s)
                            print(f"{s} FAILED: {err}", flush=True)
                    print(f"{len(seen)} subjects read, {done} written", flush=True)
            if (ROOT / sub / ".done").exists():
                continue
            files[(kind, name)] = tar.extractfile(m).read()
        flush(pool)
        for f in pending:
            s, n, err = f.result()
            if err:
                failed.append(s)
                print(f"{s} FAILED: {err}", flush=True)
    print(f"finished: {len(seen)} subjects in archive, failed: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
