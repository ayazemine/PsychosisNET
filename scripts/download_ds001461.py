"""Download OpenNeuro ds001461 (resting-state fMRI, schizophrenia x smoking, Chongqing) from the public S3 bucket.

The dataset has one resting-state run per subject and no anatomical images.
Re-running resumes: complete files are kept.
"""
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
OUT = Path(__file__).resolve().parents[1] / "datasets" / "ds001461"
WANT = re.compile(r"^ds001461/[^.]")


def listing(prefix):
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        for key, size in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", xml):
            yield key, int(size)
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return
        token = m[1]


def fetch(item):
    key, size = item
    dst = OUT / key.split("/", 1)[1]
    if dst.exists() and dst.stat().st_size == size:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", tmp)
            if tmp.stat().st_size == size:
                tmp.replace(dst)
                return size
        except Exception as e:
            err = e
    raise RuntimeError(f"{key}: download failed ({err if 'err' in dir() else 'size mismatch'})")


if __name__ == "__main__":
    files = [(k, s) for k, s in listing("ds001461/") if WANT.match(k) and "/.datalad/" not in k]
    print(f"{len(files)} files, {sum(s for _, s in files) / 1e9:.1f} GB -> {OUT}", flush=True)
    failed, done = [], 0
    with ThreadPoolExecutor(16) as pool:
        futures = {pool.submit(fetch, f): f[0] for f in files}
        for i, (fut, key) in enumerate(futures.items(), 1):
            try:
                done += fut.result()
            except Exception as e:
                failed.append(key)
                print("FAILED", e, flush=True)
            if i % 100 == 0:
                print(f"{i}/{len(files)}  {done / 1e9:.1f} GB new", flush=True)
    print(f"finished: {len(files) - len(failed)} ok, {len(failed)} failed", flush=True)
    sys.exit(1 if failed else 0)
