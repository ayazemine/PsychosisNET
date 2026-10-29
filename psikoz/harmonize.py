"""ComBat harmonisation of connectivity values across sites (Johnson et al. 2007, parametric
empirical Bayes), written so that it can be fitted on the training part of a fold only.

Site location and scale are estimated with age, sex and the class label in the model, so that
they are not biased by different case/control ratios between sites. When the fitted model is
applied, only the age and sex effects are protected: the label is never needed for a new
subject, and the group difference simply stays in the data.
"""
import numpy as np
import torch


class ComBat:
    def fit(self, x, site, preserve, protect=None):
        """x (n, p) features, site (n,) ints, preserve (n, k) covariates known for every subject,
        protect (n, m) covariates known for the fitting subjects only (class indicators)."""
        self.sites = np.unique(site)
        onehot = (site[:, None] == self.sites[None]).astype(float)
        assert onehot.sum(0).min() >= 3, "every site needs at least 3 training subjects"
        self.center = preserve.mean(0)
        cov = preserve - self.center
        if protect is not None:
            cov = np.hstack([cov, protect])
        design = np.hstack([onehot, cov])
        assert np.linalg.matrix_rank(design) == design.shape[1], "site and covariates are collinear"
        beta = np.linalg.lstsq(design, x, rcond=None)[0]
        k, n_site = len(self.sites), onehot.sum(0)
        self.alpha = (n_site / len(x)) @ beta[:k]
        self.beta = beta[k:k + preserve.shape[1]]
        resid = x - design @ beta
        self.sd = np.sqrt((resid ** 2).mean(0)).clip(1e-8)
        z = (resid + onehot @ beta[:k] - self.alpha) / self.sd

        self.gamma, self.delta = np.zeros((k, x.shape[1])), np.ones((k, x.shape[1]))
        for i in range(k):
            zi = z[onehot[:, i] > 0]
            n, g_hat, d_hat = len(zi), zi.mean(0), zi.var(0, ddof=1)
            g_bar, t2 = g_hat.mean(), g_hat.var(ddof=1)
            m, s2 = d_hat.mean(), d_hat.var(ddof=1)
            a, b = (2 * s2 + m ** 2) / s2, (m * s2 + m ** 3) / s2       # inverse-gamma prior
            g, d = g_hat.copy(), d_hat.copy()
            for _ in range(200):
                g_new = (t2 * n * g_hat + d * g_bar) / (t2 * n + d)
                d_new = (0.5 * ((zi - g_new) ** 2).sum(0) + b) / (n / 2 + a - 1)
                change = max(np.abs(g_new - g).max(), np.abs(d_new - d).max())
                g, d = g_new, d_new
                if change < 1e-6:
                    break
            self.gamma[i], self.delta[i] = g, d
        return self

    def transform(self, x, site, preserve):
        i = np.searchsorted(self.sites, site)
        assert (self.sites[i.clip(max=len(self.sites) - 1)] == site).all(), "site not seen during fitting"
        fixed = self.alpha + (preserve - self.center) @ self.beta
        z = (x - fixed) / self.sd
        return self.sd * (z - self.gamma[i]) / np.sqrt(self.delta[i]) + fixed


def harmonize_loaders(loaders):
    """Fit ComBat on the training part of a fold and replace the matrices of all parts in place."""
    def arrays(ds):
        x = torch.stack([it[0][ds.triu[0], ds.triu[1]] for it in ds.items]).double().numpy()
        age = ds.age.double().numpy()
        return x, ds.site.numpy(), np.column_stack([age, ds.sex.double().numpy()])

    train = loaders["train"].dataset
    x, site, cov = arrays(train)
    fill = np.nanmean(cov, 0)                                            # missing age -> training mean
    cov = np.where(np.isnan(cov), fill, cov)
    protect = torch.nn.functional.one_hot(train.y)[:, 1:].double().numpy()
    combat = ComBat().fit(x, site, cov, protect)
    for loader in loaders.values():
        ds = loader.dataset
        x, site, cov = arrays(ds)
        h = torch.from_numpy(combat.transform(x, site, np.where(np.isnan(cov), fill, cov))).float()
        for j in range(len(ds.items)):
            fc = torch.zeros_like(ds.items[j][0])
            fc[ds.triu[0], ds.triu[1]] = h[j]
            ds.items[j] = (fc + fc.T,)
    return combat


def tangent_loaders(loaders):
    """Replace the correlation matrices of all parts by tangent-space matrices (Varoquaux et al.
    2010) whose reference point is the mean covariance of the training part of the fold.
    Returns the fitted reference and scale, so that they can be saved with the model."""
    from nilearn.connectome import ConnectivityMeasure

    train = tangent_series(loaders["train"].dataset)
    cm = ConnectivityMeasure(kind="tangent").fit(train)
    scale = cm.transform(train).std()
    apply_tangent(loaders, cm, scale)
    return cm, scale


def apply_tangent(loaders, cm, scale):
    """Tangent matrices with an already fitted reference (psikoz.evaluate: saved classical models)."""
    for loader in loaders.values():
        ds = loader.dataset
        ds.items = [(torch.tensor(m / scale, dtype=torch.float32),) for m in cm.transform(tangent_series(ds))]


def tangent_series(ds):
    out = []
    for f in ds.index["file"]:
        z = np.load(ds.cfg["root"] / f)
        out.append(z["timeseries"][z["frame_mask"]].astype(float))
    return out


def correlation_loaders(loaders):
    """Fisher-z matrices -> correlation matrices with unit diagonal (input of the SPD layer)."""
    for loader in loaders.values():
        ds = loader.dataset
        ds.items = [(torch.tanh(it[0]).fill_diagonal_(1.0),) for it in ds.items]


def view_loaders(loaders, views):
    """Stack extra matrices behind the connectivity matrix of every item: (R, R) -> (1 + V, R, R).

      slow5 / slow4   tangent-space covariance of the 0.010-0.027 / 0.027-0.073 Hz band
                      (reference: training part of the fold)
      lag1            lag-1 cross-correlation, split into its symmetric and antisymmetric part

    Every view is divided by its standard deviation over the training part.
    """
    from nilearn.connectome import ConnectivityMeasure
    from scipy import signal

    bands = {"slow5": (0.010, 0.027), "slow4": (0.027, 0.073)}

    def subject(ds):
        out = []
        for f in ds.index["file"]:
            z = np.load(ds.cfg["root"] / f)
            ts, good = z["timeseries"].astype(float), z["frame_mask"]
            row = {}
            for name in set(views) & set(bands):
                sos = signal.butter(2, bands[name], btype="band", fs=1 / float(z["tr"]), output="sos")
                row[name] = signal.sosfiltfilt(sos, ts, axis=0)[good]
            if "lag1" in views:
                pair = good[1:] & good[:-1]
                a, b = ((v - v.mean(0)) / np.maximum(v.std(0), 1e-8) for v in (ts[:-1][pair], ts[1:][pair]))
                lag = a.T @ b / max(len(a), 1)
                row["lag1"] = np.stack([(lag + lag.T) / 2, (lag - lag.T) / 2])
            out.append(row)
        return out

    data = {part: subject(loader.dataset) for part, loader in loaders.items()}
    mats = {part: [] for part in loaders}
    for name in views:
        if name in bands:
            cm = ConnectivityMeasure(kind="tangent").fit([r[name] for r in data["train"]])
            m = {part: cm.transform([r[name] for r in rows])[:, None] for part, rows in data.items()}
        else:
            m = {part: np.stack([r[name] for r in rows]) for part, rows in data.items()}
        scale = m["train"].std(axis=(0, 2, 3), keepdims=True)
        for part in loaders:
            mats[part].append(m[part] / scale)
    for part, loader in loaders.items():
        ds = loader.dataset
        extra = torch.tensor(np.concatenate(mats[part], 1), dtype=torch.float32)
        ds.items = [(torch.cat([it[0][None], extra[j]]),) for j, it in enumerate(ds.items)]
