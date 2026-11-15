"""Step 6b evaluation changes: coverage tests per quarter-hour slot, Table 1 with both models, the XGB macros,
Figure 1 with panel (b). Synthetic inputs only; figures go to tmp_path."""
import re

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from esann27 import data
from esann27 import evaluate as ev


def _synthetic_frame(n_days=60, seed=3, hourly_repeat=False):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-11-01", periods=n_days * 96, freq="15min")
    df = pd.DataFrame({"datetime_local": t.strftime("%Y-%m-%d %H:%M:%S"), "date": t.strftime("%Y-%m-%d"),
                       "hour": t.hour, "quarter": t.minute // 15})
    df["actual"] = rng.normal(0, 1, len(df))
    z = np.sort(rng.normal(0, 1.2, (len(df), 9)), axis=1)
    df[data.QCOLS] = z
    return df.sample(frac=1.0, random_state=1).reset_index(drop=True)      # order must not matter


def test_coverage_tests_per_slot():
    df = _synthetic_frame()
    k = ev.coverage_tests(df)
    assert len(k) == 2 * 96 and sorted(k["slot"].unique()) == list(range(96))
    assert (k["hour"] == k["slot"] // 4).all() and (k["quarter"] == k["slot"] % 4).all() and (k["n"] == 60).all()
    for lev in ev.LEVELS:
        s = k[k["level"] == lev].sort_values("slot")
        assert np.allclose(s["p_uc_holm"], multipletests(s["p_uc"], method="holm")[1])
        assert np.allclose(s["p_cc_holm"], multipletests(s["p_cc"], method="holm")[1])
    # one slot recomputed directly: hits of that slot over the days in chronological order
    d = df.sort_values("datetime_local")
    sel = d[(d["hour"] == 17) & (d["quarter"] == 2)]
    hit = ((sel["actual"] < sel["q10"]) | (sel["actual"] > sel["q90"])).astype(int).to_numpy()
    r = k[(k["slot"] == 70) & (k["level"] == 80)].iloc[0]
    assert r["misses"] == hit.sum() and r["LR_uc"] == pytest.approx(ev.kupiec_lr(hit.sum(), len(hit), 0.2))
    assert r["LR_ind"] == pytest.approx(ev.christoffersen_ind_lr(hit))
    rej = ev.coverage_rejections(k.assign(model="x", setting="interp", layer="raw"))
    assert len(rej) == 2 and (rej["slots"] == 96).all()
    assert rej["kupiec_rej"].tolist() == [int((k[k["level"] == L]["p_uc_holm"] < 0.05).sum()) for L in ev.LEVELS]


def test_coverage_tests_dst_slot():
    """A day without hour 2 (29 Mar 2026) leaves slots 8-11 with one observation less."""
    df = _synthetic_frame(10)
    first = df["date"].min()
    df = df[~((df["date"] == first) & (df["hour"] == 2))]
    k = ev.coverage_tests(df)
    assert (k.loc[k["hour"] == 2, "n"] == 9).all() and (k.loc[k["hour"] != 2, "n"] == 10).all()


def _metrics_all():
    rows = []
    for mi, m in enumerate(("lear", "xgb")):
        for li, lay in enumerate(("raw", "aci")):
            for si, s in enumerate(ev.SETTINGS):
                v = 10 + mi + 0.1 * li + 0.01 * si
                rows.append({"model": m, "setting": s, "layer": lay, "MAE": v, "CRPS": v - 1, "MCB": 0.5, "DSC": 2.0,
                             "PICP80": 75.0 + li * 5, "width80": 40.0})
    return pd.DataFrame(rows)


def test_table1_twelve_rows():
    dm = {("lear", "raw"): 0.0001, ("lear", "aci"): 0.002, ("xgb", "raw"): 0.04, ("xgb", "aci"): 0.5}
    tex = ev.table1_tex(_metrics_all(), dm)
    lines = tex.splitlines()
    assert lines[0] == "\\begin{tabular}{|l|l|c|c|c|c|c|c|c|}" and lines[-1] == "\\end{tabular}"
    body = [l for l in lines if l.endswith("\\\\")][1:]
    assert len(body) == 12
    assert [r.split(" & ")[1] for r in body[::3]] == ["LEAR", "LEAR+ACI", "XGB", "XGB+ACI"]
    assert tex.count("\\hline") == 2 + 4
    dms = [r.split(" & ")[-1].rstrip(" \\") for r in body]
    assert dms[2::3] == ["\\ensuremath{<}0.001", "0.002", "0.040", "0.500"]
    assert all(x == "--" for i, x in enumerate(dms) if i % 3 != 2)
    assert body[0].startswith("Hourly & LEAR & 10.00 & 9.00")
    assert "\\caption" not in tex and "\\small" not in tex


def _boot_reg(model):
    boot = pd.DataFrame([{"regime": r, "quantity": q, "diff_native_minus_interp": -0.3, "lo5": -0.4, "hi95": -0.2}
                         for r in ("raw", "aci") for q in ("CRPS", "MCB", "DSC")])
    reg = pd.DataFrame([{"regime": r, "slope": 0.4, "slope_se": 0.2, "slope_p": 0.05, "dm_p": 0.0001}
                        for r in ("raw", "aci")])
    return boot, reg


@pytest.mark.parametrize("model", ["lear", "xgb"])
def test_model_macros(model):
    met = _metrics_all()
    met.loc[(met["model"] == model) & (met["setting"] == "interp") & (met["layer"] == "raw"), "CRPS"] = 15.0
    met.loc[(met["model"] == model) & (met["setting"] == "native") & (met["layer"] == "raw"), "CRPS"] = 14.5
    met.loc[(met["model"] == model) & (met["setting"] == "interp") & (met["layer"] == "aci"), "CRPS"] = 14.9
    met.loc[(met["model"] == model) & (met["setting"] == "native") & (met["layer"] == "aci"), "CRPS"] = 14.5
    t2 = pd.DataFrame({"regime": ["split", "rolling", "acifrozen", "aci"], "PICP80": [77.04, 79.0, 79.9, 80.06],
                       "CRPS": [15.234, 14.976, 14.941, 15.049], "width80": [54.5, 56.204, 57.926, 60.584],
                       "unbounded": [0.0, 0.0, 0.0, 0.0789]})
    boot, reg = _boot_reg(model)
    m = ev.model_macros(model, met, t2, boot, reg, {s: 30.0 for s in ev.SETTINGS})
    M = ev.MODEL_LABEL[model]
    for k in ("gapCrpsRaw", "gapMcbAci", "gapDscRaw", "slopeRaw", "slopeAci", "dmPRaw", "dmPAci"):
        assert k + M in m
    assert m[f"slopeRaw{M}"] == "0.40" and m[f"slopeAci{M}P"] == "0.050" and m[f"gapCrpsRaw{M}Lo"] == "\\ensuremath{-0.40}"
    reg = {"regCrpsSplit": "15.23", "regCrpsRolling": "14.98", "regCrpsAciFrozen": "14.94", "regCrpsAci": "15.05",
           "regPicpRolling": "79.0", "regPicpAciFrozen": "79.9", "regWidthAciFrozen": "57.93",
           "regWidthAci": "60.58", "regUnboundedAci": "0.1"}
    if model == "xgb":
        assert m["hlGapRemovedXGB"] == "20.0" and m["hlPicpSplit"] == "77.0" and m["hlPicpAci"] == "80.1"
        assert not any(k.endswith("LEAR") for k in m)
        assert {k: m[k] for k in reg} == reg and sum(k.startswith("reg") for k in m) == 9
    else:
        assert m["hlGapRemovedLEAR"] == "20.0" and m["hlPicpSplitLEAR"] == "77.0"
        assert not {"hlGapRemovedXGB", "hlPicpSplit", "hlPicpAci"} & set(m) and not any("XGB" in k for k in m)
        assert not any(k.startswith("reg") for k in m)
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in m)


def _fig_inputs():
    months = pd.period_range("2025-11", "2026-09", freq="M").astype(str)
    rows = []
    for m in ("lear", "xgb"):
        for lay in ("raw", "aci"):
            for s in ev.SETTINGS:
                for i, mth in enumerate(months):
                    rows.append({"model": m, "setting": s, "layer": lay, "month": mth, "mid": f"{mth}-15",
                                 "CRPS": 15 + (0.3 + 0.02 * i if s == "interp" else 0.0)})
    mon = pd.DataFrame(rows)
    g = np.linspace(0, 1, 101)
    pred = pd.concat([pd.DataFrame({"model": m, "kind": "grid", "position": "pooled", "pi": g,
                                    "gap": 0.1 * (g - 1), "gap_interp_minus_native": 0.1 * g}) for m in ("lear", "xgb")])
    abl = pd.DataFrame({"level": ["regular", "0", "0.25", "0.5", "0.75"], "pi": [0.9629, 0, .25, .5, .75],
                        "gap_crps": [0, .4, .3, .15, .05], "PICP80": [75.2, 71.0, 72.3, 73.8, 74.6]})
    ap = pd.DataFrame({"pi": np.append(g, 0.9629), "pred_gap": np.append(0.1 * (0.9629 - g), 0.0),
                       "kind": ["grid"] * 101 + ["regular"]})
    return mon, pred, abl, ap


def _pdf_box_cm(path):
    b = path.read_bytes()
    box = re.search(rb"MediaBox \[ ?0 0 ([\d.]+) ([\d.]+) ?\]", b)
    assert b"/Type3" not in b and b"FontType 3" not in b        # pdf.fonttype 42, no Type 3 fonts
    return float(box.group(1)) / 72 * 2.54, float(box.group(2)) / 72 * 2.54


def test_figure1_sizes(tmp_path):
    mon, pred, abl, ap = _fig_inputs()
    ev.figure1(mon, pred, tmp_path / "two.pdf", abl, ap)
    ev.figure1(mon, pred, tmp_path / "one.pdf")
    assert _pdf_box_cm(tmp_path / "two.pdf") == pytest.approx((12.2, 4.6), abs=1e-3)
    assert _pdf_box_cm(tmp_path / "one.pdf") == pytest.approx((12.2, 5.0), abs=1e-3)


def test_figure2_size_legend_outside(tmp_path):
    """12.2 x 4.6 cm (rev 2.8.14); both legends below the axes, so none overlaps a data marker of (b)."""
    rng = np.random.default_rng(0)
    frames = {}
    for s in ("interp", "native"):
        for reg in ("split", "rolling", "acifrozen", "aci", "raw", "acipooled"):
            frames["xgb", s, reg] = _synthetic_frame(n_days=40, seed=int(rng.integers(1000)))
    pe = pd.DataFrame({"kind": ["eval_interp"] * 4 + ["eval_native"] * 4, "position": list(range(4)) * 2,
                       "picp80": [72, 78, 78, 72, 77, 79, 79, 77]})
    ev.figure2(frames, "xgb", pe, tmp_path / "f2.pdf")
    assert _pdf_box_cm(tmp_path / "f2.pdf") == pytest.approx((12.2, 4.6), abs=1e-3)
    plt = ev._mpl()
    import matplotlib
    orig = matplotlib.figure.Figure.savefig
    seen = {}
    matplotlib.figure.Figure.savefig = lambda self, *a, **k: seen.setdefault("fig", self)
    try:
        ev.figure2(frames, "xgb", pe, tmp_path / "unused.pdf")
    finally:
        matplotlib.figure.Figure.savefig = orig
    fig = seen["fig"]
    fig.canvas.draw()
    axes_bottom = min(ax.get_window_extent().y0 for ax in fig.axes)
    assert len(fig.legends) == 2 and all(not ax.get_legend() for ax in fig.axes)
    assert all(L.get_window_extent().y1 < axes_bottom for L in fig.legends)
    w = fig.bbox.width
    assert all(0 <= L.get_window_extent().x0 and L.get_window_extent().x1 <= w for L in fig.legends)
    la, lb = sorted(fig.legends, key=lambda L: L.get_window_extent().x0)
    assert la.get_window_extent().x1 < lb.get_window_extent().x0
    plt.close(fig)


def test_load_ablation(tmp_path):
    assert ev.load_ablation(tmp_path) == (None, None)
    _, _, abl, ap = _fig_inputs()
    abl.to_csv(tmp_path / "ablation.csv", index=False)
    ap.to_csv(tmp_path / "ablation_pred.csv", index=False)
    a, p = ev.load_ablation(tmp_path)
    assert a["level"].tolist() == ["regular", "0", "0.25", "0.5", "0.75"] and len(p) == 102
