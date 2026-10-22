from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

ROOT = Path("body_mass_index_bmi")

print("=" * 110)
print("BMI PRS-ONLY R2 REPAIR")
print("Fits BMI ~ intercept + PRS in TRAIN, then applies SAME fitted model to TEST")
print("Only Train_prs_only, Test_prs_only and Delta_prs_only are changed")
print("=" * 110)


def read_fam(path):
    """Read phenotype together with FID/IID so matching is by participant ID."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1, 5],
        names=["FID", "IID", "BMI"]
    )

    df["FID"] = df["FID"].astype(str)
    df["IID"] = df["IID"].astype(str)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")

    df = df[
        np.isfinite(df["BMI"]) &
        (df["BMI"] != -9)
    ].copy()

    return df


def read_profile(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        usecols=["FID", "IID", "SCORE"]
    )

    df["FID"] = df["FID"].astype(str)
    df["IID"] = df["IID"].astype(str)
    df["SCORE"] = pd.to_numeric(df["SCORE"], errors="coerce")

    return df[np.isfinite(df["SCORE"])].copy()


def calibrated_r2(train_fam, test_fam, train_profile, test_profile):
    """
    Correct PRS-only continuous-trait evaluation:

        TRAIN: BMI ~ intercept + PRS
        TEST : use exact slope/intercept fitted in TRAIN

    Returns train R2 and held-out test R2.
    """

    ytr = read_fam(train_fam)
    yte = read_fam(test_fam)

    ptr = read_profile(train_profile)
    pte = read_profile(test_profile)

    tr = ytr.merge(ptr, on=["FID", "IID"], how="inner")
    te = yte.merge(pte, on=["FID", "IID"], how="inner")

    if len(tr) < 3 or len(te) < 2:
        raise ValueError(
            f"Too few matched samples: train={len(tr)}, test={len(te)}"
        )

    Xtr = sm.add_constant(
        tr[["SCORE"]].astype(float),
        has_constant="add"
    )

    Xte = sm.add_constant(
        te[["SCORE"]].astype(float),
        has_constant="add"
    )

    model = sm.OLS(
        tr["BMI"].astype(float).values,
        Xtr
    ).fit()

    pred_tr = model.predict(Xtr)
    pred_te = model.predict(Xte)

    train_r2 = r2_score(
        tr["BMI"].astype(float).values,
        pred_tr
    )

    test_r2 = r2_score(
        te["BMI"].astype(float).values,
        pred_te
    )

    return float(train_r2), float(test_r2), len(tr), len(te), model.params


def find_gcta_profiles(directory, pvalue):
    """
    GCTA filenames:
      train_data.pv_<threshold>.profile
      test_data.pv_<threshold>.profile
    """

    candidates = []

    for f in directory.glob("train_data.pv_*.profile"):
        m = re.search(r"\.pv_(.+)\.profile$", f.name)

        if not m:
            continue

        try:
            val = float(m.group(1))
            candidates.append((abs(val - float(pvalue)), val, f))
        except Exception:
            pass

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])
    _, _, tr = candidates[0]

    te = directory / tr.name.replace(
        "train_data.",
        "test_data."
    )

    if not te.exists():
        return None, None

    return tr, te


def update_companion_file(path, repaired):
    """
    If Results_Liability.csv exists, keep its PRS-only columns synchronized.
    BMI itself has no liability transformation.
    """
    if not path.exists():
        return

    try:
        d = pd.read_csv(path)

        if len(d) != len(repaired):
            print("  WARN companion length differs:", path)
            return

        for c in [
            "Train_prs_only",
            "Test_prs_only",
            "Delta_prs_only"
        ]:
            if c in d.columns:
                d[c] = repaired[c].values

        backup = path.with_name(
            path.stem + ".before_BMI_PRS_fix.csv"
        )

        if not backup.exists():
            shutil.copy2(path, backup)

        d.to_csv(path, index=False)

    except Exception as e:
        print("  WARN could not update companion:", path, e)


updated_files = 0
updated_rows = 0
failed = []


# ============================================================
# GCTA-SBLUP
# ============================================================

for result_file in sorted(
    ROOT.glob("Fold_*/GCTA-SBLUP-PRS/*/Results.csv")
):

    try:
        df = pd.read_csv(result_file)

        if "pvalue_threshold" not in df.columns:
            continue

        fold_dir = result_file.parents[2]

        train_fam = fold_dir / "train_data.QC.fam"
        test_fam = fold_dir / "test_data.fam"

        if not train_fam.exists() or not test_fam.exists():
            failed.append((str(result_file), "FAM missing"))
            continue

        changed = 0

        for i, row in df.iterrows():

            tr_prof, te_prof = find_gcta_profiles(
                result_file.parent,
                row["pvalue_threshold"]
            )

            if tr_prof is None:
                continue

            tr_r2, te_r2, ntr, nte, params = calibrated_r2(
                train_fam,
                test_fam,
                tr_prof,
                te_prof
            )

            old_tr = df.at[i, "Train_prs_only"]
            old_te = df.at[i, "Test_prs_only"]

            df.at[i, "Train_prs_only"] = tr_r2
            df.at[i, "Test_prs_only"] = te_r2
            df.at[i, "Delta_prs_only"] = tr_r2 - te_r2

            changed += 1

            print(
                f"GCTA {result_file.parent.name} row={i} "
                f"Ntrain={ntr} Ntest={nte} "
                f"OLD=({old_tr:.4f},{old_te:.4f}) "
                f"NEW=({tr_r2:.4f},{te_r2:.4f}) "
                f"beta={params.get('SCORE', np.nan):.6g}"
            )

        if changed:

            backup = result_file.with_name(
                "Results.before_BMI_PRS_fix.csv"
            )

            if not backup.exists():
                shutil.copy2(result_file, backup)

            df.to_csv(result_file, index=False)

            update_companion_file(
                result_file.with_name("Results_Liability.csv"),
                df
            )

            updated_files += 1
            updated_rows += changed

    except Exception as e:
        failed.append((str(result_file), str(e)))


# ============================================================
# LDPRED2
# ============================================================

ldpred_dirs = []

for pattern in [
    "Fold_*/LDpred2-PRS/*/Results.csv",
    "Fold_*/LDpred2-PRS-v2/*/Results.csv"
]:
    ldpred_dirs.extend(ROOT.glob(pattern))


for result_file in sorted(set(ldpred_dirs)):

    try:
        df = pd.read_csv(result_file)

        fold_dir = result_file.parents[2]

        train_fam = fold_dir / "train_data.QC.fam"
        test_fam = fold_dir / "test_data.fam"

        if not train_fam.exists() or not test_fam.exists():
            failed.append((str(result_file), "FAM missing"))
            continue

        changed = 0

        # Results rows were written in grid order:
        # grid_0, grid_1, ..., grid_9
        for i in range(len(df)):

            tr_prof = (
                result_file.parent /
                f"train_grid_{i}.profile"
            )

            te_prof = (
                result_file.parent /
                f"test_grid_{i}.profile"
            )

            if not tr_prof.exists() or not te_prof.exists():
                continue

            tr_r2, te_r2, ntr, nte, params = calibrated_r2(
                train_fam,
                test_fam,
                tr_prof,
                te_prof
            )

            old_tr = df.at[i, "Train_prs_only"]
            old_te = df.at[i, "Test_prs_only"]

            df.at[i, "Train_prs_only"] = tr_r2
            df.at[i, "Test_prs_only"] = te_r2
            df.at[i, "Delta_prs_only"] = tr_r2 - te_r2

            changed += 1

            print(
                f"LDpred2 {result_file.parent.name} grid={i} "
                f"Ntrain={ntr} Ntest={nte} "
                f"OLD=({old_tr:.4f},{old_te:.4f}) "
                f"NEW=({tr_r2:.4f},{te_r2:.4f}) "
                f"beta={params.get('SCORE', np.nan):.6g}"
            )

        if changed:

            backup = result_file.with_name(
                "Results.before_BMI_PRS_fix.csv"
            )

            if not backup.exists():
                shutil.copy2(result_file, backup)

            df.to_csv(result_file, index=False)

            update_companion_file(
                result_file.with_name("Results_Liability.csv"),
                df
            )

            updated_files += 1
            updated_rows += changed

    except Exception as e:
        failed.append((str(result_file), str(e)))


print("\n" + "=" * 110)
print("FINISHED")
print("=" * 110)
print("Results.csv files updated :", updated_files)
print("PRS grid rows updated     :", updated_rows)
print("Failures                  :", len(failed))

if failed:
    print("\nFAILURES:")
    for f, e in failed[:30]:
        print(f, "->", e)

print("""
IMPORTANT:
  * Only BMI PRS-only metrics were changed.
  * Full-model metrics were NOT changed.
  * Null-model metrics were NOT changed.
  * Heritability values were NOT changed.
  * Binary phenotypes were NOT changed.
  * Original Results.csv files were backed up as:
        Results.before_BMI_PRS_fix.csv
""")
