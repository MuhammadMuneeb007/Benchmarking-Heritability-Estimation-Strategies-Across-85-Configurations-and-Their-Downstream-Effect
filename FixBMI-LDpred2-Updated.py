from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score

ROOT = Path("body_mass_index_bmi")

print("=" * 110)
print("FIX BMI LDpred2 PRS-ONLY")
print("Target: body_mass_index_bmi/Fold_*/LDpred2-PRS-Updated/*/Results.csv")
print("Fits BMI ~ intercept + PRS on TRAIN and applies SAME model to TEST")
print("Only Train_prs_only, Test_prs_only, Delta_prs_only are changed")
print("=" * 110)


def read_fam(path):
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

    df = df[np.isfinite(df["SCORE"])].copy()

    return df


def calculate_prs_only_r2(
    train_fam,
    test_fam,
    train_profile,
    test_profile
):
    # Phenotype indexed by participant ID
    ytr = read_fam(train_fam)
    yte = read_fam(test_fam)

    # PRS indexed by participant ID
    ptr = read_profile(train_profile)
    pte = read_profile(test_profile)

    # Correct ID-based matching
    tr = ytr.merge(
        ptr,
        on=["FID", "IID"],
        how="inner",
        validate="one_to_one"
    )

    te = yte.merge(
        pte,
        on=["FID", "IID"],
        how="inner",
        validate="one_to_one"
    )

    if len(tr) < 3:
        raise ValueError(
            f"Too few TRAIN samples after matching: {len(tr)}"
        )

    if len(te) < 2:
        raise ValueError(
            f"Too few TEST samples after matching: {len(te)}"
        )

    # ------------------------------------------
    # Fit ONLY on TRAIN:
    #
    # BMI = intercept + beta * PRS
    # ------------------------------------------

    Xtr = sm.add_constant(
        tr[["SCORE"]].astype(float),
        has_constant="add"
    )

    model = sm.OLS(
        tr["BMI"].astype(float).values,
        Xtr
    ).fit()

    # ------------------------------------------
    # Apply SAME training model to TEST
    # ------------------------------------------

    Xte = sm.add_constant(
        te[["SCORE"]].astype(float),
        has_constant="add"
    )

    pred_train = model.predict(Xtr)
    pred_test = model.predict(Xte)

    train_r2 = r2_score(
        tr["BMI"].astype(float).values,
        pred_train
    )

    test_r2 = r2_score(
        te["BMI"].astype(float).values,
        pred_test
    )

    score_sd_train = float(tr["SCORE"].std())
    score_sd_test = float(te["SCORE"].std())

    beta = float(
        model.params["SCORE"]
        if "SCORE" in model.params.index
        else np.nan
    )

    intercept = float(
        model.params["const"]
        if "const" in model.params.index
        else np.nan
    )

    return {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "n_train": len(tr),
        "n_test": len(te),
        "beta": beta,
        "intercept": intercept,
        "score_sd_train": score_sd_train,
        "score_sd_test": score_sd_test,
    }


# ============================================================
# FIND ONLY REAL BMI LDpred2-PRS-Updated RESULTS
# ============================================================

results_files = sorted(
    ROOT.glob(
        "Fold_*/LDpred2-PRS-Updated/*/Results.csv"
    )
)

print(f"\nFound {len(results_files)} LDpred2 Results.csv files.\n")

if not results_files:
    print("ERROR: No files found.")
    print(
        "Expected something like:\n"
        "body_mass_index_bmi/Fold_0/"
        "LDpred2-PRS-Updated/<configuration>/Results.csv"
    )
    raise SystemExit(1)


updated_files = 0
updated_rows = 0
failed = []


for result_file in results_files:

    try:
        df = pd.read_csv(result_file)

        if len(df) == 0:
            continue

        # Results.csv row i corresponds to grid_i
        # in the LDpred2 scripts.
        fold_dir = result_file.parents[2]

        train_fam = fold_dir / "train_data.QC.fam"
        test_fam = fold_dir / "test_data.fam"

        if not train_fam.exists():
            failed.append(
                (str(result_file), f"Missing {train_fam}")
            )
            continue

        if not test_fam.exists():
            failed.append(
                (str(result_file), f"Missing {test_fam}")
            )
            continue

        changed = 0

        for i in range(len(df)):

            train_profile = (
                result_file.parent /
                f"train_grid_{i}.profile"
            )

            test_profile = (
                result_file.parent /
                f"test_grid_{i}.profile"
            )

            if (
                not train_profile.exists()
                or not test_profile.exists()
            ):
                continue

            res = calculate_prs_only_r2(
                train_fam,
                test_fam,
                train_profile,
                test_profile
            )

            old_train = float(
                df.at[i, "Train_prs_only"]
            )

            old_test = float(
                df.at[i, "Test_prs_only"]
            )

            df.at[i, "Train_prs_only"] = (
                res["train_r2"]
            )

            df.at[i, "Test_prs_only"] = (
                res["test_r2"]
            )

            df.at[i, "Delta_prs_only"] = (
                res["train_r2"] -
                res["test_r2"]
            )

            changed += 1

            print(
                f"{result_file.parent.name} "
                f"grid={i:02d} "
                f"Ntrain={res['n_train']} "
                f"Ntest={res['n_test']} "
                f"OLD=({old_train:.5f},{old_test:.5f}) "
                f"NEW=({res['train_r2']:.5f},"
                f"{res['test_r2']:.5f}) "
                f"beta={res['beta']:.6g}"
            )

        if changed == 0:
            continue

        # ------------------------------------------
        # BACKUP ORIGINAL Results.csv
        # ------------------------------------------

        backup = result_file.with_name(
            "Results.before_BMI_LDpred2_fix.csv"
        )

        if not backup.exists():
            shutil.copy2(
                result_file,
                backup
            )

        # ------------------------------------------
        # WRITE CORRECTED Results.csv
        # ------------------------------------------

        df.to_csv(
            result_file,
            index=False
        )

        # ------------------------------------------
        # Synchronise Results_Liability.csv if present
        # ------------------------------------------

        liability_file = result_file.with_name(
            "Results_Liability.csv"
        )

        if liability_file.exists():

            try:
                dl = pd.read_csv(
                    liability_file
                )

                if len(dl) == len(df):

                    for col in [
                        "Train_prs_only",
                        "Test_prs_only",
                        "Delta_prs_only",
                    ]:
                        if col in dl.columns:
                            dl[col] = df[col].values

                    liability_backup = (
                        liability_file.with_name(
                            "Results_Liability."
                            "before_BMI_LDpred2_fix.csv"
                        )
                    )

                    if not liability_backup.exists():
                        shutil.copy2(
                            liability_file,
                            liability_backup
                        )

                    dl.to_csv(
                        liability_file,
                        index=False
                    )

            except Exception as e:
                print(
                    "WARNING: could not update "
                    f"{liability_file}: {e}"
                )

        updated_files += 1
        updated_rows += changed

    except Exception as e:
        failed.append(
            (str(result_file), str(e))
        )


print("\n" + "=" * 110)
print("FINISHED BMI LDpred2 REPAIR")
print("=" * 110)

print(
    "Results.csv files found   :",
    len(results_files)
)

print(
    "Results.csv files updated :",
    updated_files
)

print(
    "LDpred2 grid rows updated :",
    updated_rows
)

print(
    "Failures                  :",
    len(failed)
)

if failed:
    print("\nFAILURES:")
    for f, e in failed:
        print(f"{f}")
        print(f"    {e}")


print("""
WHAT WAS CHANGED
----------------
YES:
  Train_prs_only
  Test_prs_only
  Delta_prs_only

NO CHANGE:
  Train_prs_cov
  Test_prs_cov
  Train_null
  Test_null
  h2
  LDpred2 weights
  grid parameters
  GCTA results
  simulations
  binary phenotypes
  other real phenotypes

METHOD
------
For every BMI LDpred2 grid:

  TRAIN:
      BMI ~ intercept + PRS

  TEST:
      prediction =
          TRAIN intercept +
          TRAIN beta * TEST PRS

  Test R2 is then calculated from those
  held-out predictions.

Negative TEST R2 values are allowed and
are NOT converted to positive numbers.
""")
