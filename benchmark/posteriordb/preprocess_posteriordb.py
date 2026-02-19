#!/usr/bin/env python3
"""
Preprocess posteriordb models into a standard format for eXMC validation.

For each feasible posterior (Normal linear regression + Eight Schools):
  - Extract response y and design matrix X (applying Stan transformed data)
  - Extract prior specifications
  - Extract reference posterior draws
  - Save as JSON for Elixir consumption

Usage:
  python preprocess_posteriordb.py [/path/to/posteriordb]

Outputs to: posteriordb_processed/ directory
"""

import json
import os
import sys
import zipfile
import numpy as np
from pathlib import Path


def load_json_zip(path):
    """Load a .json.zip file."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        with z.open(names[0]) as f:
            return json.load(f)


def load_json(path):
    """Load a .json file (already extracted)."""
    with open(path) as f:
        return json.load(f)


class PosteriorDBProcessor:
    def __init__(self, pdb_path):
        self.pdb_path = Path(pdb_path)
        self.data_dir = self.pdb_path / "posterior_database" / "data" / "data"
        self.model_dir = self.pdb_path / "posterior_database" / "models" / "stan"
        self.draws_dir = self.pdb_path / "posterior_database" / "reference_posteriors" / "draws" / "draws"
        self.out_dir = Path(__file__).parent / "posteriordb_processed"
        self.out_dir.mkdir(exist_ok=True)

    def load_data(self, name):
        """Load dataset by name."""
        json_path = self.data_dir / f"{name}.json"
        zip_path = self.data_dir / f"{name}.json.zip"
        if json_path.exists():
            return load_json(json_path)
        elif zip_path.exists():
            return load_json_zip(zip_path)
        else:
            raise FileNotFoundError(f"Data not found: {name}")

    def load_draws(self, posterior_name):
        """Load reference posterior draws."""
        json_path = self.draws_dir / f"{posterior_name}.json"
        zip_path = self.draws_dir / f"{posterior_name}.json.zip"
        if json_path.exists():
            return load_json(json_path)
        elif zip_path.exists():
            return load_json_zip(zip_path)
        else:
            raise FileNotFoundError(f"Draws not found: {posterior_name}")

    def flatten_draws(self, draws):
        """Flatten multi-chain draws into single arrays per parameter."""
        flat = {}
        for chain in draws:
            for param, values in chain.items():
                if param not in flat:
                    flat[param] = []
                flat[param].extend(values)
        return flat

    def process_all(self):
        """Process all feasible posteriors."""
        processors = {
            # Eight Schools (hierarchical, special case)
            "eight_schools-eight_schools_noncentered": self.process_eight_schools,

            # Earnings (7 variants)
            "earnings-earn_height": lambda: self.process_earnings("earn_height"),
            "earnings-logearn_height": lambda: self.process_earnings("logearn_height"),
            "earnings-log10earn_height": lambda: self.process_earnings("log10earn_height"),
            "earnings-logearn_height_male": lambda: self.process_earnings("logearn_height_male"),
            "earnings-logearn_interaction": lambda: self.process_earnings("logearn_interaction"),
            "earnings-logearn_interaction_z": lambda: self.process_earnings("logearn_interaction_z"),
            "earnings-logearn_logheight_male": lambda: self.process_earnings("logearn_logheight_male"),

            # KidIQ (8 variants)
            "kidiq-kidscore_momiq": lambda: self.process_kidiq("momiq"),
            "kidiq-kidscore_momhs": lambda: self.process_kidiq("momhs"),
            "kidiq-kidscore_momhsiq": lambda: self.process_kidiq("momhsiq"),
            "kidiq-kidscore_interaction": lambda: self.process_kidiq("interaction"),
            "kidiq_with_mom_work-kidscore_mom_work": lambda: self.process_kidiq_work("mom_work"),
            "kidiq_with_mom_work-kidscore_interaction_c": lambda: self.process_kidiq_work("interaction_c"),
            "kidiq_with_mom_work-kidscore_interaction_c2": lambda: self.process_kidiq_work("interaction_c2"),
            "kidiq_with_mom_work-kidscore_interaction_z": lambda: self.process_kidiq_work("interaction_z"),

            # BLR (2 variants)
            "sblrc-blr": lambda: self.process_blr("sblrc"),
            "sblri-blr": lambda: self.process_blr("sblri"),

            # Kilpisjarvi (1)
            "kilpisjarvi_mod-kilpisjarvi": self.process_kilpisjarvi,

            # Mesquite (6 variants)
            "mesquite-mesquite": lambda: self.process_mesquite("mesquite"),
            "mesquite-logmesquite": lambda: self.process_mesquite("logmesquite"),
            "mesquite-logmesquite_logva": lambda: self.process_mesquite("logmesquite_logva"),
            "mesquite-logmesquite_logvas": lambda: self.process_mesquite("logmesquite_logvas"),
            "mesquite-logmesquite_logvash": lambda: self.process_mesquite("logmesquite_logvash"),
            "mesquite-logmesquite_logvolume": lambda: self.process_mesquite("logmesquite_logvolume"),

            # NES (8 variants)
            "nes1972-nes": lambda: self.process_nes("nes1972"),
            "nes1976-nes": lambda: self.process_nes("nes1976"),
            "nes1980-nes": lambda: self.process_nes("nes1980"),
            "nes1984-nes": lambda: self.process_nes("nes1984"),
            "nes1988-nes": lambda: self.process_nes("nes1988"),
            "nes1992-nes": lambda: self.process_nes("nes1992"),
            "nes1996-nes": lambda: self.process_nes("nes1996"),
            "nes2000-nes": lambda: self.process_nes("nes2000"),
        }

        results = {}
        for name, proc_fn in processors.items():
            try:
                data = proc_fn()
                draws = self.load_draws(name)
                flat_draws = self.flatten_draws(draws)

                output = {
                    "name": name,
                    "model_type": data["model_type"],
                    **data,
                    "reference_draws": flat_draws,
                    "n_reference_chains": len(draws),
                    "n_reference_draws_per_chain": len(next(iter(draws[0].values()))),
                }

                out_path = self.out_dir / f"{name}.json"
                with open(out_path, "w") as f:
                    json.dump(output, f)

                n_params = len(flat_draws)
                n_draws = len(next(iter(flat_draws.values())))
                print(f"  OK  {name:50s}  params={n_params:3d}  draws={n_draws}")
                results[name] = "OK"
            except Exception as e:
                print(f"  FAIL {name:50s}  {e}")
                results[name] = f"FAIL: {e}"

        # Summary
        ok = sum(1 for v in results.values() if v == "OK")
        fail = len(results) - ok
        print(f"\nProcessed: {ok} OK, {fail} FAIL out of {len(results)} posteriors")

        # Write manifest
        manifest = {
            "posteriors": [k for k, v in results.items() if v == "OK"],
            "n_posteriors": ok,
            "n_failed": fail,
            "failures": {k: v for k, v in results.items() if v != "OK"},
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return results

    # ---- Model-specific processors ----

    def process_eight_schools(self):
        d = self.load_data("eight_schools")
        return {
            "model_type": "eight_schools",
            "J": d["J"],
            "y": d["y"],
            "sigma": d["sigma"],
            "priors": {
                "mu": {"dist": "normal", "mu": 0.0, "sigma": 5.0},
                "tau": {"dist": "half_cauchy", "scale": 5.0},
                "theta_trans": {"dist": "normal", "mu": 0.0, "sigma": 1.0},
            },
        }

    def process_earnings(self, variant):
        d = self.load_data("earnings")
        N = d["N"]
        earn = np.array(d["earn"])
        height = np.array(d["height"])
        male = np.array(d["male"])

        if variant == "earn_height":
            # y = earn, X = [1, height]
            X = np.column_stack([np.ones(N), height])
            y = earn
        elif variant == "logearn_height":
            X = np.column_stack([np.ones(N), height])
            y = np.log(earn)
        elif variant == "log10earn_height":
            X = np.column_stack([np.ones(N), height])
            y = np.log10(earn)
        elif variant == "logearn_height_male":
            X = np.column_stack([np.ones(N), height, male])
            y = np.log(earn)
        elif variant == "logearn_interaction":
            X = np.column_stack([np.ones(N), height, male, height * male])
            y = np.log(earn)
        elif variant == "logearn_interaction_z":
            # Stan model: z_height = (height - mean) / sd, NOT 2*sd
            # And male is NOT standardized, interaction = z_height * male
            z_height = (height - height.mean()) / height.std(ddof=1)
            X = np.column_stack([np.ones(N), z_height, male, z_height * male])
            y = np.log(earn)
        elif variant == "logearn_logheight_male":
            X = np.column_stack([np.ones(N), np.log(height), male])
            y = np.log(earn)
        else:
            raise ValueError(f"Unknown earnings variant: {variant}")

        return {
            "model_type": "linear_regression",
            "y": y.tolist(),
            "X": X.tolist(),
            "n_beta": X.shape[1],
            "n_obs": N,
            "priors": {
                "beta": {"dist": "flat"},
                "sigma": {"dist": "flat_positive"},
            },
            "param_names": [f"beta[{i+1}]" for i in range(X.shape[1])] + ["sigma"],
        }

    def process_kidiq(self, variant):
        d = self.load_data("kidiq")
        N = d["N"]
        kid_score = np.array(d["kid_score"])
        mom_iq = np.array(d["mom_iq"])
        mom_hs = np.array(d["mom_hs"])

        if variant == "momiq":
            X = np.column_stack([np.ones(N), mom_iq])
            y = kid_score
        elif variant == "momhs":
            X = np.column_stack([np.ones(N), mom_hs])
            y = kid_score
        elif variant == "momhsiq":
            X = np.column_stack([np.ones(N), mom_hs, mom_iq])
            y = kid_score
        elif variant == "interaction":
            inter = mom_hs * mom_iq
            X = np.column_stack([np.ones(N), mom_hs, mom_iq, inter])
            y = kid_score
        else:
            raise ValueError(f"Unknown kidiq variant: {variant}")

        return {
            "model_type": "linear_regression",
            "y": y.tolist(),
            "X": X.tolist(),
            "n_beta": X.shape[1],
            "n_obs": N,
            "priors": {
                "beta": {"dist": "flat"},
                "sigma": {"dist": "cauchy", "scale": 2.5},
            },
            "param_names": [f"beta[{i+1}]" for i in range(X.shape[1])] + ["sigma"],
        }

    def process_kidiq_work(self, variant):
        d = self.load_data("kidiq_with_mom_work")
        # kidiq_with_mom_work has the same kidiq fields + mom_work
        N = d["N"]
        kid_score = np.array(d["kid_score"])
        mom_iq = np.array(d["mom_iq"])
        mom_hs = np.array(d["mom_hs"])
        mom_work = np.array(d["mom_work"])

        # Indicator variables for mom_work categories
        work2 = (mom_work == 2).astype(float)
        work3 = (mom_work == 3).astype(float)
        work4 = (mom_work == 4).astype(float)

        if variant == "mom_work":
            X = np.column_stack([np.ones(N), work2, work3, work4])
            y = kid_score
        elif variant == "interaction_c":
            # Centered predictors
            c_mom_hs = mom_hs - mom_hs.mean()
            c_mom_iq = mom_iq - mom_iq.mean()
            inter = c_mom_hs * c_mom_iq
            X = np.column_stack([np.ones(N), c_mom_hs, c_mom_iq, inter])
            y = kid_score
        elif variant == "interaction_c2":
            # Centered on reference points
            c2_mom_hs = mom_hs - 0.5
            c2_mom_iq = mom_iq - 100
            inter = c2_mom_hs * c2_mom_iq
            X = np.column_stack([np.ones(N), c2_mom_hs, c2_mom_iq, inter])
            y = kid_score
        elif variant == "interaction_z":
            # Standardized
            z_mom_hs = (mom_hs - mom_hs.mean()) / (2 * mom_hs.std(ddof=1))
            z_mom_iq = (mom_iq - mom_iq.mean()) / (2 * mom_iq.std(ddof=1))
            inter = z_mom_hs * z_mom_iq
            X = np.column_stack([np.ones(N), z_mom_hs, z_mom_iq, inter])
            y = kid_score
        else:
            raise ValueError(f"Unknown kidiq_work variant: {variant}")

        return {
            "model_type": "linear_regression",
            "y": y.tolist(),
            "X": X.tolist(),
            "n_beta": X.shape[1],
            "n_obs": N,
            "priors": {
                "beta": {"dist": "flat"},
                "sigma": {"dist": "flat_positive"},
            },
            "param_names": [f"beta[{i+1}]" for i in range(X.shape[1])] + ["sigma"],
        }

    def process_blr(self, data_name):
        d = self.load_data(data_name)
        N = d["N"]
        D = d["D"]
        X = np.array(d["X"])  # already a matrix
        y = np.array(d["y"])

        return {
            "model_type": "linear_regression",
            "y": y.tolist(),
            "X": X.tolist(),
            "n_beta": D,
            "n_obs": N,
            "priors": {
                "beta": {"dist": "normal", "mu": 0.0, "sigma": 10.0},
                "sigma": {"dist": "half_normal", "sigma": 10.0},
            },
            "param_names": [f"beta[{i+1}]" for i in range(D)] + ["sigma"],
        }

    def process_kilpisjarvi(self):
        d = self.load_data("kilpisjarvi_mod")
        N = d["N"]
        x = np.array(d["x"])
        y = np.array(d["y"])

        X = np.column_stack([np.ones(N), x])

        return {
            "model_type": "linear_regression",
            "y": y.tolist(),
            "X": X.tolist(),
            "n_beta": 2,
            "n_obs": N,
            "priors": {
                "beta": {
                    "dist": "normal_per_param",
                    "params": [
                        {"mu": d["pmualpha"], "sigma": d["psalpha"]},
                        {"mu": d["pmubeta"], "sigma": d["psbeta"]},
                    ],
                },
                "sigma": {"dist": "flat_positive"},
            },
            "param_names": ["alpha", "beta", "sigma"],
        }

    def process_mesquite(self, variant):
        d = self.load_data("mesquite")
        N = d["N"]
        weight = np.array(d["weight"])
        diam1 = np.array(d["diam1"])
        diam2 = np.array(d["diam2"])
        canopy_height = np.array(d["canopy_height"])
        total_height = np.array(d["total_height"])
        density = np.array(d["density"])
        group = np.array(d["group"])

        if variant == "mesquite":
            X = np.column_stack([np.ones(N), diam1, diam2, canopy_height, total_height, density, group])
            y = weight
        elif variant == "logmesquite":
            X = np.column_stack([
                np.ones(N), np.log(diam1), np.log(diam2), np.log(canopy_height),
                np.log(total_height), np.log(density), group
            ])
            y = np.log(weight)
        elif variant == "logmesquite_logva":
            log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
            log_canopy_area = np.log(diam1 * diam2)
            X = np.column_stack([np.ones(N), log_canopy_volume, log_canopy_area, group])
            y = np.log(weight)
        elif variant == "logmesquite_logvas":
            log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
            log_canopy_area = np.log(diam1 * diam2)
            log_canopy_shape = np.log(diam1 / diam2)
            X = np.column_stack([
                np.ones(N), log_canopy_volume, log_canopy_area,
                log_canopy_shape, np.log(total_height), np.log(density), group
            ])
            y = np.log(weight)
        elif variant == "logmesquite_logvash":
            log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
            log_canopy_area = np.log(diam1 * diam2)
            log_canopy_shape = np.log(diam1 / diam2)
            X = np.column_stack([
                np.ones(N), log_canopy_volume, log_canopy_area,
                log_canopy_shape, np.log(total_height), group
            ])
            y = np.log(weight)
        elif variant == "logmesquite_logvolume":
            log_canopy_volume = np.log(diam1 * diam2 * canopy_height)
            X = np.column_stack([np.ones(N), log_canopy_volume])
            y = np.log(weight)
        else:
            raise ValueError(f"Unknown mesquite variant: {variant}")

        return {
            "model_type": "linear_regression",
            "y": y.tolist(),
            "X": X.tolist(),
            "n_beta": X.shape[1],
            "n_obs": N,
            "priors": {
                "beta": {"dist": "flat"},
                "sigma": {"dist": "flat_positive"},
            },
            "param_names": [f"beta[{i+1}]" for i in range(X.shape[1])] + ["sigma"],
        }

    def process_nes(self, data_name):
        d = self.load_data(data_name)
        N = d["N"]
        partyid7 = np.array(d["partyid7"])
        real_ideo = np.array(d["real_ideo"])
        race_adj = np.array(d["race_adj"])
        educ1 = np.array(d["educ1"])
        gender = np.array(d["gender"])
        income = np.array(d["income"])
        age_discrete = np.array(d["age_discrete"])

        # Age factors (from Stan transformed data)
        age30_44 = (age_discrete == 2).astype(float)
        age45_64 = (age_discrete == 3).astype(float)
        age65up = (age_discrete == 4).astype(float)

        X = np.column_stack([
            np.ones(N), real_ideo, race_adj, age30_44, age45_64,
            age65up, educ1, gender, income
        ])
        y = partyid7

        return {
            "model_type": "linear_regression",
            "y": y.tolist(),
            "X": X.tolist(),
            "n_beta": 9,
            "n_obs": N,
            "priors": {
                "beta": {"dist": "flat"},
                "sigma": {"dist": "flat_positive"},
            },
            "param_names": [f"beta[{i+1}]" for i in range(9)] + ["sigma"],
        }


def main():
    pdb_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/posteriordb"

    if not os.path.exists(pdb_path):
        print(f"posteriordb not found at {pdb_path}")
        print("Clone it: git clone --depth 1 https://github.com/stan-dev/posteriordb.git /tmp/posteriordb")
        sys.exit(1)

    print(f"Processing posteriordb at {pdb_path}")
    print(f"Output: posteriordb_processed/\n")

    proc = PosteriorDBProcessor(pdb_path)
    proc.process_all()


if __name__ == "__main__":
    main()
