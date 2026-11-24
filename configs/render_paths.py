"""
experiment_name = 'render_paths'
dataset_name    = 'n/a'

Renders configs/paths.yaml from .env plus configs/paths.yaml.template, and
audits whether each paper figure can be reproduced on this machine.

src/config.py in the frozen repo reads external_xai_ood/configs/paths.yaml and
knows nothing about environment variables, so the flow is: edit .env, render
the yaml here, then point the frozen repo at it (see README, "Wiring").

Usage:
  $PYTHON_BIN configs/render_paths.py            # render, then audit
  $PYTHON_BIN configs/render_paths.py --check    # audit only
  $PYTHON_BIN configs/render_paths.py --env other.env
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

VAR_PATTERN = re.compile(r"\$\{(\w+)\}")
ARTIFACT_ROOTS = {"mosaix": "MOSAIX_ROOT", "xai": "XAI_OOD_ROOT"}

FIGURES = {
    "Fig. 1 - teaser": dict(
        pdf="teaser/fig1_teaser_inline.pdf",
        command="fig1_code/fig1_teaser.py --bench in1k --layout inline --norm panel",
        env=["CIFAR_TORCHVISION_ROOT"],
        artifacts=["mosaix:results/grid_dump/grids__ce.npz",
                   "mosaix:results/grid_dump/grids__ls.npz",
                   "mosaix:results/grid_dump/grids__mixup.npz"],
        note="Reads the frozen grid dumps of scripts/43 and loads no "
             "checkpoint of its own. CE_CKPT, LS_CKPT and MIXUP_CKPT are "
             "needed only to regenerate those dumps.",
    ),
    "Fig. 2 - method": dict(
        pdf="method/fig2_method.pdf",
        command="fig2_code/fig2_method_data.py --key ce --query <id>, then "
                "fig2_code/fig2_method.py",
        env=["CE_CKPT", "CE_FEATDIR", "IN1K_TRAIN"],
        artifacts=["mosaix:results/method_fig",
                   "mosaix:results/ref_bank/bank__ce.npz",
                   "mosaix:figures/method/assets/snowflake_cskd.png"],
        note="Script 56 runs the backbone live over IN1K_TRAIN to build the "
             "panels. Script 57 only draws them.",
    ),
    "Fig. 3 - saliency zoo": dict(
        pdf="head_refit/149_f1__saliency_zoo.pdf",
        command="fig3_code/fig3_saliency_zoo.py",
        env=["CE_CKPT", "IN1K_TRAIN", "OOD_INATURALIST", "OOD_OPENIMAGE_O",
             "OOD_TEXTURES"],
        artifacts=["mosaix:results/saliency_zoo/winner.pt"],
        note="winner.pt caches the selected query. Delete it to re-run the "
             "search, which needs the OOD sets above.",
    ),
    "Fig. 4 - head refit": dict(
        pdf="head_refit/119_fig5_headrefit_4col.pdf",
        command="fig4_code/fig4_head_refit.py",
        env=["CE_CKPT", "LS_CKPT", "IN1K_TRAIN"],
        artifacts=["mosaix:results/head_refit/head_ce_refit__ls.pt"],
        note="Loads the CE and LS backbones live. The third column needs the "
             "refit head cached in that artifact; without it the script "
             "refits from the LS training features and also needs LS_FEATDIR.",
    ),
    "Fig. 5 - placement ablation": dict(
        pdf="ablations_col/163_ablations_placement_col.pdf",
        command="fig5_code/fig5_placement.py",
        env=[],
        artifacts=["xai:results/stats/imagenet1k/table1.csv",
                   "xai:results/protocol_placement/imagenet1k"],
        note="Reads frozen CSVs only. No checkpoint, no dataset.",
    ),
    "Fig. 6 - cross-benchmark": dict(
        pdf="crossplot_bench/157_crossplot_bench__cifar224_col.pdf",
        command="fig6_code/fig6_crossplot.py --cifar 224",
        env=[],
        artifacts=["mosaix:results/table2_multi/auroc.csv",
                   "mosaix:results/cifar100_224_panel/auroc.csv"],
        note="Reads frozen CSVs only. The upstream runs that produced them "
             "needed IM200_* and CIFAR_*.",
    ),
}

TABLES = {
    "tab:field": dict(
        command="table_code/tab_field_posemb.py, then table_code/tab_field_mosaic.py",
        env=["IN1K_VAL", "OOD_SSB_HARD", "OOD_NINCO", "OOD_INATURALIST",
             "OOD_OPENIMAGE_O", "OOD_TEXTURES", "TORCH_HOME"],
        artifacts=["mosaix:results/mosaic_sizes"],
        note="Public torchvision IMAGENET1K_V1 weights for RN-50 and ViT-B/16, "
             "downloaded on demand into TORCH_HOME. Seed 0. The query and "
             "prototype file lists are in the manifest under "
             "results/mosaic_sizes/.",
    ),
    "every other table": dict(
        command="table_code/make_tables.py",
        env=[],
        artifacts=["xai:results"],
        note="Regenerates tables/*.tex from the frozen CSVs.",
    ),
}


def load_env(path: Path) -> dict:
    values: dict[str, str] = {}
    if not path.exists():
        sys.exit(f"no env file at {path}. Copy .env.example to .env first.")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        values[key] = VAR_PATTERN.sub(
            lambda m: values.get(m.group(1), m.group(0)), value)
    return values


def render(env: dict, template_path: Path, output_path: Path) -> list[str]:
    text = template_path.read_text()
    unresolved = sorted({v for v in VAR_PATTERN.findall(text) if v not in env})
    output_path.write_text(
        VAR_PATTERN.sub(lambda m: env.get(m.group(1), m.group(0)), text))
    return unresolved


def resolve_artifact(spec: str, env: dict) -> Path:
    prefix, _, relative = spec.partition(":")
    return Path(env.get(ARTIFACT_ROOTS[prefix], "/unset")) / relative


def audit_variables(env: dict) -> set[str]:
    missing = set()
    print("environment variables")
    print("-" * 72)
    for key in sorted(env):
        value = env[key]
        if not value.startswith("/"):
            print(f"  ..  {key:24s} {value}")
            continue
        found = Path(value).exists()
        if not found and key != "TORCH_HOME":
            missing.add(key)
        print(f"  {'OK' if found else '--'} {key:24s} {value}")
    return missing


def audit_entry(name, spec, env, missing_vars) -> bool:
    env_gaps = [k for k in spec["env"] if k in missing_vars or k not in env]
    artifact_gaps = [a for a in spec["artifacts"]
                     if not resolve_artifact(a, env).exists()]
    ready = not env_gaps and not artifact_gaps
    print(f"  {'OK' if ready else '--'} {name}")
    if "pdf" in spec:
        print(f"       pdf     figures/{spec['pdf']}")
    print(f"       run     $PYTHON_BIN {spec['command']}")
    if env_gaps:
        print(f"       MISSING env       {', '.join(env_gaps)}")
    if artifact_gaps:
        print(f"       MISSING artifacts {', '.join(artifact_gaps)}")
    print(f"       note    {spec['note']}")
    print()
    return ready


def audit(env: dict) -> int:
    missing_vars = audit_variables(env)

    print("\nfigures")
    print("-" * 72)
    n_reproducible = sum(audit_entry(name, spec, env, missing_vars)
                         for name, spec in FIGURES.items())

    print("tables")
    print("-" * 72)
    for name, spec in TABLES.items():
        audit_entry(name, spec, env, missing_vars)

    print("-" * 72)
    print(f"{n_reproducible}/{len(FIGURES)} figures reproducible with what is "
          "on disk.")
    if missing_vars:
        print(f"unset or missing paths: {', '.join(sorted(missing_vars))}")
    return 0 if n_reproducible == len(FIGURES) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=str(REPO / ".env"))
    parser.add_argument("--check", action="store_true",
                        help="audit only, do not write configs/paths.yaml")
    args = parser.parse_args()

    env = load_env(Path(args.env))

    if not args.check:
        unresolved = render(env, HERE / "paths.yaml.template",
                            HERE / "paths.yaml")
        print(f"wrote {HERE / 'paths.yaml'}")
        if unresolved:
            print(f"WARNING unresolved placeholders: {', '.join(unresolved)}")
        print()

    return audit(env)


if __name__ == "__main__":
    raise SystemExit(main())
