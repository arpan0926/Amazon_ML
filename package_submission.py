"""Assemble the challenge's required submission ZIP layout."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    required_outputs = [
        ROOT / "output" / "matching_results.tsv",
        ROOT / "output" / "candidate_pairs.tsv",
    ]
    missing = [str(path) for path in required_outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Generate both required output files before packaging: " + ", ".join(missing)
        )

    package_dir = ROOT / "code" / "business_entity_resolution"
    source_files = sorted((package_dir / "src").rglob("*.py"))
    if not source_files:
        raise FileNotFoundError(f"No pipeline source files found under {package_dir / 'src'}")

    archive = Path(args.output) if args.output else ROOT / f"{args.team_name}_submission.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in required_outputs:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        for path in source_files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        for relative_path in (
            "code/business_entity_resolution/README.md",
            "code/business_entity_resolution/requirements.txt",
            "Documentation_template.md",
        ):
            path = ROOT / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Required package file missing: {path}")
            bundle.write(path, relative_path)
    print(f"Created {archive}")


if __name__ == "__main__":
    main()