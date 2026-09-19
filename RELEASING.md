# Releasing molito 0.2.0

The version is set to `0.2.0` in `pyproject.toml`. Release notes are in
[CHANGELOG.md](CHANGELOG.md).
No tag or publication is created by preparing these files.

Local verification passed on Python 3.13: 728 tests from both the source checkout and a fresh
installed wheel, including live xTB; Ruff, mypy, strict documentation build, dependency checks
and distribution checks also passed. Cross-version HDF5 checks against 0.1.1 passed, with
original files unchanged. CI still needs to validate the committed changes on Python 3.11–3.13.

## Before tagging

1. Review and commit the branch changes, open a pull request, and merge after CI passes.
   CI covers Python 3.11–3.13, core-only imports, Ruff, mypy, documentation and built-wheel checks.
2. Reuse the existing PyPI trusted publisher: owner `rssrwn`, repository `molito`,
   workflow `release.yml`, environment `pypi`. The previous 0.1.1 release workflow succeeded,
   so no new setup is expected. If those settings have changed, see
   [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
3. Note the HDF5 compatibility change in the release announcement: new complex/native shards
   use format 2; graph/protein shards retain format 1. Reading old complex interaction payloads
   requires `allow_pickle=True` for trusted files. Users can load and save them into a new directory to migrate.

Also call out the xTB output-unit change: both functions now default to kcal/mol. Existing
optimiser callers expecting Hartree must pass `units="hartree"`; stored results are unchanged.

The optional xTB integration has been tested live in the molito mamba environment with
xtb-python 22.1 and libxtb 6.7.1, including all supported methods, charged/radical molecules,
solvation, hydrogen handling and iteration limits. These tests skip when xTB is unavailable.

## Commit and merge

From the current `mol-formats` branch, after reviewing the changes:

```sh
git diff
git add -A
git commit -m "Prepare molito 0.2.0"
git push -u origin mol-formats
gh pr create --base main --head mol-formats --title "Release molito 0.2.0" --web
```

Use the 0.2.0 changelog entry for the pull request description, then merge after its checks pass.
`AGENTS.md` stays local because it is ignored.
If releasing on a different date, update the 0.2.0 changelog date before committing.

## Publish

After the reviewed changes are on `main`, from a clean checkout:

```sh
git switch main
git pull --ff-only
git tag -a v0.2.0 -m "molito 0.2.0"
git push origin v0.2.0
```

**Pushing the tag starts publication.** The release workflow validates that commit, checks that
the tag matches `pyproject.toml`, builds and checks distributions, then publishes to PyPI.
The `pypi` environment currently has no required-reviewer gate, so publication proceeds
automatically after validation. If reviewers are added later, approve its waiting deployment.
There is no separate `twine upload` step to run locally.

Once the workflow succeeds, create a GitHub Release for `v0.2.0`, using the 0.2.0 changelog entry.
The workflow publishes packages but does not create GitHub Releases. The docs workflow deploys
from `main` separately.

Verify the published package in a fresh environment outside the checkout:

```sh
python -m venv /tmp/molito-020-published
/tmp/molito-020-published/bin/pip install molito==0.2.0
/tmp/molito-020-published/bin/python -I -c 'import molito; print(molito.__version__)'
```

## Local validation

Install the project with its development, documentation and interaction extras, then run:

```sh
python -m unittest discover tests/ -v
ruff check .
ruff format --check .
mypy
mkdocs build --strict
python -m build
twine check dist/molito-0.2.0*
```

Build from a clean checkout to avoid stale files from older local builds. Local `dist/` may
contain older versions; the GitHub workflow starts clean and only uploads the distributions
it builds for the release tag. Never move an already published tag to a different commit;
fix a published release with a new version.
