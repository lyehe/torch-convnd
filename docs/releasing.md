# Releases and deployment

## Build and check

The version lives in `src/torch_convnd/_version.py`. Update it and the changelog
together. Hatch reads the version without importing PyTorch.

```sh
python -m build
python -m twine check dist/*
python tools/check_dist.py
```

Use a fresh `dist` directory for a release. The validator checks wheel/sdist
contents, typed API, source equality, version and mandatory dependencies. Setting
`RELEASE_TAG=v0.5.0` also requires the source version to match that tag. The wheel
bundles C++/CUDA source and builds at runtime when needed; it is not a precompiled
binary distribution.

Test the installed wheel from outside the source checkout:

```sh
python tools/check_dist.py --install
python -I tools/smoke_test.py --installed --onnx
```

The optional `--backend native --device cuda:0` smoke path requires a CUDA toolkit.
CI runs the complete suite before building and installing wheels across its CPU
matrix. The separate CUDA workflow runs the suite and native installed-wheel path
on a manually selected trusted GPU runner.

## GitHub Actions

| Workflow | Trigger / result |
| --- | --- |
| `ci.yml` | PR/main/weekly/manual/reusable: hooks, typing, strict docs, CPU correctness on Linux/Windows/macOS x86_64 and ARM64, the PyTorch release-line contract, CUDA build checks, coverage, wheel/source validation and isolated installed-wheel checks; weekly and manual runs add a PyTorch nightly canary |
| `cuda.yml` | Manual only: trusted self-hosted `gpu` runner on Linux/Windows; selectable PyTorch version, CUDA wheel index and Python (default 2.10, cu130, 3.12); native correctness and wheel checks |
| `release.yml` | Version tag or manual: reuse CI, build checked distributions with provenance attestations, publish a GitHub release for the tag; optional TestPyPI/PyPI publication |
| `docs.yml` | Main/manual: strict site build, optional GitHub Pages deployment |

The source repository is [lyehe/torch-convnd](https://github.com/lyehe/torch-convnd).
Workflow status is shown in its Actions tab. Local validation and remote runner
coverage are distinct; the manual CUDA workflow requires an actual GPU runner.

## Configure package publication

Configure the corresponding PyPI/TestPyPI
Trusted Publisher for workflow `release.yml`, with environment `pypi` or `testpypi`.
The workflow uses OIDC rather than a stored API token. Follow
[PyPI's setup](https://docs.pypi.org/trusted-publishers/using-a-publisher/) and
[GitHub's OIDC guide](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi).

Push a tag matching the package version. By default this runs verification,
builds artifacts with build-provenance attestations, and creates a GitHub release
for the tag with the checked wheel, sdist and that version's changelog section.
The tag needs a `## <version>` section in `CHANGELOG.md`; check it with
`python tools/release_notes.py v0.6.0`. Verify a downloaded artifact with
`gh attestation verify <file> --repo lyehe/torch-convnd`. To publish, manually
dispatch the workflow on that tag with target `testpypi` or `pypi`. For automatic
PyPI publication on version-tag pushes, set repository variable `PUBLISH_PYPI=true`
after Trusted Publishing is configured. Publishing jobs require the matching
version tag and successful verification.

A GitHub release does not publish a package to PyPI.

## Configure documentation deployment

Enable GitHub Pages with GitHub Actions as its source and set repository variable
`PUBLISH_DOCS=true`. The documentation workflow builds the site and deploys main
through the `github-pages` environment. Builds are still available as artifacts
when deployment is disabled. MkDocs also supports other static hosts; see its
[deployment guide](https://www.mkdocs.org/user-guide/deploying-your-docs/).
