# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
**Security → Report a vulnerability** (private vulnerability reporting) on this
repository. Do not open a public issue.

Include what you found, how to reproduce it and the impact you expect. You can
expect an acknowledgement within 7 days and a status update within 30 days.

## Supported versions

Only the latest commit on `main` is supported while the project is pre-1.0.

## Practices in this repository

- Secrets never live in the repository: `.env` files and keys are ignored, and CI
  scans the full history with gitleaks.
- Dependencies are audited with pip-audit in CI and updated by Dependabot.
- GitHub Actions run with read-only permissions and are pinned to commit SHAs.
- Configuration files are validated strictly before use; the simulation never
  executes code from input files.
