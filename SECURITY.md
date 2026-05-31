# Security Policy

Stable-ASR is a research toolkit for ASR manifests, streaming evaluation,
turn-taking, and reproducibility artifacts. It does not intentionally collect
secrets, credentials, or private speech data.

## Supported Versions

The project is pre-alpha. Security fixes are applied to the default branch until
the first tagged release exists.

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability. Email the maintainers
or use the private security reporting channel configured on the repository.

Please include:

- affected command, module, or artifact
- reproduction steps with synthetic or public data
- expected impact
- any relevant logs with secrets and private audio removed

We will acknowledge valid reports, triage severity, and coordinate a fix before
public disclosure.

## Scope

In scope:

- unsafe archive, manifest, or path handling
- command execution paths used by external ASR adapters
- dependency or packaging behavior that can affect users installing the toolkit
- accidental exposure of local paths, credentials, or private dataset metadata

Out of scope:

- benchmark quality disagreements
- model accuracy or fairness issues without a security impact
- vulnerabilities in upstream ASR systems unless Stable-ASR's integration makes
  them worse

## Data Handling

Reports should use public or synthetic audio whenever possible. Do not send
private speech recordings, API keys, access tokens, or proprietary datasets.
