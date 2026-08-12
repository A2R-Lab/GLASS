# Security Policy

## Supported versions

GLASS is a header-only CUDA template library; only the latest release (and
`main`) receive fixes.

| Version | Supported |
|---------|-----------|
| latest release / `main` | ✅ |
| older tags | ❌ |

## Reporting a vulnerability

Please report vulnerabilities **privately** through GitHub's security advisory
flow: [Report a vulnerability](https://github.com/A2R-Lab/GLASS/security/advisories/new)
(Repository → Security → Report a vulnerability). Do not open a public issue
for a security-sensitive report.

We aim to acknowledge reports within a week. Because GLASS runs entirely inside
the caller's own CUDA kernels (no network, no file I/O, no privileged
operations), most issues are correctness bugs rather than vulnerabilities — but
memory-safety reports (out-of-bounds shared-memory access, scratch-size
under-allocation in a documented-correct usage) are in scope and welcome.
