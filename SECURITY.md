# Security Policy

## Supported Versions

Security updates are provided for the latest stable release only.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do not open a public issue.** Security vulnerabilities must be reported privately.

### How to Report

Send an email to **[INSERT SECURITY CONTACT]** with:

1. A clear description of the vulnerability
2. Steps to reproduce (proof-of-concept code if possible)
3. Affected versions
4. Potential impact

### What to Expect

| Timeline | Action |
|----------|--------|
| Within 48 hours | Acknowledgment of receipt |
| Within 7 days | Initial assessment and severity rating |
| Within 30 days | Patch release (critical: within 7 days) |

We follow a coordinated disclosure process:

1. The reporter submits the vulnerability privately
2. We validate and develop a fix
3. We release the fix and publish a security advisory
4. The reporter is credited (unless they prefer anonymity)

### Scope

The following are **in scope**:

- Code injection vulnerabilities (prompt injection, RCE)
- Authentication / authorization bypasses
- Data exfiltration paths
- Supply chain attacks (dependency poisoning)
- Sensitive data exposure in logs or error messages

The following are **out of scope**:

- Issues requiring physical access to the machine
- Social engineering attacks
- Denial of service via resource exhaustion (without data loss)
- Issues in third-party dependencies (report to the upstream project)

## Security Best Practices for Users

- **Never hardcode API keys** in source code or config files committed to version control
- Use `$ENV_VAR` references in Octopus config files and manage keys via environment variables or a secrets manager
- Review the **Action Brain's allowed tools** list to restrict what commands agents can execute
- Enable **budget limits** to prevent runaway API costs
- Run Octopus in a sandboxed environment when using the Action Brain with shell access

## Acknowledgments

We thank the following individuals and organizations for responsibly disclosing security issues:

*(None yet — you could be first!)*
