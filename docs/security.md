# Security Hardening Notes

This project is an educational prototype. The points below describe a practical hardening baseline for handling sensitive health-adjacent data.

## PII/PHI Handling Policy

- **Data minimization**: only collect fields required for risk scoring and explainability.
- **No raw identifiers by default**: avoid name, email, phone, address, national IDs in request payloads.
- **Redaction in logs**: audit logs should include only normalized clinical features and decision metadata.
- **Need-to-know access**: restrict memory/history endpoints behind token auth in non-local environments.
- **Environment separation**: keep development/test data isolated from demo/production-like data.

## Data Retention Policy

- **Default retention (prototype)**: keep decision/audit history for 30 days.
- **Memory retention**: rotate or archive old ChromaDB patient vectors on a schedule.
- **Right to delete**: support deleting patient-associated memory records on request.
- **Backups**: encrypt backups and define retention windows for snapshots.

## Recommended Production Controls

- Enforce API auth (`REQUIRE_API_TOKEN=true`) and rotate tokens regularly.
- Move from shared token auth to per-user auth (JWT/OAuth) for multi-user environments.
- Use HTTPS termination and secure secrets storage (vault/KMS, not plaintext env files).
- Add rate limiting and request size limits at API gateway/reverse proxy level.
- Add structured security events and alerting for auth failures and unusual access patterns.

## Storage and Encryption Guidance

- Encrypt data at rest for vector store, logs, and backups.
- Encrypt data in transit between UI, API, and storage.
- Use separate credentials/roles for read-only analytics vs write paths.
- Prevent accidental commits of logs/databases by keeping them in `.gitignore`.

## Compliance Notes

- Treat this as non-diagnostic decision support.
- For regulated use, complete threat modeling, DPIA/HIPAA review, and clinical validation.
- Maintain auditability of model version, feature inputs, and recommendation provenance.
