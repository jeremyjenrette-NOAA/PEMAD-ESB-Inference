# Permissions model

Two-tier grant model observed in practice (project `ggn-nmfs-osi-dev-1`):

- **service_admins / OSI team** (fast, informal -- ask directly): Docker
  Artifact Registry push/pull, source-repo read access, Cloud Workstation
  user-sharing, general Cloud Batch usage.
- **Platform Team** (formal, JIRA-gated): Cloud Workstation *creation*,
  **`composer.environments.executeAirflowCommand`** on a specific Composer
  environment (needed for *any* headless/CLI DAG triggering -- the Airflow
  web UI uses a separate access path and does not imply this), and "View
  Batch Jobs" (though `roles/batch.viewer` project-wide may already cover
  this for you).

Holding the broader `roles/composer.user` IAM role at the project level is
**not sufficient by itself** for headless triggering -- confirmed directly:
`roles/composer.user` was present and `gcloud composer environments run`
still failed with `PERMISSION_DENIED: composer.environments.executeAirflowCommand`
until a separate Platform Team JIRA ticket was filed and granted.

## Filing a Platform Team ticket

Portal: https://apps-st.fisheries.noaa.gov/jira/servicedesk/customer/portal/14
-> "Google Cloud Platform (GCP) Service Request" -> Request Type:
"Permissions Change".

- **Project/Compartment Name**: use the project that actually hosts the
  resource -- get this from the exact error message
  (`projects/<PROJECT>/locations/.../environments/...`), not from another
  ticket's example. For `composer-env1` this is `ggn-nmfs-osi-dev-1` (no
  `-data` suffix).
- **Additional comments**: name the exact permission and resource, cite
  any IAM role you already hold that *should* cover it but doesn't, and
  paste the literal failing command + error text as evidence.

## Diagnostic notes

- gcloud's active account may not match the identity you expect --
  `gcloud auth list` / the account named in any `PERMISSION_DENIED` error
  is ground truth, not whatever email you assume is "yours."
- `getIamPolicy` failures on a bucket/repo are a different permission
  class from resource *usage* -- don't chase them; test with
  `gcloud storage cat` / `gcloud storage ls` etc. instead.
- Project/environment IDs are easiest to get right by reading them out of
  an actual `PERMISSION_DENIED` error's resource path rather than copying
  from someone else's example ticket or doc.
