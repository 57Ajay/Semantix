# Postmortem: public API outage, 7 May 2024

**Duration:** 47 minutes (09:02 to 09:49 UTC)
**Impact:** every request to the public API failed with a TLS handshake error. Mobile apps
showed "cannot connect". Webhooks to customers were queued and delivered late.

## What happened

The TLS certificate on the API gateway expired at 09:00 UTC. Renewal had been automatic
until we moved the gateway to the new load balancer in February; the renewal job still
pointed at the old one, so the certificate was renewed on a machine nobody used.

Clients saw `x509: certificate has expired or is not yet valid`.

## Timeline

- 09:02 first alert: error rate above 50 percent.
- 09:15 on-call engineer finds the expired certificate.
- 09:41 new certificate issued manually and uploaded.
- 09:49 error rate back to normal.

## Action items

1. Move certificate renewal to cert-manager on the cluster that serves the traffic.
2. Alert 21 days before any certificate expires.
3. Add the gateway certificate to the monthly checklist until (1) is done.
