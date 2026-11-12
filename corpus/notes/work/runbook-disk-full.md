# Runbook: disk full on an application server

Alert: `node_filesystem_avail_bytes` below 5 percent on `/var`.

## Check

```
df -h /var
sudo du -xh /var --max-depth=2 | sort -h | tail -20
```

Usually it is `/var/log` (application logs not rotated) or `/var/lib/docker`
(old images and stopped containers).

## Fix

- Logs: `sudo logrotate -f /etc/logrotate.d/app`, then delete compressed logs older than
  14 days. Shrink the systemd journal with `sudo journalctl --vacuum-size=500M`.
- Docker: `docker system prune -af --filter "until=168h"`.
- Never delete files a running process still has open; check with `lsof +L1`.

## Afterwards

If the same server fills up twice in a month, open a ticket to add log shipping or a
bigger volume.
