#!/usr/bin/env bash
OBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for svc in loki promtail grafana; do
  if [ -f "$OBS_DIR/$svc.pid" ]; then
    kill "$(cat "$OBS_DIR/$svc.pid")" 2>/dev/null || true
    rm -f "$OBS_DIR/$svc.pid"
    echo "Stopped $svc"
  fi
done
