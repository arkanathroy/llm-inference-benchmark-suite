#!/usr/bin/env bash
# Downloads and starts Loki + Promtail + Grafana as standalone binaries
# directly inside the Colab VM (no Docker needed -- Colab does not
# support nested/privileged Docker daemons reliably).
#
# Usage (from repo root, inside a Colab cell via !bash):
#   !bash observability/setup_observability.sh
#
# This starts all three as background processes and prints the Grafana
# URL to open (via a Colab port-forward / ngrok, see docs/observability.md).
set -euo pipefail

OBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$OBS_DIR/bin"
LOG_DIR="$OBS_DIR/../logs"
mkdir -p "$BIN_DIR" "$LOG_DIR"

LOKI_VERSION="2.9.6"
PROMTAIL_VERSION="2.9.6"
GRAFANA_VERSION="10.4.2"

echo "[observability] Downloading Loki ${LOKI_VERSION}..."
if [ ! -f "$BIN_DIR/loki" ]; then
  curl -sSL -o /tmp/loki.zip "https://github.com/grafana/loki/releases/download/v${LOKI_VERSION}/loki-linux-amd64.zip"
  unzip -oq /tmp/loki.zip -d "$BIN_DIR"
  mv "$BIN_DIR/loki-linux-amd64" "$BIN_DIR/loki"
  chmod +x "$BIN_DIR/loki"
fi

echo "[observability] Downloading Promtail ${PROMTAIL_VERSION}..."
if [ ! -f "$BIN_DIR/promtail" ]; then
  curl -sSL -o /tmp/promtail.zip "https://github.com/grafana/loki/releases/download/v${PROMTAIL_VERSION}/promtail-linux-amd64.zip"
  unzip -oq /tmp/promtail.zip -d "$BIN_DIR"
  mv "$BIN_DIR/promtail-linux-amd64" "$BIN_DIR/promtail"
  chmod +x "$BIN_DIR/promtail"
fi

echo "[observability] Downloading Grafana ${GRAFANA_VERSION}..."
if [ ! -d "$BIN_DIR/grafana-v${GRAFANA_VERSION}" ]; then
  curl -sSL -o /tmp/grafana.tar.gz "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"
  tar -xzf /tmp/grafana.tar.gz -C "$BIN_DIR"
fi

echo "[observability] Starting Loki on :3100..."
nohup "$BIN_DIR/loki" -config.file="$OBS_DIR/loki-config.yaml" > "$LOG_DIR/loki.out" 2>&1 &
echo $! > "$OBS_DIR/loki.pid"

sleep 3

echo "[observability] Starting Promtail (tailing $LOG_DIR/*.log)..."
nohup "$BIN_DIR/promtail" -config.file="$OBS_DIR/promtail-config.yaml" > "$LOG_DIR/promtail.out" 2>&1 &
echo $! > "$OBS_DIR/promtail.pid"

echo "[observability] Starting Grafana on :3000 (anonymous admin access enabled)..."
GRAFANA_HOME="$BIN_DIR/grafana-v${GRAFANA_VERSION}"
export GF_AUTH_ANONYMOUS_ENABLED=true
export GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
export GF_AUTH_DISABLE_LOGIN_FORM=true
nohup "$GRAFANA_HOME/bin/grafana" server \
  --homepath="$GRAFANA_HOME" \
  --config="$GRAFANA_HOME/conf/defaults.ini" \
  > "$LOG_DIR/grafana.out" 2>&1 &
echo $! > "$OBS_DIR/grafana.pid"

sleep 5
echo "[observability] All services started."
echo "[observability] Loki:     http://localhost:3100"
echo "[observability] Grafana:  http://localhost:3000  (anonymous admin access)"
echo "[observability] Use Colab's port-forwarding (see docs/observability.md) to open Grafana in your browser."
