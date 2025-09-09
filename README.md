WhatsApp Auto-Reply Bridge to LM Studio

[![Build](https://github.com/SatyamSaxena1/watusi-bridge/actions/workflows/build.yml/badge.svg)](https://github.com/SatyamSaxena1/watusi-bridge/actions/workflows/build.yml)

Overview
- Python Flask webhook and Go single-binary alternative that accept Watusi Auto-Reply POSTs and reply via LM Studio’s OpenAI-compatible API.
- Per-JID short memory with inactivity reset. Strong defaults for safe, on-topic replies. Admin page for live model/prompt changes.

Components
- Python: app.py (Flask), requirements.txt
- Go: main.go, go.mod, go.sum
- Config: config.yaml (optional)
- Samples: sample_watusi_payload.json, test_client.py

Quick start (Go binary)
- Build: go mod download && go build -o watusi-bridge.exe
- Run: .\watusi-bridge.exe (opens admin at /admin)
- Phone URL: http://<PC_IP>:8001/
- Watusi Webhook: http://<PC_IP>:8001/auto-reply

Admin UI (/admin)
- Change System Instructions on the fly
- Switch model (reads LM Studio /v1/models)
- Writes config.yaml so changes persist

Key env/config
- host, port, lm_studio_host, lm_studio_port, lm_model (auto), temperature, top_p, max_tokens
- inactivity_ttl_seconds, history_len, allowed_jids, webhook_secret, rate_limit_per_minute
- system_instructions, per_jid_instructions, stop_sequences, reset_codewords

Notes
- Ensure Windows Firewall allows inbound TCP 8001 on Private network.
- LM Studio must be running with a model loaded.
