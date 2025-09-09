import os
import logging
from typing import Dict, List, Optional, Tuple

import requests
from flask import Flask, request, jsonify


app = Flask(__name__)

# Configuration via environment variables with sensible defaults
LM_STUDIO_HOST = os.getenv("LM_STUDIO_HOST", "192.168.80.1")
LM_STUDIO_PORT = int(os.getenv("LM_STUDIO_PORT", "1234"))
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "auto")
LM_TEMPERATURE = float(os.getenv("LM_TEMPERATURE", "0.7"))
LM_MAX_TOKENS = int(os.getenv("LM_MAX_TOKENS", "300"))
LM_REQUEST_TIMEOUT = float(os.getenv("LM_REQUEST_TIMEOUT", "20"))  # seconds for LM Studio request
# Custom system instructions injected as a system message at the top of each chat
SYSTEM_INSTRUCTIONS = os.getenv("SYSTEM_INSTRUCTIONS", "").strip()
# Auto reset per-JID chat history after this many seconds of inactivity
INACTIVITY_RESET_SECONDS = int(os.getenv("INACTIVITY_RESET_SECONDS", "120"))
ALLOWED_JIDS = {
	j.strip()
	for j in os.getenv("ALLOWED_JIDS", "").split(",")
	if j.strip()
}

LM_STUDIO_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/chat/completions"
LM_STUDIO_COMPLETIONS_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/completions"

# In-memory conversation store keyed by sender JID: { jid: {"history": List[messages], "last_ts": float, "session": int} }
conversations: Dict[str, Dict[str, object]] = {}

# Basic logging configuration (configurable via LOG_LEVEL env)
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO))


def _extract_message_text(payload: Dict) -> str:
	"""Best-effort extraction of message text from various possible keys."""
	candidates = [
		"text",
		"message",
		"body",
		"MESSAGE",
		"MESSAGE-TEXT",
		"message-text",
		"content",
	]
	for k in candidates:
		v = payload.get(k)
		if isinstance(v, str) and v.strip():
			return v
	# Some payloads wrap data in 'data' or 'payload'
	for wrapper in ("data", "payload"):
		inner = payload.get(wrapper)
		if isinstance(inner, dict):
			for k in candidates:
				v = inner.get(k)
				if isinstance(v, str) and v.strip():
					return v
	return ""


def _extract_ai_text(result: Dict) -> str:
	"""Handle both chat.completions and completions shapes from LM Studio."""
	try:
		choices = result.get("choices")
		if isinstance(choices, list) and choices:
			ch0 = choices[0]
			# chat.completions
			msg = ch0.get("message") if isinstance(ch0, dict) else None
			if isinstance(msg, dict) and isinstance(msg.get("content"), str):
				return msg["content"].strip()
			# completions
			if isinstance(ch0, dict) and isinstance(ch0.get("text"), str):
				return ch0["text"].strip()
		# fallback
		if isinstance(result.get("content"), str):
			return result["content"].strip()
	except Exception:
		pass
	return ""


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
	"""Simple conversion of chat messages to a plain prompt for completions fallback."""
	parts: List[str] = []
	for m in messages[-8:]:  # keep it short
		role = m.get("role")
		content = (m.get("content") or "").strip()
		if not content:
			continue
		if role == "user":
			parts.append(f"User: {content}")
		elif role == "assistant":
			parts.append(f"Assistant: {content}")
		else:
			parts.append(content)
	parts.append("Assistant:")
	return "\n".join(parts)


def _call_lm_studio(messages: List[Dict[str, str]]) -> str:
	"""Call LM Studio with retry and a fallback to /v1/completions. Returns AI text or ''."""
	def _chat_payload(model_id: str) -> Dict:
		return {
			"model": model_id,
			"messages": messages,
			"temperature": LM_TEMPERATURE,
			"max_tokens": LM_MAX_TOKENS,
			"stream": False,
		}

	model_id = get_active_model()
	for attempt in range(2):  # one retry max
		try:
			resp = requests.post(LM_STUDIO_URL, json=_chat_payload(model_id), timeout=LM_REQUEST_TIMEOUT)
			resp.raise_for_status()
			result = resp.json()
			ai_text = _extract_ai_text(result)
			if ai_text:
				return ai_text
			logging.error("Unexpected LM Studio chat response: %s", str(result)[:800])
		except Exception as e:
			logging.warning("LM Studio chat call failed (attempt %s): %s", attempt + 1, e)
			# refresh model id and retry once
			model_id = get_active_model()

	# Fallback to completions endpoint
	try:
		prompt = _messages_to_prompt(messages)
		payload = {
			"model": get_active_model(),
			"prompt": prompt,
			"temperature": LM_TEMPERATURE,
			"max_tokens": LM_MAX_TOKENS,
		}
		resp = requests.post(LM_STUDIO_COMPLETIONS_URL, json=payload, timeout=LM_REQUEST_TIMEOUT)
		resp.raise_for_status()
		result = resp.json()
		ai_text = _extract_ai_text(result)
		return ai_text or ""
	except Exception as e:
		logging.error("LM Studio completions fallback failed: %s", e)
		return ""


def _now_ts() -> float:
	try:
		import time
		return time.time()
	except Exception:
		return 0.0


def _get_or_reset_session(jid: str) -> Tuple[List[Dict[str, str]], int]:
	"""Return (history, session_id) resetting if inactive for > INACTIVITY_RESET_SECONDS."""
	now = _now_ts()
	entry = conversations.get(jid)
	if not isinstance(entry, dict):
		entry = {"history": [], "last_ts": 0.0, "session": 1}
		conversations[jid] = entry
	last_ts = float(entry.get("last_ts") or 0.0)
	session = int(entry.get("session") or 1)
	history = entry.get("history")
	if not isinstance(history, list):
		history = []
	# Reset if inactive
	if last_ts and INACTIVITY_RESET_SECONDS > 0 and (now - last_ts) > INACTIVITY_RESET_SECONDS:
		session += 1
		history = []
		entry["history"] = history
		entry["session"] = session
		logging.info("Session reset for %s due to inactivity (> %ss). New session #%s", jid, INACTIVITY_RESET_SECONDS, session)
	entry["last_ts"] = now
	# Trim to last 8 turns
	trimmed = history[-8:]
	return trimmed, session


def build_messages(sender_name: str, sender_jid: str, text: str) -> List[Dict[str, str]]:
	# Keep chat context on our side and auto-reset by inactivity
	history, session = _get_or_reset_session(sender_jid)
	messages: List[Dict[str, str]] = []
	# Inject system instructions if configured
	if SYSTEM_INSTRUCTIONS:
		messages.append({"role": "system", "content": SYSTEM_INSTRUCTIONS})
	# Add prior context and the new user message
	messages.extend(history)
	messages.append({"role": "user", "content": text})
	return messages


def get_active_model() -> str:
	"""Confirm the model to use. Prefer LM_STUDIO_MODEL if available; else pick first from /v1/models."""
	try:
		models_resp = requests.get(f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/models", timeout=10)
		models_resp.raise_for_status()
		data = models_resp.json()
		ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
		pref = (LM_STUDIO_MODEL or "").strip().lower()
		# Auto mode: use whichever model LM Studio reports (first in list)
		if pref in ("", "auto", "*"):
			if ids:
				return ids[0]
			# No models reported; fall through to return configured value
		else:
			# Explicit preference: use if present, else fallback to first available
			if LM_STUDIO_MODEL in ids:
				return LM_STUDIO_MODEL
			if ids:
				logging.info("Requested model '%s' not found. Using first available: %s", LM_STUDIO_MODEL, ids[0])
				return ids[0]
	except Exception as e:
		logging.warning("Could not fetch /v1/models from LM Studio: %s", e)
	return LM_STUDIO_MODEL


@app.post("/auto-reply")
def auto_reply():
	try:
		# Accept JSON or form-encoded bodies
		data = request.get_json(silent=True) or {}
		if not data and request.form:
			data = request.form.to_dict()

		sender_jid = data.get("jid") or data.get("wa_id") or data.get("from") or "unknown"
		sender_name = data.get("name") or data.get("profile_name") or data.get("chat_name") or "User"
		message_text = _extract_message_text(data)

		logging.info("Incoming message from %s (%s): %s", sender_name, sender_jid, (message_text or "").strip()[:200])
		if SYSTEM_INSTRUCTIONS:
			logging.debug("Using system instructions (%s chars)", len(SYSTEM_INSTRUCTIONS))

		# Allow only specific sender JIDs if configured
		if ALLOWED_JIDS and sender_jid not in ALLOWED_JIDS:
			logging.info("Ignoring message from unauthorized JID: %s", sender_jid)
			return jsonify({"message": ""})

		if not message_text:
			return jsonify({"message": ""})  # no-op

		# Build conversation and LM Studio payload
		messages = build_messages(sender_name, sender_jid, message_text)
		ai_text = _call_lm_studio(messages)
		if not ai_text:
			# Silent no-op to avoid sending technical error text to the contact
			logging.info("No AI text produced; returning empty message to avoid user-facing errors.")
			return jsonify({"message": ""})

		logging.info("Reply to %s (%s): %s", sender_name, sender_jid, ai_text.strip()[:200])

		# Update memory
		# Persist to session history
		entry = conversations.setdefault(sender_jid, {"history": [], "last_ts": _now_ts(), "session": 1})
		hist = entry.get("history")
		if not isinstance(hist, list):
			hist = []
			entry["history"] = hist
		hist.append({"role": "user", "content": message_text})
		hist.append({"role": "assistant", "content": ai_text})
		# Trim stored history to last 8 items (messages alternate user/assistant)
		entry["history"] = hist[-8:]
		entry["last_ts"] = _now_ts()

		# Return verbatim reply
		return jsonify({"message": ai_text})

	except requests.RequestException as e:
		logging.exception("Error calling LM Studio: %s", e)
		# Return empty to avoid exposing errors to the chat; Watusi should then send nothing.
		return jsonify({"message": ""}), 200
	except Exception as e:
		logging.exception("Unhandled error handling payload: %s", e)
		return jsonify({"message": ""}), 200


@app.get("/")
def root():
	return jsonify({
		"status": "ok",
		"lm_studio_url": LM_STUDIO_URL,
		"configured_model": LM_STUDIO_MODEL,
		"active_model": get_active_model(),
		"inactivity_reset_seconds": INACTIVITY_RESET_SECONDS,
		"system_instructions": bool(SYSTEM_INSTRUCTIONS),
		"tracked_sessions": len(conversations),
	})


def main():
	host = os.getenv("HOST", "0.0.0.0")
	port = int(os.getenv("PORT", "8000"))
	app.run(host=host, port=port)


if __name__ == "__main__":
	main()
