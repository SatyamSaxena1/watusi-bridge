package main

import (
    "bytes"
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "io"
    "net/http"
    "os"
    "strconv"
    "strings"
    "time"
    "unicode"
    "sync"
    "os/exec"
    "runtime"

    backoff "github.com/cenkalti/backoff/v4"
    "github.com/google/uuid"
    lru "github.com/hashicorp/golang-lru/v2"
    "github.com/rs/zerolog"
    "github.com/rs/zerolog/log"
    "gopkg.in/yaml.v3"
)

// Config holds server and LM Studio settings.
type Config struct {
    Host                   string            `yaml:"host"`
    Port                   int               `yaml:"port"`
    LMStudioHost           string            `yaml:"lm_studio_host"`
    LMStudioPort           int               `yaml:"lm_studio_port"`
    LMModel                string            `yaml:"lm_model"`
    Temperature            float64           `yaml:"temperature"`
    TopP                   float64           `yaml:"top_p"`
    FrequencyPenalty       float64           `yaml:"frequency_penalty"`
    PresencePenalty        float64           `yaml:"presence_penalty"`
    MaxTokens              int               `yaml:"max_tokens"`
    RequestTimeoutSeconds  int               `yaml:"request_timeout_seconds"`
    InactivityTTLSeconds   int               `yaml:"inactivity_ttl_seconds"`
    HistoryLen             int               `yaml:"history_len"`
    AllowedJIDs            []string          `yaml:"allowed_jids"`
    WebhookSecret          string            `yaml:"webhook_secret"`
    RateLimitPerMinute     int               `yaml:"rate_limit_per_minute"`
    SystemInstructions     string            `yaml:"system_instructions"`
    PerJIDInstructions     map[string]string `yaml:"per_jid_instructions"`
    ProcessedTTLSeconds    int               `yaml:"processed_ttl_seconds"`
    MaxBodyBytes           int64             `yaml:"max_body_bytes"`
    LogLevel               string            `yaml:"log_level"`
    StopSequences          []string          `yaml:"stop_sequences"`
    ResetCodewords         []string          `yaml:"reset_codewords"`
}

func defaultConfig() Config {
    return Config{
        Host:                  envOr("HOST", "0.0.0.0"),
        Port:                  envOrInt("PORT", 8000),
        LMStudioHost:          envOr("LM_STUDIO_HOST", "127.0.0.1"),
        LMStudioPort:          envOrInt("LM_STUDIO_PORT", 1234),
        LMModel:               envOr("LM_STUDIO_MODEL", "auto"),
        Temperature:           envOrFloat("LM_TEMPERATURE", 0.2),
        TopP:                  envOrFloat("LM_TOP_P", 0.9),
    FrequencyPenalty:      envOrFloat("LM_FREQUENCY_PENALTY", 0.6),
    PresencePenalty:       envOrFloat("LM_PRESENCE_PENALTY", 0.3),
    MaxTokens:             envOrInt("LM_MAX_TOKENS", 80),
        RequestTimeoutSeconds: envOrInt("LM_REQUEST_TIMEOUT", 20),
        InactivityTTLSeconds:  envOrInt("INACTIVITY_RESET_SECONDS", 120),
        HistoryLen:            envOrInt("HISTORY_LEN", 8),
        AllowedJIDs:           splitCSV(os.Getenv("ALLOWED_JIDS")),
        WebhookSecret:         os.Getenv("WEBHOOK_SECRET"),
        RateLimitPerMinute:    envOrInt("RATE_LIMIT_PER_MINUTE", 60),
        SystemInstructions:    withDefaultSystemInstructions(os.Getenv("SYSTEM_INSTRUCTIONS")),
        PerJIDInstructions:    map[string]string{},
        ProcessedTTLSeconds:   envOrInt("PROCESSED_TTL_SECONDS", 300),
        MaxBodyBytes:          envOrInt64("MAX_BODY_BYTES", 64*1024),
        LogLevel:              envOr("LOG_LEVEL", "info"),
        StopSequences:         defaultStopSequences(os.Getenv("STOP_SEQUENCES")),
    ResetCodewords:        defaultResetCodewords(os.Getenv("RESET_CODEWORDS")),
    }
}

// withDefaultSystemInstructions uses provided env if set; otherwise a focused, safe default.
func withDefaultSystemInstructions(envVal string) string {
    if strings.TrimSpace(envVal) != "" {
        return envVal
    }
    return "" +
        "You are a WhatsApp auto-reply assistant for the user.\n" +
        "Goals: (1) Keep replies short and directly on-topic. (2) Ask at most one concise clarifying question if needed. (3) Never invent facts or make commitments.\n" +
        "Style: friendly, professional, 12 sentences max, mirror the users language.\n" +
        "Never: apologize, mention you are an AI/assistant/chatbot, mention tools/capabilities, or discuss policies.\n" +
        "Constraints: plain text only (no JSON, no code, no tags).\n"
}

// defaultStopSequences returns configured sequences or sensible defaults if none provided.
func defaultStopSequences(envVal string) []string {
    v := strings.TrimSpace(envVal)
    if v == "" {
        return []string{"User:", "Assistant:", "</s>", "```"}
    }
    // Support comma-separated list in env
    parts := splitCSV(v)
    if len(parts) == 0 {
        return nil
    }
    return parts
}

func defaultResetCodewords(envVal string) []string {
    // Comma-separated list; default to ["narnia"]
    v := strings.TrimSpace(envVal)
    if v == "" {
        return []string{"narnia"}
    }
    return splitCSV(v)
}

func splitCSV(v string) []string {
    if strings.TrimSpace(v) == "" {
        return nil
    }
    parts := strings.Split(v, ",")
    out := make([]string, 0, len(parts))
    for _, p := range parts {
        p = strings.TrimSpace(p)
        if p != "" {
            out = append(out, p)
        }
    }
    return out
}

func envOr(key, def string) string {
    if v := os.Getenv(key); v != "" {
        return v
    }
    return def
}
func envOrInt(key string, def int) int {
    if v := os.Getenv(key); v != "" {
        if n, err := strconv.Atoi(v); err == nil {
            return n
        }
    }
    return def
}
func envOrInt64(key string, def int64) int64 {
    if v := os.Getenv(key); v != "" {
        if n, err := strconv.ParseInt(v, 10, 64); err == nil {
            return n
        }
    }
    return def
}
func envOrFloat(key string, def float64) float64 {
    if v := os.Getenv(key); v != "" {
        if f, err := strconv.ParseFloat(v, 64); err == nil {
            return f
        }
    }
    return def
}

// Session and store
type Message struct {
    Role    string `json:"role"`
    Content string `json:"content"`
}

type Session struct {
    History        []Message
    LastTS         time.Time
    SessionNum     int
    RateWindow     time.Time
    RateCount      int
}

type SessionStore struct {
    cache *lru.Cache[string, *Session]
    ttl   time.Duration
    maxH  int
}

func NewSessionStore(capacity int, ttlSeconds int, historyLen int) *SessionStore {
    c, _ := lru.New[string, *Session](capacity)
    return &SessionStore{cache: c, ttl: time.Duration(ttlSeconds) * time.Second, maxH: historyLen}
}

func (s *SessionStore) GetOrInit(jid string) *Session {
    if sess, ok := s.cache.Get(jid); ok {
        // TTL reset behavior: if expired, start a new session
        if !sess.LastTS.IsZero() && s.ttl > 0 && time.Since(sess.LastTS) > s.ttl {
            sess = &Session{History: []Message{}, LastTS: time.Now(), SessionNum: sess.SessionNum + 1}
            s.cache.Add(jid, sess)
            return sess
        }
        return sess
    }
    sess := &Session{History: []Message{}, LastTS: time.Now(), SessionNum: 1}
    s.cache.Add(jid, sess)
    return sess
}

func (s *SessionStore) Append(jid string, m Message) *Session {
    sess := s.GetOrInit(jid)
    sess.History = append(sess.History, m)
    if len(sess.History) > s.maxH {
        sess.History = sess.History[len(sess.History)-s.maxH:]
    }
    sess.LastTS = time.Now()
    s.cache.Add(jid, sess)
    return sess
}

// Processed ID store for idempotency
type ProcessedStore struct {
    cache *lru.Cache[string, time.Time]
    ttl   time.Duration
}

func NewProcessedStore(capacity int, ttlSeconds int) *ProcessedStore {
    c, _ := lru.New[string, time.Time](capacity)
    return &ProcessedStore{cache: c, ttl: time.Duration(ttlSeconds) * time.Second}
}

func (p *ProcessedStore) Seen(id string) bool {
    if id == "" {
        return false
    }
    if ts, ok := p.cache.Get(id); ok {
        if time.Since(ts) < p.ttl {
            return true
        }
        // stale; treat as new and overwrite
    }
    p.cache.Add(id, time.Now())
    return false
}

// LM Studio integration
type LMStudioClient struct {
    baseURL string
    timeout time.Duration
    cfg     *Config
    // circuit breaker
    consecFailures int
    openUntil      time.Time
}

func NewLMStudioClient(cfg *Config) *LMStudioClient {
    base := fmt.Sprintf("http://%s:%d", cfg.LMStudioHost, cfg.LMStudioPort)
    return &LMStudioClient{baseURL: base, timeout: time.Duration(cfg.RequestTimeoutSeconds) * time.Second, cfg: cfg}
}

func (c *LMStudioClient) getActiveModel(ctx context.Context) string {
    pref := strings.TrimSpace(strings.ToLower(c.cfg.LMModel))
    type model struct{ ID string `json:"id"` }
    var out struct{
        Data []model `json:"data"`
    }
    req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/v1/models", nil)
    httpClient := &http.Client{Timeout: c.timeout}
    resp, err := httpClient.Do(req)
    if err != nil {
        log.Warn().Err(err).Msg("/v1/models failed")
        return c.cfg.LMModel
    }
    defer resp.Body.Close()
    if resp.StatusCode/100 != 2 {
        return c.cfg.LMModel
    }
    if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
        return c.cfg.LMModel
    }
    var ids []string
    for _, m := range out.Data {
        if m.ID != "" {
            ids = append(ids, m.ID)
        }
    }
    if pref == "" || pref == "auto" || pref == "*" {
        if len(ids) > 0 {
            return ids[0]
        }
        return c.cfg.LMModel
    }
    for _, id := range ids {
        if id == c.cfg.LMModel {
            return id
        }
    }
    if len(ids) > 0 {
        log.Info().Str("requested", c.cfg.LMModel).Str("using", ids[0]).Msg("model not found; using first available")
        return ids[0]
    }
    return c.cfg.LMModel
}

// listModels returns available model IDs from LM Studio.
func (c *LMStudioClient) listModels(ctx context.Context) []string {
    type model struct{ ID string `json:"id"` }
    var out struct{ Data []model `json:"data"` }
    req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/v1/models", nil)
    httpClient := &http.Client{Timeout: c.timeout}
    resp, err := httpClient.Do(req)
    if err != nil {
        log.Warn().Err(err).Msg("/v1/models failed")
        return nil
    }
    defer resp.Body.Close()
    if resp.StatusCode/100 != 2 {
        return nil
    }
    if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
        return nil
    }
    ids := make([]string, 0, len(out.Data))
    for _, m := range out.Data { if m.ID != "" { ids = append(ids, m.ID) } }
    return ids
}

func (c *LMStudioClient) callChat(ctx context.Context, messages []Message) (string, error) {
    if time.Now().Before(c.openUntil) {
        return "", errors.New("circuit open")
    }
    model := c.getActiveModel(ctx)
    payload := map[string]any{
        "model":       model,
        "messages":    messages,
        "temperature": c.cfg.Temperature,
        "top_p":       c.cfg.TopP,
        "max_tokens":  c.cfg.MaxTokens,
        "stream":      false,
        "frequency_penalty": c.cfg.FrequencyPenalty,
        "presence_penalty":  c.cfg.PresencePenalty,
    }
    if len(c.cfg.StopSequences) > 0 {
        payload["stop"] = c.cfg.StopSequences
    }
    b, _ := json.Marshal(payload)
    op := func() error {
        req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(b))
        req.Header.Set("Content-Type", "application/json")
        httpClient := &http.Client{Timeout: c.timeout}
        resp, err := httpClient.Do(req)
        if err != nil {
            return err
        }
        defer resp.Body.Close()
        if resp.StatusCode/100 != 2 {
            body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
            return fmt.Errorf("lm chat http %d: %s", resp.StatusCode, string(body))
        }
        var result map[string]any
        if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
            return err
        }
        ai := extractAIText(result)
        if ai == "" {
            return errors.New("empty ai_text")
        }
        // success, reset breaker
        c.consecFailures = 0
        return backoff.Permanent(&withResult{val: ai})
    }
    bo := backoff.NewExponentialBackOff()
    bo.InitialInterval = 300 * time.Millisecond
    bo.MaxElapsedTime = 3 * time.Second
    var wr *withResult
    _ = backoff.Retry(func() error {
        err := op()
        var ok bool
        wr, ok = err.(*withResult)
        if ok {
            return nil
        }
        return err
    }, bo)
    if wr != nil {
        return wr.val, nil
    }
    c.consecFailures++
    if c.consecFailures >= 5 {
        c.openUntil = time.Now().Add(10 * time.Second)
        log.Warn().Msg("circuit opened for 10s after repeated failures")
    }
    // fallback to completions
    prompt := messagesToPrompt(messages)
    payload2 := map[string]any{
        "model":       model,
        "prompt":      prompt,
        "temperature": c.cfg.Temperature,
        "top_p":       c.cfg.TopP,
        "max_tokens":  c.cfg.MaxTokens,
    }
    if len(c.cfg.StopSequences) > 0 {
        payload2["stop"] = c.cfg.StopSequences
    }
    payload2["frequency_penalty"] = c.cfg.FrequencyPenalty
    payload2["presence_penalty"] = c.cfg.PresencePenalty
    b2, _ := json.Marshal(payload2)
    req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/completions", bytes.NewReader(b2))
    req.Header.Set("Content-Type", "application/json")
    httpClient := &http.Client{Timeout: c.timeout}
    resp, err2 := httpClient.Do(req)
    if err2 != nil {
        return "", err2
    }
    defer resp.Body.Close()
    if resp.StatusCode/100 != 2 {
        body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
        return "", fmt.Errorf("lm completions http %d: %s", resp.StatusCode, string(body))
    }
    var result map[string]any
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return "", err
    }
    ai := extractAIText(result)
    if ai == "" {
        return "", errors.New("empty ai_text")
    }
    return ai, nil
}

type withResult struct{ val string }
func (w *withResult) Error() string { return "with result" }

func extractAIText(result map[string]any) string {
    if choices, ok := result["choices"].([]any); ok && len(choices) > 0 {
        if ch, ok := choices[0].(map[string]any); ok {
            if msg, ok := ch["message"].(map[string]any); ok {
                if c, ok := msg["content"].(string); ok {
                    return strings.TrimSpace(c)
                }
            }
            if t, ok := ch["text"].(string); ok {
                return strings.TrimSpace(t)
            }
        }
    }
    if c, ok := result["content"].(string); ok {
        return strings.TrimSpace(c)
    }
    return ""
}

func messagesToPrompt(messages []Message) string {
    var b strings.Builder
    start := 0
    if len(messages) > 8 {
        start = len(messages) - 8
    }
    for _, m := range messages[start:] {
        role := m.Role
        content := strings.TrimSpace(m.Content)
        if content == "" {
            continue
        }
        switch role {
        case "user":
            b.WriteString("User: ")
            b.WriteString(content)
            b.WriteString("\n")
        case "assistant":
            b.WriteString("Assistant: ")
            b.WriteString(content)
            b.WriteString("\n")
        default:
            b.WriteString(content)
            b.WriteString("\n")
        }
    }
    b.WriteString("Assistant:")
    return b.String()
}

// Request payload extraction
func extractMessageText(m map[string]any) string {
    candidates := []string{"text", "message", "body", "MESSAGE", "MESSAGE-TEXT", "message-text", "content"}
    for _, k := range candidates {
        if v, ok := m[k].(string); ok && strings.TrimSpace(v) != "" {
            return v
        }
    }
    for _, wrap := range []string{"data", "payload"} {
        if inner, ok := m[wrap].(map[string]any); ok {
            for _, k := range candidates {
                if v, ok := inner[k].(string); ok && strings.TrimSpace(v) != "" {
                    return v
                }
            }
        }
    }
    return ""
}

func strField(m map[string]any, keys ...string) string {
    for _, k := range keys {
        if v, ok := m[k]; ok {
            switch t := v.(type) {
            case string:
                if t != "" {
                    return t
                }
            }
        }
    }
    return ""
}

func contains(list []string, v string) bool {
    if len(list) == 0 {
        return true
    }
    for _, x := range list {
        if x == v {
            return true
        }
    }
    return false
}

func main() {
    cfg := defaultConfig()
    var cfgMu sync.RWMutex
    // Optional: read config.yaml if present
    if f, err := os.Open("config.yaml"); err == nil {
        defer f.Close()
        dec := yaml.NewDecoder(f)
        _ = dec.Decode(&cfg)
    }

    // logging
    lvl, err := zerolog.ParseLevel(strings.ToLower(cfg.LogLevel))
    if err != nil {
        lvl = zerolog.InfoLevel
    }
    zerolog.TimeFieldFormat = time.RFC3339
    zerolog.SetGlobalLevel(lvl)

    sessions := NewSessionStore(1000, cfg.InactivityTTLSeconds, cfg.HistoryLen)
    processed := NewProcessedStore(2000, cfg.ProcessedTTLSeconds)
    client := NewLMStudioClient(&cfg)

    mux := http.NewServeMux()

    mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodGet {
            w.WriteHeader(http.StatusMethodNotAllowed)
            return
        }
        cfgMu.RLock()
        resp := map[string]any{
            "status": "ok",
            "lm_studio_url": fmt.Sprintf("http://%s:%d", cfg.LMStudioHost, cfg.LMStudioPort),
            "configured_model": cfg.LMModel,
            "active_model": client.getActiveModel(r.Context()),
            "inactivity_ttl_seconds": cfg.InactivityTTLSeconds,
            "system_instructions": cfg.SystemInstructions != "",
            "tracked_sessions": sessions.cache.Len(),
        }
        cfgMu.RUnlock()
        writeJSON(w, http.StatusOK, resp)
    })

    mux.HandleFunc("/sessions", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodGet {
            w.WriteHeader(http.StatusMethodNotAllowed)
            return
        }
        writeJSON(w, http.StatusOK, map[string]any{
            "tracked_sessions": sessions.cache.Len(),
        })
    })

    mux.HandleFunc("/auto-reply", func(w http.ResponseWriter, r *http.Request) {
        reqID := uuid.New().String()
        lg := log.With().Str("req_id", reqID).Logger()
        // auth
        if cfg.WebhookSecret != "" {
            if r.Header.Get("X-Webhook-Secret") != cfg.WebhookSecret {
                lg.Warn().Msg("unauthorized: bad secret")
                writeJSON(w, http.StatusOK, map[string]string{"message": ""})
                return
            }
        }
        // cap body size
        r.Body = http.MaxBytesReader(w, r.Body, cfg.MaxBodyBytes)

        var data map[string]any
        ct := strings.ToLower(r.Header.Get("Content-Type"))
        if strings.Contains(ct, "application/json") {
            if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
                lg.Error().Err(err).Msg("bad json body")
                writeJSON(w, http.StatusOK, map[string]string{"message": ""})
                return
            }
        } else {
            if err := r.ParseForm(); err != nil {
                lg.Error().Err(err).Msg("bad form body")
                writeJSON(w, http.StatusOK, map[string]string{"message": ""})
                return
            }
            data = make(map[string]any, len(r.Form))
            for k := range r.Form {
                data[k] = r.Form.Get(k)
            }
        }

    senderJID := strField(data, "jid", "wa_id", "from")
        if senderJID == "" {
            senderJID = "unknown"
        }
        senderName := strField(data, "name", "profile_name", "chat_name")
        if senderName == "" {
            senderName = "User"
        }
        msgID := strField(data, "id", "message_id", "msg_id")
    messageText := extractMessageText(data)

        lg.Info().Str("jid", senderJID).Str("name", senderName).Str("preview", truncate(messageText, 200)).Msg("incoming message")

        if len(cfg.AllowedJIDs) > 0 && !contains(cfg.AllowedJIDs, senderJID) {
            lg.Info().Msg("unauthorized JID; ignoring")
            writeJSON(w, http.StatusOK, map[string]string{"message": ""})
            return
        }
        // Idempotency guard
        if processed.Seen(msgID) {
            lg.Info().Str("msg_id", msgID).Msg("duplicate message; ignoring")
            writeJSON(w, http.StatusOK, map[string]string{"message": ""})
            return
        }
        if strings.TrimSpace(messageText) == "" {
            writeJSON(w, http.StatusOK, map[string]string{"message": ""})
            return
        }

        // Reset command handling: if the message starts with a reset codeword, clear session
        // and, if any remainder exists, use it as the new message; otherwise reply empty.
        if rem, didReset := handleResetIfAny(&cfg, sessions, senderJID, messageText); didReset {
            if strings.TrimSpace(rem) == "" {
                lg.Info().Msg("reset command processed; no remainder; replying empty")
                writeJSON(w, http.StatusOK, map[string]string{"message": ""})
                return
            }
            messageText = rem
        }

        // Rate limiting per JID
        sess := sessions.GetOrInit(senderJID)
        now := time.Now()
        if sess.RateWindow.IsZero() || now.Sub(sess.RateWindow) > time.Minute {
            sess.RateWindow = now
            sess.RateCount = 0
        }
        if sess.RateCount >= cfg.RateLimitPerMinute {
            lg.Warn().Int("limit", cfg.RateLimitPerMinute).Msg("rate limited")
            writeJSON(w, http.StatusOK, map[string]string{"message": ""})
            return
        }
        sess.RateCount++
        sessions.cache.Add(senderJID, sess)

    // Build messages
    cfgMu.RLock()
        messages := make([]Message, 0, cfg.HistoryLen+3)
        if ins := strings.TrimSpace(cfg.SystemInstructions); ins != "" {
            messages = append(messages, Message{Role: "system", Content: ins})
        }
        if per, ok := cfg.PerJIDInstructions[senderJID]; ok && strings.TrimSpace(per) != "" {
            messages = append(messages, Message{Role: "system", Content: per})
        }
        messages = append(messages, sess.History...)
        messages = append(messages, Message{Role: "user", Content: messageText})
    cfgMu.RUnlock()

        // Call LM Studio with backoff and breaker
        ctx, cancel := context.WithTimeout(r.Context(), time.Duration(cfg.RequestTimeoutSeconds+2)*time.Second)
        defer cancel()
        aiText, err := client.callChat(ctx, messages)
        if err != nil || strings.TrimSpace(aiText) == "" {
            lg.Error().Err(err).Msg("lm studio failed; replying empty")
            writeJSON(w, http.StatusOK, map[string]string{"message": ""})
            return
        }

    // Post-process to keep it concise and on-topic
    aiText = sanitizeReply(messageText, aiText)

    lg.Info().Str("reply", truncate(aiText, 200)).Msg("sending reply")

        // Update session history
        sessions.Append(senderJID, Message{Role: "user", Content: messageText})
        sessions.Append(senderJID, Message{Role: "assistant", Content: aiText})

        writeJSON(w, http.StatusOK, map[string]string{"message": aiText})
    })

    // Admin UI: simple page to edit system instructions and switch model
    mux.HandleFunc("/admin", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodGet { w.WriteHeader(http.StatusMethodNotAllowed); return }
        w.Header().Set("Content-Type", "text/html; charset=utf-8")
        io.WriteString(w, adminHTML)
    })

    // Get current config subset for UI
    mux.HandleFunc("/admin/config", func(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case http.MethodGet:
            cfgMu.RLock()
            resp := map[string]any{
                "system_instructions": cfg.SystemInstructions,
                "lm_model": cfg.LMModel,
            }
            cfgMu.RUnlock()
            writeJSON(w, http.StatusOK, resp)
        case http.MethodPost, http.MethodPatch:
            var in struct{
                SystemInstructions string `json:"system_instructions"`
                LMModel string `json:"lm_model"`
            }
            if err := json.NewDecoder(r.Body).Decode(&in); err != nil { writeJSON(w, 400, map[string]string{"error":"bad json"}); return }
            cfgMu.Lock()
            if strings.TrimSpace(in.SystemInstructions) != "" { cfg.SystemInstructions = in.SystemInstructions }
            if strings.TrimSpace(in.LMModel) != "" { cfg.LMModel = in.LMModel }
            // Update client reference
            client.cfg = &cfg
            // Persist to config.yaml (best-effort)
            persistMinimalConfig(&cfg)
            cfgMu.Unlock()
            writeJSON(w, http.StatusOK, map[string]string{"status":"ok"})
        default:
            w.WriteHeader(http.StatusMethodNotAllowed)
        }
    })

    // List available models from LM Studio
    mux.HandleFunc("/admin/models", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodGet { w.WriteHeader(http.StatusMethodNotAllowed); return }
        ids := client.listModels(r.Context())
        writeJSON(w, http.StatusOK, map[string]any{"models": ids})
    })

    srv := &http.Server{
        Addr:              fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
        Handler:           loggingMiddleware(mux),
        ReadHeaderTimeout: 10 * time.Second,
        ReadTimeout:       15 * time.Second,
        WriteTimeout:      30 * time.Second,
        IdleTimeout:       60 * time.Second,
    }

    log.Info().Str("addr", srv.Addr).Msg("watusi bridge (go) listening")
    // Try to open admin UI on start (best-effort)
    go func(){
        time.Sleep(1200 * time.Millisecond)
        _ = openBrowser("http://127.0.0.1:" + strconv.Itoa(cfg.Port) + "/admin")
    }()
    if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
        log.Fatal().Err(err).Msg("server failed")
    }
}

func writeJSON(w http.ResponseWriter, code int, v any) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(code)
    _ = json.NewEncoder(w).Encode(v)
}

func truncate(s string, n int) string {
    s = strings.TrimSpace(s)
    if len(s) <= n {
        return s
    }
    return s[:n] + "…"
}

// sanitizeReply picks sentences with highest lexical overlap with the user message
// and trims the result to ~180 chars and max 2 sentences to stay concise.
func sanitizeReply(userText, aiText string) string {
    uTok := tokenize(userText)
    if len(uTok) == 0 {
        return firstNSentences(removeMeta(aiText), 2, 160)
    }
    sentences := splitSentences(removeMeta(aiText))
    if len(sentences) == 0 {
    return truncate(strings.TrimSpace(aiText), 160)
    }
    type cand struct { s string; score int }
    best := make([]cand, 0, len(sentences))
    for _, s := range sentences {
        tTok := tokenize(s)
        sc := overlap(uTok, tTok)
        best = append(best, cand{s: strings.TrimSpace(s), score: sc})
    }
    // pick top 2 by score (stable single-pass selection)
    var top1, top2 cand
    for _, c := range best {
        if c.score > top1.score {
            top2 = top1
            top1 = c
        } else if c.score > top2.score {
            top2 = c
        }
    }
    out := strings.TrimSpace(top1.s)
    if out == "" {
        return firstNSentences(aiText, 2, 160)
    }
    if top2.score > 0 && top2.s != "" && out != top2.s {
        out = out + " " + strings.TrimSpace(top2.s)
    }
    return truncate(out, 160)
}

func splitSentences(s string) []string {
    s = strings.TrimSpace(s)
    if s == "" {
        return nil
    }
    // naive split by ., !, ? while preserving simple cases
    var parts []string
    var b strings.Builder
    for _, r := range s {
        b.WriteRune(r)
        if r == '.' || r == '!' || r == '?' || r == '\n' {
            parts = append(parts, b.String())
            b.Reset()
        }
    }
    if b.Len() > 0 {
        parts = append(parts, b.String())
    }
    // trim whitespace
    for i := range parts {
        parts[i] = strings.TrimSpace(parts[i])
    }
    return parts
}

func firstNSentences(s string, n, limit int) string {
    ss := splitSentences(s)
    if len(ss) == 0 {
        return truncate(strings.TrimSpace(s), limit)
    }
    if len(ss) > n {
        ss = ss[:n]
    }
    out := strings.Join(ss, " ")
    return truncate(out, limit)
}

// removeMeta strips apology, AI/assistant disclaimers, and generic boilerplate.
func removeMeta(s string) string {
    s = strings.TrimSpace(s)
    if s == "" { return s }
    badStarts := []string{
        "i'm sorry", "i am sorry", "sorry", "as an ai", "as a chatbot", "as an assistant",
        "i don't have", "i do not have", "i cannot", "i can not", "i'm unable", "i am unable",
        "you can now use the chatbot", "as a whatsapp auto-reply assistant",
    }
    // Drop sentences that start with these
    parts := splitSentences(s)
    kept := make([]string, 0, len(parts))
    for _, p := range parts {
        pl := strings.ToLower(strings.TrimSpace(p))
        drop := false
        for _, b := range badStarts {
            if strings.HasPrefix(pl, b) {
                drop = true
                break
            }
        }
        if !drop {
            kept = append(kept, p)
        }
    }
    if len(kept) == 0 {
        return s
    }
    return strings.Join(kept, " ")
}

func tokenize(s string) map[string]struct{} {
    s = strings.ToLower(s)
    var b strings.Builder
    words := make([]string, 0, 32)
    for _, r := range s {
        if unicode.IsLetter(r) || unicode.IsDigit(r) {
            b.WriteRune(r)
        } else {
            if b.Len() > 0 {
                words = append(words, b.String())
                b.Reset()
            }
        }
    }
    if b.Len() > 0 {
        words = append(words, b.String())
    }
    m := make(map[string]struct{}, len(words))
    for _, w := range words {
        if len(w) <= 2 {
            continue
        }
        m[w] = struct{}{}
    }
    return m
}

func overlap(a, b map[string]struct{}) int {
    if len(a) == 0 || len(b) == 0 {
        return 0
    }
    sc := 0
    for k := range a {
        if _, ok := b[k]; ok {
            sc++
        }
    }
    return sc
}

// handleResetIfAny checks if the message starts with a reset codeword; if so, clears the session
// and returns the remainder of the message (after the codeword and trailing separators). The second
// return value indicates whether a reset was performed.
func handleResetIfAny(cfg *Config, sessions *SessionStore, jid, msg string) (string, bool) {
    original := strings.TrimSpace(msg)
    lower := strings.ToLower(original)
    for _, cw := range cfg.ResetCodewords {
        cw = strings.TrimSpace(strings.ToLower(cw))
        if cw == "" { continue }
        if strings.HasPrefix(lower, cw) {
            // Compute remainder by slicing the original string at cw length
            rem := strings.TrimSpace(original[len(cw):])
            // Trim common separators right after codeword
            rem = strings.TrimLeft(rem, ":-—– \t")
            // Clear session
            sess := sessions.GetOrInit(jid)
            sess.History = nil
            sess.SessionNum++
            sess.LastTS = time.Now()
            sessions.cache.Add(jid, sess)
            log.Info().Str("jid", jid).Str("codeword", cw).Msg("session reset via codeword")
            return rem, true
        }
    }
    return msg, false
}

// persistMinimalConfig writes selected fields back to config.yaml.
func persistMinimalConfig(cfg *Config) {
        m := map[string]any{
                "host": cfg.Host,
                "port": cfg.Port,
                "lm_studio_host": cfg.LMStudioHost,
                "lm_studio_port": cfg.LMStudioPort,
                "inactivity_ttl_seconds": cfg.InactivityTTLSeconds,
                "system_instructions": cfg.SystemInstructions,
                "lm_model": cfg.LMModel,
        }
        b, err := yaml.Marshal(m)
        if err != nil { return }
        _ = os.WriteFile("config.yaml", b, 0644)
}

// openBrowser opens the URL in the system default browser on Windows/macOS/Linux
func openBrowser(url string) error {
        var cmd *exec.Cmd
        switch runtime.GOOS {
        case "windows":
                cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
        case "darwin":
                cmd = exec.Command("open", url)
        default:
                cmd = exec.Command("xdg-open", url)
        }
        return cmd.Start()
}

// Minimal admin page (no external assets)
var adminHTML = `<!doctype html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Watusi Bridge Admin</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;max-width:760px}
        textarea{width:100%;height:160px;font-family:inherit}
        .row{margin:12px 0}
        label{display:block;font-weight:600;margin-bottom:6px}
        button{background:#0d6efd;color:#fff;border:0;padding:8px 14px;border-radius:6px;cursor:pointer}
        select{min-width:300px;padding:6px}
        .ok{color:#0a0}
        .err{color:#a00}
    </style>
    </head>
<body>
    <h2>Watusi Bridge Admin</h2>
    <div id="status">Loading…</div>
    <div class="row">
        <label for="model">Model</label>
        <select id="model"></select>
    </div>
    <div class="row">
        <label for="sys">System Instructions</label>
        <textarea id="sys" placeholder="Enter custom system instructions…"></textarea>
    </div>
    <div class="row">
        <button id="save">Save</button>
    </div>
    <script>
    async function load(){
        try{
            const [cfg, mods] = await Promise.all([
                fetch('/admin/config').then(r=>r.json()),
                fetch('/admin/models').then(r=>r.json())
            ]);
            const sel = document.getElementById('model');
            sel.innerHTML = '';
            const models = mods.models || [];
            const optAuto = document.createElement('option'); optAuto.value = 'auto'; optAuto.textContent = 'auto'; sel.appendChild(optAuto);
            for(const m of models){ const o = document.createElement('option'); o.value = m; o.textContent = m; sel.appendChild(o); }
            if(cfg.lm_model){ sel.value = cfg.lm_model; }
            document.getElementById('sys').value = cfg.system_instructions || '';
            document.getElementById('status').textContent = 'Online'; document.getElementById('status').className='ok';
        }catch(e){ document.getElementById('status').textContent = 'Offline'; document.getElementById('status').className='err'; }
    }
    async function save(){
        const body = { lm_model: document.getElementById('model').value, system_instructions: document.getElementById('sys').value };
        const r = await fetch('/admin/config',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
        if(r.ok){ alert('Saved.'); } else { alert('Save failed.'); }
    }
    document.getElementById('save').addEventListener('click', save);
    load();
    </script>
</body>
</html>`

// loggingMiddleware adds request IDs and structured logs.
func loggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        rid := r.Header.Get("X-Request-ID")
        if rid == "" {
            rid = uuid.New().String()
        }
        ww := &respWriter{ResponseWriter: w, status: 200}
        start := time.Now()
        next.ServeHTTP(ww, r)
        log.Info().
            Str("req_id", rid).
            Str("method", r.Method).
            Str("path", r.URL.Path).
            Int("status", ww.status).
            Dur("dur", time.Since(start)).
            Msg("http")
    })
}

type respWriter struct {
    http.ResponseWriter
    status int
}

func (rw *respWriter) WriteHeader(code int) {
    rw.status = code
    rw.ResponseWriter.WriteHeader(code)
}
