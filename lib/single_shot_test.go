package lib

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"luckclaw/internal/config"
)

type singleShotProviderRequest struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	} `json:"messages"`
	Tools []any `json:"tools"`
}

func TestSingleShotBotSendsOnlyCanonicalSystemAndCurrentContext(t *testing.T) {
	const (
		systemPrompt  = "canonical AGENTS.md\nraw-byte-sentinel\n"
		firstContext  = "first composed context\ncontext-secret-sentinel"
		secondContext = "second composed context"
		sessionID     = "same-conversation"
	)
	var (
		mu       sync.Mutex
		requests []singleShotProviderRequest
	)
	provider := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var probe singleShotProviderRequest
		if err := json.NewDecoder(request.Body).Decode(&probe); err != nil {
			http.Error(response, "bad request", http.StatusBadRequest)
			return
		}
		mu.Lock()
		requests = append(requests, probe)
		call := len(requests)
		mu.Unlock()
		writeProviderReply(t, response, `{"status":"answered","reply":"turn `+string(rune('0'+call))+`"}`, nil)
	}))
	t.Cleanup(provider.Close)

	configPath, workspace := writeSingleShotConfig(t, provider.URL)
	sessions := filepath.Join(workspace, "sessions")
	if err := os.Mkdir(sessions, 0o750); err != nil {
		t.Fatal(err)
	}
	sessionPath := filepath.Join(sessions, sessionID+".jsonl")
	const priorSession = "session-history-sentinel"
	if err := os.WriteFile(sessionPath, []byte(priorSession+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	bot, err := NewSingleShotBot(configPath, systemPrompt)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bot.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if reply, err := bot.Chat(ctx, firstContext, sessionID); err != nil || !strings.Contains(reply, "turn 1") {
		t.Fatalf("first reply=%q err=%v", reply, err)
	}
	if reply, err := bot.Chat(ctx, secondContext, sessionID); err != nil || !strings.Contains(reply, "turn 2") {
		t.Fatalf("second reply=%q err=%v", reply, err)
	}

	mu.Lock()
	got := append([]singleShotProviderRequest(nil), requests...)
	mu.Unlock()
	if len(got) != 2 {
		t.Fatalf("provider requests=%d, want 2", len(got))
	}
	for index, expectedContext := range []string{firstContext, secondContext} {
		request := got[index]
		if request.Model != "test-model" || len(request.Tools) != 0 || len(request.Messages) != 2 {
			t.Fatalf("request %d = %#v", index+1, request)
		}
		if request.Messages[0].Role != "system" || request.Messages[0].Content != systemPrompt ||
			request.Messages[1].Role != "user" || request.Messages[1].Content != expectedContext {
			t.Fatalf("request %d messages=%#v", index+1, request.Messages)
		}
	}
	secondEncoded, _ := json.Marshal(got[1])
	if bytes.Contains(secondEncoded, []byte(firstContext)) || bytes.Contains(secondEncoded, []byte(priorSession)) {
		t.Fatalf("second request contains prior context/session: %s", secondEncoded)
	}
	if current, err := os.ReadFile(sessionPath); err != nil || string(current) != priorSession+"\n" {
		t.Fatalf("session changed: %q err=%v", current, err)
	}
	if logs := bot.GetLogs(); len(logs) != 0 {
		encoded, _ := json.Marshal(logs)
		if bytes.Contains(encoded, []byte("sentinel")) {
			t.Fatalf("context/session sentinel reached logs: %s", encoded)
		}
		t.Fatalf("single-shot bot produced logs: %s", encoded)
	}
	if got := bot.ToolNames(); len(got) != 0 {
		t.Fatalf("single-shot tools=%v", got)
	}
}

func TestSingleShotBotNeverRetriesOrExecutesReturnedTools(t *testing.T) {
	for _, test := range []struct {
		name      string
		firstCode int
		content   string
		tools     []map[string]any
	}{
		{name: "transient provider error", firstCode: http.StatusServiceUnavailable},
		{name: "empty response", firstCode: http.StatusOK},
		{name: "unsolicited tool call", firstCode: http.StatusOK, content: "ignored", tools: []map[string]any{{
			"id": "call_1", "type": "function", "function": map[string]any{"name": "read_file", "arguments": `{}`},
		}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
				if calls.Add(1) == 1 {
					if test.firstCode != http.StatusOK {
						http.Error(response, "transient", test.firstCode)
						return
					}
					writeProviderReply(t, response, test.content, test.tools)
					return
				}
				writeProviderReply(t, response, "unexpected retry", nil)
			}))
			t.Cleanup(provider.Close)

			configPath, _ := writeSingleShotConfig(t, provider.URL)
			bot, err := NewSingleShotBot(configPath, "canonical prompt\n")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(bot.Close)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if _, err := bot.Chat(ctx, "current context", "ignored-session"); err == nil {
				t.Fatal("unsafe provider result accepted")
			}
			if calls.Load() != 1 {
				t.Fatalf("provider calls=%d, want 1", calls.Load())
			}
		})
	}
}

func TestSingleShotBotRejectsProviderRedirect(t *testing.T) {
	var sourceHits atomic.Int32
	var destinationHits atomic.Int32
	destination := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		destinationHits.Add(1)
		writeProviderReply(t, response, `{"status":"answered","reply":"redirected"}`, nil)
	}))
	t.Cleanup(destination.Close)
	source := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		sourceHits.Add(1)
		http.Redirect(response, request, destination.URL+"/chat/completions", http.StatusTemporaryRedirect)
	}))
	t.Cleanup(source.Close)

	configPath, _ := writeSingleShotConfig(t, source.URL)
	bot, err := NewSingleShotBot(configPath, "canonical prompt\n")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bot.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := bot.Chat(ctx, "current context", "ignored-session"); err == nil {
		t.Fatal("redirected provider response was accepted")
	}
	if sourceHits.Load() != 1 || destinationHits.Load() != 0 {
		t.Fatalf("redirect hits: source=%d destination=%d, want 1/0", sourceHits.Load(), destinationHits.Load())
	}
}

func writeSingleShotConfig(t *testing.T, providerURL string) (string, string) {
	t.Helper()
	workspace := t.TempDir()
	cfg := config.Default()
	cfg.Agents.Defaults.Workspace = workspace
	cfg.Agents.Defaults.Model = "openai/test-model"
	cfg.Agents.Defaults.Provider = "openai"
	cfg.Providers.OpenAI.APIKey = "test-key"
	cfg.Providers.OpenAI.APIBase = providerURL
	cfg.Models.ContextWindow = map[string]int{"openai/test-model": 32_000}
	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := config.Save(configPath, cfg); err != nil {
		t.Fatal(err)
	}
	return configPath, workspace
}

func writeProviderReply(t *testing.T, response http.ResponseWriter, content string, tools []map[string]any) {
	t.Helper()
	response.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(response).Encode(map[string]any{
		"choices": []any{map[string]any{
			"finish_reason": "stop",
			"message":       map[string]any{"content": content, "tool_calls": tools},
		}},
		"usage": map[string]any{},
	}); err != nil {
		t.Error(err)
	}
}
