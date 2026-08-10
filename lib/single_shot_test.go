package lib

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
	"luckclaw/internal/providers/openaiapi"
)

type singleShotProviderRequest struct {
	Model               string   `json:"model"`
	MaxTokens           int      `json:"max_tokens"`
	MaxCompletionTokens int      `json:"max_completion_tokens"`
	Temperature         *float64 `json:"temperature"`
	ResponseFormat      *struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Name   string         `json:"name"`
			Strict bool           `json:"strict"`
			Schema map[string]any `json:"schema"`
		} `json:"json_schema"`
	} `json:"response_format"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	} `json:"messages"`
	Tools []any `json:"tools"`
}

func TestProviderFailuresAreMappedWithoutProviderDetails(t *testing.T) {
	for _, test := range []struct {
		reason openaiapi.FailoverReason
		status int
		want   string
	}{
		{reason: openaiapi.ReasonRateLimit, status: http.StatusTooManyRequests, want: "provider_rate_limited"},
		{reason: openaiapi.ReasonAuth, status: http.StatusUnauthorized, want: "provider_auth_rejected"},
		{reason: openaiapi.ReasonAuth, status: http.StatusForbidden, want: "provider_permission_denied"},
		{reason: openaiapi.ReasonBilling, status: http.StatusPaymentRequired, want: "provider_quota_exceeded"},
		{reason: openaiapi.ReasonTimeout, want: "provider_timeout"},
		{reason: openaiapi.ReasonServer, status: http.StatusServiceUnavailable, want: "provider_unavailable"},
		{reason: openaiapi.ReasonFormat, status: http.StatusBadRequest, want: "provider_incompatible"},
		{reason: openaiapi.ReasonBadParameter, status: http.StatusBadRequest, want: "provider_parameter_unsupported"},
		{reason: openaiapi.ReasonModelNotFound, status: http.StatusBadRequest, want: "provider_model_not_found"},
		{reason: openaiapi.ReasonContextWindow, status: http.StatusBadRequest, want: "model_context_too_small"},
	} {
		t.Run(test.want, func(t *testing.T) {
			secret := "provider-secret-response-body"
			err := classifyProviderFailure(&openaiapi.FailoverError{
				Reason: test.reason,
				Status: test.status,
				Body:   secret,
			}, "openai")
			var failure *ProviderFailure
			if !errors.As(err, &failure) || failure.Code != test.want {
				t.Fatalf("failure=%v", err)
			}
			if strings.Contains(err.Error(), secret) {
				t.Fatal("provider detail escaped through bounded failure")
			}
		})
	}
}

func TestMissingOllamaModelUsesLocalFailureCategory(t *testing.T) {
	err := classifyProviderFailure(&openaiapi.FailoverError{
		Reason: openaiapi.ReasonModelNotFound,
		Status: http.StatusNotFound,
		Body:   "private provider response",
	}, "ollama")
	var failure *ProviderFailure
	if !errors.As(err, &failure) || failure.Code != "local_model_unavailable" {
		t.Fatalf("failure=%v", err)
	}
}

func TestSingleShotBotUsesCurrentOpenAIChatFields(t *testing.T) {
	var request singleShotProviderRequest
	provider := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, raw *http.Request) {
		if err := json.NewDecoder(raw.Body).Decode(&request); err != nil {
			http.Error(response, "bad request", http.StatusBadRequest)
			return
		}
		writeProviderReply(t, response, `{"status":"answered","reply":"bounded"}`, nil)
	}))
	t.Cleanup(provider.Close)

	configPath, _ := writeSingleShotCurrentOpenAIConfig(t, provider.URL)
	bot, err := NewSingleShotBot(configPath, "canonical prompt\n")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bot.Close)
	if _, err := bot.Chat(context.Background(), "current context", "session", 6_400); err != nil {
		t.Fatal(err)
	}
	if request.Model != "gpt-5.6-terra" || request.MaxCompletionTokens != 6_400 {
		t.Fatalf("current OpenAI request = %#v", request)
	}
	if request.MaxTokens != 0 || request.Temperature != nil {
		t.Fatalf("current OpenAI request kept unsupported controls: %#v", request)
	}
	if request.ResponseFormat == nil || request.ResponseFormat.Type != "json_schema" ||
		request.ResponseFormat.JSONSchema == nil || request.ResponseFormat.JSONSchema.Strict {
		t.Fatalf("current OpenAI response_format=%#v, want non-strict swarmboard_turn schema", request.ResponseFormat)
	}
}

func TestSingleShotBotOmitsUnsupportedAnthropicControls(t *testing.T) {
	// Anthropic's OpenAI-compatible endpoint rejects response_format, and
	// current Claude models reject temperature ("`temperature` is deprecated
	// for this model"), so an anthropic single-shot request must carry
	// neither. The live symptom was every provider validation failing as
	// provider_incompatible while the identical request without temperature
	// succeeded.
	var request singleShotProviderRequest
	provider := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, raw *http.Request) {
		if err := json.NewDecoder(raw.Body).Decode(&request); err != nil {
			http.Error(response, "bad request", http.StatusBadRequest)
			return
		}
		writeProviderReply(t, response, `{"status":"answered","reply":"bounded"}`, nil)
	}))
	t.Cleanup(provider.Close)

	configPath, _ := writeSingleShotAnthropicConfig(t, provider.URL)
	bot, err := NewSingleShotBot(configPath, "canonical prompt\n")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bot.Close)
	if _, err := bot.Chat(context.Background(), "current context", "session", 6_400); err != nil {
		t.Fatal(err)
	}
	if request.Model != "claude-test" || request.MaxTokens != 6_400 {
		t.Fatalf("anthropic request = %#v", request)
	}
	if request.Temperature != nil {
		t.Fatalf("anthropic request kept temperature: %#v", request)
	}
	if request.ResponseFormat != nil {
		t.Fatalf("anthropic request kept response_format: %#v", request)
	}
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
	if reply, err := bot.Chat(ctx, firstContext, sessionID, 6_400); err != nil || !strings.Contains(reply, "turn 1") {
		t.Fatalf("first reply=%q err=%v", reply, err)
	}
	if reply, err := bot.Chat(ctx, secondContext, sessionID, 6_400); err != nil || !strings.Contains(reply, "turn 2") {
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
		if request.Model != "test-model" || request.MaxTokens != 6_400 || len(request.Tools) != 0 || len(request.Messages) != 2 {
			t.Fatalf("request %d = %#v", index+1, request)
		}
		if request.ResponseFormat == nil || request.ResponseFormat.Type != "json_schema" ||
			request.ResponseFormat.JSONSchema == nil || request.ResponseFormat.JSONSchema.Strict {
			t.Fatalf("request %d response_format=%#v, want non-strict swarmboard_turn schema", index+1, request.ResponseFormat)
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

func TestSingleShotBotUsesStrictTurnSchemaForOllama(t *testing.T) {
	var request singleShotProviderRequest
	provider := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, raw *http.Request) {
		if err := json.NewDecoder(raw.Body).Decode(&request); err != nil {
			http.Error(response, "bad request", http.StatusBadRequest)
			return
		}
		writeProviderReply(t, response, `{"status":"answered","reply":"bounded","proposal":null}`, nil)
	}))
	t.Cleanup(provider.Close)

	configPath, _ := writeSingleShotOllamaConfig(t, provider.URL)
	bot, err := NewSingleShotBot(configPath, "canonical prompt\n")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bot.Close)
	if _, err := bot.Chat(context.Background(), "current context", "session", 6_400); err != nil {
		t.Fatal(err)
	}

	if request.ResponseFormat == nil || request.ResponseFormat.Type != "json_schema" ||
		request.ResponseFormat.JSONSchema == nil ||
		request.ResponseFormat.JSONSchema.Name != "swarmboard_turn" ||
		!request.ResponseFormat.JSONSchema.Strict {
		t.Fatalf("response_format=%#v, want strict swarmboard_turn schema", request.ResponseFormat)
	}
	schema := request.ResponseFormat.JSONSchema.Schema
	properties, _ := schema["properties"].(map[string]any)
	status, _ := properties["status"].(map[string]any)
	proposalUnion, _ := properties["proposal"].(map[string]any)
	proposalBranches, _ := proposalUnion["anyOf"].([]any)
	clarificationUnion, _ := properties["clarification"].(map[string]any)
	clarificationBranches, _ := clarificationUnion["anyOf"].([]any)
	if schema["type"] != "object" || schema["additionalProperties"] != false ||
		len(proposalBranches) != 2 || len(clarificationBranches) != 2 || status["enum"] == nil {
		t.Fatalf("response schema is incomplete: %#v", schema)
	}
	proposal, _ := proposalBranches[0].(map[string]any)
	proposalProperties, _ := proposal["properties"].(map[string]any)
	inputs, _ := proposalProperties["inputs"].(map[string]any)
	inputProperties, _ := inputs["properties"].(map[string]any)
	effects, _ := inputProperties["effects"].(map[string]any)
	effectDeclaration, _ := effects["items"].(map[string]any)
	effectBranches, _ := effectDeclaration["oneOf"].([]any)
	requestedEffects, _ := proposalProperties["requestedEffects"].(map[string]any)
	effectItems, _ := requestedEffects["items"].(map[string]any)
	requestedLimits, _ := proposalProperties["requestedLimits"].(map[string]any)
	limitProperties, _ := requestedLimits["properties"].(map[string]any)
	if inputs["additionalProperties"] != false || inputs["required"] == nil ||
		len(effectBranches) != 4 || effectItems["enum"] == nil {
		t.Fatalf("execution proposal schema does not constrain declarations: %#v", proposal)
	}
	controllerCall, _ := effectBranches[0].(map[string]any)
	controllerProperties, _ := controllerCall["properties"].(map[string]any)
	controllerKind, _ := controllerProperties["kind"].(map[string]any)
	controllerRequired, _ := controllerCall["required"].([]any)
	if controllerCall["additionalProperties"] != false || controllerKind["const"] != "controller.call" ||
		len(controllerRequired) != 3 || controllerProperties["operationId"] == nil ||
		controllerProperties["arguments"] == nil {
		t.Fatalf("controller call declaration schema is incomplete: %#v", controllerCall)
	}
	source, _ := proposalProperties["source"].(map[string]any)
	sourceDescription, _ := source["description"].(string)
	if !strings.Contains(sourceDescription, "from swarmboard import controller") {
		t.Fatalf("proposal source schema omits restricted-client import guidance: %#v", source)
	}
	if requestedLimits["additionalProperties"] != false || len(limitProperties) != 6 {
		t.Fatalf("execution proposal schema does not constrain resource limits: %#v", requestedLimits)
	}
	for name, maximum := range map[string]int{
		"cpuPercent": 25, "externalCalls": 64, "memoryMiB": 48,
		"outputBytes": 1_048_576, "pids": 16, "wallTimeSeconds": 30,
	} {
		property, _ := limitProperties[name].(map[string]any)
		if property["type"] != "integer" || property["minimum"] != float64(1) ||
			property["maximum"] != float64(maximum) {
			t.Fatalf("resource limit %q schema=%#v", name, property)
		}
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
			if _, err := bot.Chat(ctx, "current context", "ignored-session", 6_400); err == nil {
				t.Fatal("unsafe provider result accepted")
			} else if test.name != "transient provider error" {
				var failure *ProviderFailure
				if !errors.As(err, &failure) || failure.Code != "invalid_model_reply" {
					t.Fatalf("failure=%v, want invalid_model_reply", err)
				}
			}
			if calls.Load() != 1 {
				t.Fatalf("provider calls=%d, want 1", calls.Load())
			}
		})
	}
}

func TestSingleShotBotClassifiesEmptyProviderRepliesWithoutLeakingDetails(t *testing.T) {
	for _, test := range []struct {
		name         string
		finishReason string
		refusal      string
		want         string
	}{
		{name: "output exhausted", finishReason: "length", want: "provider_output_exhausted"},
		{name: "content filter", finishReason: "content_filter", want: "provider_refused"},
		{name: "explicit refusal", finishReason: "stop", refusal: "private refusal detail", want: "provider_refused"},
	} {
		t.Run(test.name, func(t *testing.T) {
			provider := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
				response.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(response).Encode(map[string]any{
					"choices": []any{map[string]any{
						"finish_reason": test.finishReason,
						"message": map[string]any{
							"content": "", "refusal": test.refusal, "tool_calls": []any{},
						},
					}},
					"usage": map[string]any{},
				})
			}))
			t.Cleanup(provider.Close)

			configPath, _ := writeSingleShotConfig(t, provider.URL)
			bot, err := NewSingleShotBot(configPath, "canonical prompt\n")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(bot.Close)
			_, err = bot.Chat(context.Background(), "current context", "ignored-session", 6_400)
			var failure *ProviderFailure
			if !errors.As(err, &failure) || failure.Code != test.want {
				t.Fatalf("failure=%v, want %s", err, test.want)
			}
			if test.refusal != "" && strings.Contains(err.Error(), test.refusal) {
				t.Fatal("provider refusal escaped through bounded failure")
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
	if _, err := bot.Chat(ctx, "current context", "ignored-session", 6_400); err == nil ||
		err.Error() != "single-shot provider call failed" {
		t.Fatalf("redirected provider response returned unbounded or missing error: %v", err)
	}
	if sourceHits.Load() != 1 || destinationHits.Load() != 0 {
		t.Fatalf("redirect hits: source=%d destination=%d, want 1/0", sourceHits.Load(), destinationHits.Load())
	}
}

func TestSingleShotBotUsesOnlyValidOutputReserve(t *testing.T) {
	for _, test := range []struct {
		name          string
		window        int
		outputReserve int
		wantCall      bool
	}{
		{name: "8k exact cap", window: 8_000, outputReserve: 4_000, wantCall: true},
		{name: "32k exact cap", window: 32_000, outputReserve: 6_400, wantCall: true},
		{name: "200k keeps a full fifth", window: 200_000, outputReserve: 40_000, wantCall: true},
		{name: "zero", window: 32_000, outputReserve: 0},
		{name: "negative", window: 32_000, outputReserve: -1},
		{name: "below 8k minimum", window: 8_000, outputReserve: 3_999},
		{name: "below 32k minimum", window: 32_000, outputReserve: 6_399},
		{name: "200k rejects a capped reserve", window: 200_000, outputReserve: 8_192},
		{name: "reaches window", window: 32_000, outputReserve: 32_000},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			var request singleShotProviderRequest
			provider := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, providerRequest *http.Request) {
				calls.Add(1)
				if err := json.NewDecoder(providerRequest.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				writeProviderReply(t, response, `{"status":"answered","reply":"bounded"}`, nil)
			}))
			t.Cleanup(provider.Close)

			configPath, _ := writeSingleShotConfigForWindow(t, provider.URL, test.window)
			bot, err := NewSingleShotBot(configPath, "canonical prompt\n")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(bot.Close)
			_, err = bot.Chat(context.Background(), "current context", "ignored-session", test.outputReserve)
			if test.wantCall {
				if err != nil || calls.Load() != 1 || request.MaxTokens != test.outputReserve {
					t.Fatalf("error=%v calls=%d max_tokens=%d, want nil/1/%d", err, calls.Load(), request.MaxTokens, test.outputReserve)
				}
				return
			}
			if err == nil || calls.Load() != 0 {
				t.Fatalf("invalid cap error=%v provider calls=%d", err, calls.Load())
			}
		})
	}
}

func writeSingleShotConfig(t *testing.T, providerURL string) (string, string) {
	return writeSingleShotConfigForWindow(t, providerURL, 32_000)
}

func writeSingleShotConfigForWindow(t *testing.T, providerURL string, modelWindowTokens int) (string, string) {
	t.Helper()
	workspace := t.TempDir()
	cfg := config.Default()
	cfg.Agents.Defaults.Workspace = workspace
	cfg.Agents.Defaults.Model = "openai/test-model"
	cfg.Agents.Defaults.Provider = "openai"
	cfg.Providers.OpenAI.APIKey = "test-key"
	cfg.Providers.OpenAI.APIBase = providerURL
	cfg.Models.ContextWindow = map[string]int{"openai/test-model": modelWindowTokens}
	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := config.Save(configPath, cfg); err != nil {
		t.Fatal(err)
	}
	return configPath, workspace
}

func writeSingleShotCurrentOpenAIConfig(t *testing.T, providerURL string) (string, string) {
	t.Helper()
	workspace := t.TempDir()
	cfg := config.Default()
	cfg.Agents.Defaults.Workspace = workspace
	cfg.Agents.Defaults.Model = "openai/gpt-5.6-terra"
	cfg.Agents.Defaults.Provider = "openai"
	cfg.Providers.OpenAI.APIKey = "test-key"
	cfg.Providers.OpenAI.APIBase = providerURL
	cfg.Models.ContextWindow = map[string]int{"openai/gpt-5.6-terra": 32_000}
	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := config.Save(configPath, cfg); err != nil {
		t.Fatal(err)
	}
	return configPath, workspace
}

func writeSingleShotAnthropicConfig(t *testing.T, providerURL string) (string, string) {
	t.Helper()
	workspace := t.TempDir()
	cfg := config.Default()
	cfg.Agents.Defaults.Workspace = workspace
	cfg.Agents.Defaults.Model = "anthropic/claude-test"
	cfg.Agents.Defaults.Provider = "anthropic"
	cfg.Providers.Anthropic.APIKey = "test-key"
	cfg.Providers.Anthropic.APIBase = providerURL
	cfg.Models.ContextWindow = map[string]int{"anthropic/claude-test": 32_000}
	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := config.Save(configPath, cfg); err != nil {
		t.Fatal(err)
	}
	return configPath, workspace
}

func writeSingleShotOllamaConfig(t *testing.T, providerURL string) (string, string) {
	t.Helper()
	workspace := t.TempDir()
	cfg := config.Default()
	cfg.Agents.Defaults.Workspace = workspace
	cfg.Agents.Defaults.Model = "ollama/test-model"
	cfg.Agents.Defaults.Provider = "ollama"
	cfg.Providers.Ollama.APIBase = providerURL
	cfg.Models.ContextWindow = map[string]int{"ollama/test-model": 32_000}
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
