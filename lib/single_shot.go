package lib

import (
	"context"
	"errors"
	"net"
	"strings"
	"time"

	"luckclaw/internal/config"
	"luckclaw/internal/logging"
	"luckclaw/internal/providers/openaiapi"
)

// ProviderFailure is the bounded provider classification exposed to the
// controller adapter. It never includes response bodies, headers, or secrets.
type ProviderFailure struct {
	Code string
}

func (e *ProviderFailure) Error() string {
	return "single-shot provider call failed"
}

// SingleShotBot sends one stateless, two-message request to one explicitly
// selected provider. It deliberately has no agent loop, tools, skills,
// sessions, memory, routing, retries, or logging.
type SingleShotBot struct {
	config         config.Config
	provider       *openaiapi.Client
	model          string
	modelWindow    int
	systemPrompt   string
	temperature    float64
	responseFormat *openaiapi.ResponseFormat
}

func NewSingleShotBot(configPath, systemPrompt string) (*SingleShotBot, error) {
	if strings.TrimSpace(configPath) == "" {
		return nil, errors.New("single-shot config path is required")
	}
	if len(systemPrompt) == 0 {
		return nil, errors.New("single-shot system prompt is required")
	}
	cfg, err := config.Load(configPath)
	if err != nil {
		return nil, errors.New("single-shot config could not be loaded")
	}
	// Config loading overlays the file onto defaults, so an explicit empty
	// mcpServers object otherwise retains the example default entry. The
	// adapter prevalidates that the file contains exactly an empty object.
	cfg.Tools.MCPServers = map[string]config.MCPServerConfig{}
	model := strings.TrimSpace(cfg.Agents.Defaults.Model)
	providerName := strings.ToLower(strings.TrimSpace(cfg.Agents.Defaults.Provider))
	if model == "" || providerName == "" || providerName == "auto" {
		return nil, errors.New("single-shot provider identity is not explicit")
	}
	selected := cfg.SelectProvider(model)
	if selected == nil || selected.Name != providerName {
		return nil, errors.New("single-shot provider does not match the explicit selection")
	}
	modelWindow, exactWindow := cfg.Models.ContextWindow[model]
	if !exactWindow || modelWindow <= 0 {
		return nil, errors.New("single-shot model window is unavailable")
	}
	provider := &openaiapi.Client{
		APIKey:       selected.APIKey,
		APIBase:      selected.APIBase,
		Provider:     selected.Name,
		ExtraHeaders: selected.ExtraHeaders,
		HTTPClient:   openaiapi.NewHTTPClientWithProxy(&cfg.Tools.Web, 120*time.Second),
		// Prompt caching rewrites a raw system string into content blocks.
		// Keep it disabled so the provider sees the exact caller-owned bytes.
		SupportsPromptCaching: false,
	}
	responseFormat := &openaiapi.ResponseFormat{Type: "json_object"}
	if providerName == "openai" || providerName == "ollama" {
		responseFormat = swarmboardTurnResponseFormat(providerName == "ollama")
	}
	temperature := cfg.Agents.Defaults.Temperature
	if providerName == "anthropic" {
		// Anthropic's OpenAI-compatible endpoint rejects response_format, and
		// current Claude models reject temperature outright ("`temperature` is
		// deprecated for this model"). Omit both; the JSON reply shape is
		// enforced by the system prompt and the caller-side reply validation.
		responseFormat = nil
		temperature = 0
	}
	return &SingleShotBot{
		config:         cfg,
		provider:       provider,
		model:          cfg.ModelIDForAPI(model),
		modelWindow:    modelWindow,
		systemPrompt:   systemPrompt,
		temperature:    temperature,
		responseFormat: responseFormat,
	}, nil
}

func (b *SingleShotBot) Chat(ctx context.Context, contextText, _ string, outputReserveTokens int) (string, error) {
	if b == nil || b.provider == nil {
		return "", errors.New("single-shot provider is unavailable")
	}
	if !validOutputReserve(outputReserveTokens, b.modelWindow) {
		return "", errors.New("single-shot output reserve is outside the model window")
	}
	result, err := b.provider.Chat(ctx, openaiapi.ChatRequest{
		Model: b.model,
		Messages: []openaiapi.Message{
			{Role: "system", Content: b.systemPrompt},
			{Role: "user", Content: contextText},
		},
		Temperature:     b.temperature,
		MaxTokens:       outputReserveTokens,
		ReasoningEffort: b.config.Agents.Defaults.ReasoningEffort,
		ResponseFormat:  b.responseFormat,
	})
	if err != nil {
		return "", classifyProviderFailure(err, b.provider.Provider)
	}
	if len(result.ToolCalls) != 0 {
		return "", &ProviderFailure{Code: "invalid_model_reply"}
	}
	if strings.TrimSpace(result.Content) == "" {
		if strings.TrimSpace(result.Refusal) != "" || strings.EqualFold(result.FinishReason, "content_filter") {
			return "", &ProviderFailure{Code: "provider_refused"}
		}
		if strings.EqualFold(result.FinishReason, "length") {
			return "", &ProviderFailure{Code: "provider_output_exhausted"}
		}
		return "", &ProviderFailure{Code: "invalid_model_reply"}
	}
	return result.Content, nil
}

func classifyProviderFailure(err error, provider string) error {
	var providerError *openaiapi.FailoverError
	if !errors.As(err, &providerError) {
		return &ProviderFailure{Code: "provider_request_failed"}
	}
	code := "provider_request_failed"
	switch providerError.Reason {
	case openaiapi.ReasonRateLimit:
		code = "provider_rate_limited"
	case openaiapi.ReasonAuth:
		if providerError.Status == 403 {
			code = "provider_permission_denied"
		} else {
			code = "provider_auth_rejected"
		}
	case openaiapi.ReasonBilling:
		code = "provider_quota_exceeded"
	case openaiapi.ReasonTimeout:
		code = "provider_timeout"
	case openaiapi.ReasonServer:
		code = "provider_unavailable"
	case openaiapi.ReasonFormat:
		code = "provider_incompatible"
	case openaiapi.ReasonBadParameter:
		code = "provider_parameter_unsupported"
	case openaiapi.ReasonOutputCap:
		code = "model_output_limit_exceeded"
	case openaiapi.ReasonModelNotFound:
		if strings.EqualFold(strings.TrimSpace(provider), "ollama") {
			code = "local_model_unavailable"
		} else {
			code = "provider_model_not_found"
		}
	case openaiapi.ReasonContextWindow:
		code = "model_context_too_small"
	case openaiapi.ReasonUnknown:
		var dnsError *net.DNSError
		switch {
		case errors.As(providerError.Wrapped, &dnsError):
			code = "provider_dns_failed"
		case providerError.Wrapped != nil && containsTLSError(providerError.Wrapped.Error()):
			code = "provider_tls_failed"
		default:
			code = "provider_unreachable"
		}
	}
	return &ProviderFailure{Code: code}
}

func containsTLSError(message string) bool {
	lower := strings.ToLower(message)
	return strings.Contains(lower, "tls") ||
		strings.Contains(lower, "x509") ||
		strings.Contains(lower, "certificate") ||
		strings.Contains(lower, "unknown authority")
}

// minimumOutputReserveTokens is the smallest completion budget a single-shot
// turn may request, regardless of the model window.
const minimumOutputReserveTokens = 4_000

func validOutputReserve(outputReserveTokens, modelWindowTokens int) bool {
	if modelWindowTokens <= 0 {
		return false
	}
	// A fixed floor, not a window share. Output capacity is a model property
	// independent of the context window: the controller chooses the reserve
	// per plan type and narrows it on the provider's typed output-limit
	// errors (classified below as model_output_limit_exceeded), so this only
	// enforces the sanity bounds. Deriving the floor from the window rejected
	// models whose context was fine: a 1.1M-token window demanded a 220k
	// completion the model's output cap refused, and the swarm-controller
	// adapter's 4k validation probe could not activate any window above 20k.
	return outputReserveTokens >= minimumOutputReserveTokens && outputReserveTokens < modelWindowTokens
}

func swarmboardTurnResponseFormat(strict bool) *openaiapi.ResponseFormat {
	effectDeclaration := map[string]any{
		"oneOf": []any{
			map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"properties": map[string]any{
					"kind":        map[string]any{"const": "controller.call"},
					"operationId": map[string]any{"type": "string", "minLength": 1},
					"arguments":   map[string]any{"type": "object"},
				},
				"required": []string{"kind", "operationId", "arguments"},
			},
			map[string]any{
				"type":                 "object",
				"additionalProperties": true,
				"properties": map[string]any{
					"kind": map[string]any{
						"type": "string",
						"enum": []string{"sds.read", "sds.condition", "sds.write_config"},
					},
					"device":         map[string]any{"type": "string", "minLength": 1},
					"appId":          map[string]any{"type": "string", "minLength": 1},
					"schemaRevision": map[string]any{"type": "string", "minLength": 1},
					"fieldPath":      map[string]any{"type": "string", "minLength": 1},
				},
				"required": []string{"kind", "device", "appId", "schemaRevision", "fieldPath"},
			},
			map[string]any{
				"type":                 "object",
				"additionalProperties": true,
				"properties": map[string]any{
					"kind": map[string]any{
						"type": "string",
						"enum": []string{"workflow.save", "schedule.save"},
					},
				},
				"required": []string{"kind"},
			},
			map[string]any{
				"type":                 "object",
				"additionalProperties": true,
				"properties": map[string]any{
					"kind": map[string]any{
						"type": "string",
						"enum": []string{"memory.read", "memory.write"},
					},
					"key": map[string]any{"type": "string", "minLength": 1},
				},
				"required": []string{"kind", "key"},
			},
		},
	}
	proposal := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"purpose": map[string]any{"type": "string", "minLength": 1},
			"source": map[string]any{
				"type":        "string",
				"minLength":   1,
				"description": "Complete Python source with exactly one def main(context), not JSON or a filename. Import every used restricted client; controller calls require from swarmboard import controller.",
			},
			"inputs": map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"properties": map[string]any{
					"effects": map[string]any{
						"type":  "array",
						"items": effectDeclaration,
					},
				},
				"required": []string{"effects"},
			},
			"requestedTargets": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string", "minLength": 1},
				"uniqueItems": true,
			},
			"requestedEffects": map[string]any{
				"type":        "array",
				"uniqueItems": true,
				"items": map[string]any{
					"type": "string",
					"enum": []string{
						"read",
						"config_mutation",
						"controller_mutation",
						"workflow_schedule_mutation",
						"memory_mutation",
						"high_risk_preparation",
					},
				},
			},
			"requestedLimits": map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"description":          "Optional sandbox resource ceilings only; API query arguments belong in inputs.effects.",
				"properties": map[string]any{
					"cpuPercent":      map[string]any{"type": "integer", "minimum": 1, "maximum": 25},
					"externalCalls":   map[string]any{"type": "integer", "minimum": 1, "maximum": 64},
					"memoryMiB":       map[string]any{"type": "integer", "minimum": 1, "maximum": 48},
					"outputBytes":     map[string]any{"type": "integer", "minimum": 1, "maximum": 1_048_576},
					"pids":            map[string]any{"type": "integer", "minimum": 1, "maximum": 16},
					"wallTimeSeconds": map[string]any{"type": "integer", "minimum": 1, "maximum": 30},
				},
			},
		},
		"required": []string{
			"purpose",
			"source",
			"inputs",
			"requestedTargets",
			"requestedEffects",
			"requestedLimits",
		},
	}
	clarification := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"kind": map[string]any{
				"type": "string",
				"enum": []string{"ambiguous_resource", "missing_argument", "missing_decision"},
			},
			"field": map[string]any{"type": "string", "minLength": 1},
			"resourceType": map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string", "minLength": 1},
					map[string]any{"type": "null"},
				},
			},
			"candidateValues": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string", "minLength": 1},
			},
		},
		"required": []string{"kind", "field", "resourceType", "candidateValues"},
	}
	groundingSelector := map[string]any{
		"type":        "array",
		"maxItems":    8,
		"uniqueItems": true,
		"items":       map[string]any{"type": "string", "minLength": 1, "maxLength": 128},
	}
	grounding := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties": map[string]any{
			"queries": map[string]any{
				"type":     "array",
				"minItems": 1,
				"maxItems": 4,
				"items": map[string]any{
					"type":                 "object",
					"additionalProperties": false,
					"properties": map[string]any{
						"kind": map[string]any{
							"type": "string",
							"enum": []string{"app_schema", "field_values", "devices", "health", "workflows"},
						},
						"appIds":    groundingSelector,
						"deviceIds": groundingSelector,
						"fieldPaths": map[string]any{
							"type":        "array",
							"maxItems":    8,
							"uniqueItems": true,
							"items":       map[string]any{"type": "string", "minLength": 1, "maxLength": 256},
						},
					},
					// Selectors are optional per kind; only the read kind is required.
					"required": []string{"kind"},
				},
			},
		},
		"required": []string{"queries"},
	}
	return &openaiapi.ResponseFormat{
		Type: "json_schema",
		JSONSchema: &openaiapi.JSONSchemaResponseFormat{
			Name:   "swarmboard_turn",
			Strict: strict,
			Schema: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"properties": map[string]any{
					"status": map[string]any{
						"type": "string",
						"enum": []string{"answered", "awaiting_clarification", "execution_proposed", "needs_grounding"},
					},
					"reply": map[string]any{"type": "string", "minLength": 1},
					"proposal": map[string]any{
						"anyOf": []any{proposal, map[string]any{"type": "null"}},
					},
					"clarification": map[string]any{
						"anyOf": []any{clarification, map[string]any{"type": "null"}},
					},
					"grounding": map[string]any{
						"anyOf": []any{grounding, map[string]any{"type": "null"}},
					},
				},
				"required": []string{"status", "reply"},
			},
		},
	}
}

func (b *SingleShotBot) Close() {
	if b != nil && b.provider != nil && b.provider.HTTPClient != nil {
		b.provider.HTTPClient.CloseIdleConnections()
	}
}

func (*SingleShotBot) ToolNames() []string { return nil }

func (b *SingleShotBot) GetConfig() (interface{}, error) {
	if b == nil {
		return nil, errors.New("single-shot config is unavailable")
	}
	return b.config, nil
}

func (*SingleShotBot) GetLogs() []logging.Entry { return nil }
