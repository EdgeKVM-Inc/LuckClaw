package lib

import (
	"context"
	"errors"
	"strings"
	"time"

	"luckclaw/internal/config"
	"luckclaw/internal/logging"
	"luckclaw/internal/providers/openaiapi"
)

// SingleShotBot sends one stateless, two-message request to one explicitly
// selected provider. It deliberately has no agent loop, tools, skills,
// sessions, memory, routing, retries, or logging.
type SingleShotBot struct {
	config       config.Config
	provider     *openaiapi.Client
	model        string
	modelWindow  int
	systemPrompt string
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
		ExtraHeaders: selected.ExtraHeaders,
		HTTPClient:   openaiapi.NewHTTPClientWithProxy(&cfg.Tools.Web, 120*time.Second),
		// Prompt caching rewrites a raw system string into content blocks.
		// Keep it disabled so the provider sees the exact caller-owned bytes.
		SupportsPromptCaching: false,
	}
	return &SingleShotBot{
		config:       cfg,
		provider:     provider,
		model:        cfg.ModelIDForAPI(model),
		modelWindow:  modelWindow,
		systemPrompt: systemPrompt,
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
		Temperature:     b.config.Agents.Defaults.Temperature,
		MaxTokens:       outputReserveTokens,
		ReasoningEffort: b.config.Agents.Defaults.ReasoningEffort,
		ResponseFormat:  &openaiapi.ResponseFormat{Type: "json_object"},
	})
	if err != nil {
		return "", errors.New("single-shot provider call failed")
	}
	if len(result.ToolCalls) != 0 {
		return "", errors.New("single-shot provider returned a tool call")
	}
	if strings.TrimSpace(result.Content) == "" {
		return "", errors.New("single-shot provider returned an empty response")
	}
	return result.Content, nil
}

func validOutputReserve(outputReserveTokens, modelWindowTokens int) bool {
	if modelWindowTokens <= 0 {
		return false
	}
	minimum := modelWindowTokens / 5
	if modelWindowTokens%5 != 0 {
		minimum++
	}
	if minimum < 4_000 {
		minimum = 4_000
	}
	return outputReserveTokens >= minimum && outputReserveTokens < modelWindowTokens
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
