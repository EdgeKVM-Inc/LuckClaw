package lib

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"luckclaw/internal/config"
)

func TestLegacySessionMigrationCanBeDisabled(t *testing.T) {
	home := t.TempDir()
	workspace := filepath.Join(t.TempDir(), "workspace")
	t.Setenv("HOME", home)
	t.Setenv("LUCKCLAW_HOME", "")
	t.Setenv("LUCKCLAW_CONFIG", "")

	const sessionID = "conversation-legacy"
	legacyPath := filepath.Join(home, ".luckclaw", "sessions", sessionID+".jsonl")
	writeSessionTranscript(t, legacyPath, sessionID, "legacy-only-secret")
	legacyBefore, err := os.ReadFile(legacyPath)
	if err != nil {
		t.Fatal(err)
	}

	bot := newSessionPolicyBot(t, workspace, WithLegacySessionMigrationDisabled())
	history, err := bot.GetHistory(sessionID)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 0 {
		t.Fatalf("legacy history was imported: %#v", history)
	}
	workspacePath := filepath.Join(workspace, "sessions", sessionID+".jsonl")
	if _, err := os.Lstat(workspacePath); !os.IsNotExist(err) {
		t.Fatalf("legacy transcript was copied into workspace: %v", err)
	}
	legacyAfter, err := os.ReadFile(legacyPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(legacyAfter) != string(legacyBefore) {
		t.Fatal("legacy transcript was modified")
	}
}

func TestWorkspaceSessionReloadsWithLegacyMigrationDisabled(t *testing.T) {
	home := t.TempDir()
	workspace := filepath.Join(t.TempDir(), "workspace")
	t.Setenv("HOME", home)
	t.Setenv("LUCKCLAW_HOME", "")
	t.Setenv("LUCKCLAW_CONFIG", "")

	const sessionID = "conversation-workspace"
	workspacePath := filepath.Join(workspace, "sessions", sessionID+".jsonl")
	writeSessionTranscript(t, workspacePath, sessionID, "workspace-only-message")

	first := newSessionPolicyBot(t, workspace, WithLegacySessionMigrationDisabled())
	firstHistory, err := first.GetHistory(sessionID)
	if err != nil {
		t.Fatal(err)
	}
	assertSingleHistoryMessage(t, firstHistory, "workspace-only-message")
	first.Close()

	second := newSessionPolicyBot(t, workspace, WithLegacySessionMigrationDisabled())
	secondHistory, err := second.GetHistory(sessionID)
	if err != nil {
		t.Fatal(err)
	}
	assertSingleHistoryMessage(t, secondHistory, "workspace-only-message")
}

func newSessionPolicyBot(t *testing.T, workspace string, options ...BotOption) *Bot {
	t.Helper()
	cfg := config.Default()
	cfg.Agents.Defaults.Workspace = workspace
	cfg.Agents.Defaults.Model = "openai/test-model"
	cfg.Agents.Defaults.Provider = "openai"
	cfg.Providers.OpenAI.APIKey = "test-key"
	cfg.Tools.MCPServers = nil
	cfgPath := filepath.Join(t.TempDir(), "config.json")
	if err := config.Save(cfgPath, cfg); err != nil {
		t.Fatal(err)
	}
	bot, err := NewBot(cfgPath, options...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bot.Close)
	return bot
}

func writeSessionTranscript(t *testing.T, path, sessionID, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	metadata, _ := json.Marshal(map[string]any{
		"_type":       "metadata",
		"session_key": sessionID,
		"created_at":  now,
		"updated_at":  now,
		"metadata":    map[string]any{},
	})
	message, _ := json.Marshal(map[string]any{
		"role":      "user",
		"content":   content,
		"timestamp": now,
	})
	data := append(metadata, '\n')
	data = append(data, message...)
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func assertSingleHistoryMessage(t *testing.T, history []map[string]any, want string) {
	t.Helper()
	if len(history) != 1 || history[0]["content"] != want {
		t.Fatalf("history = %#v, want one message %q", history, want)
	}
}
