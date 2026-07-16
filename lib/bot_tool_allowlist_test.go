package lib

import (
	"context"
	"sort"
	"strings"
	"testing"

	"luckclaw/internal/config"
	"luckclaw/internal/tools"
)

func TestWithToolAllowlistRestrictsEffectiveRegistry(t *testing.T) {
	bot := newPolicyTestBot(t, WithToolAllowlist("read_file"))

	if got, want := definitionNames(bot), []string{"read_file"}; !equalStrings(got, want) {
		t.Fatalf("tool definitions = %v, want %v", got, want)
	}
	if bot.agent.Tools.Get("read_file") == nil {
		t.Fatal("allowed tool read_file is missing from direct lookup")
	}

	disallowed := []string{
		"write_file", "edit_file", "list_dir", // filesystem
		"exec", "ssh", "terminal_transfer",
		"modbus_tcp", "mqtt",
		"web_search", "web_fetch", "browser",
		"cron", "web_design",
		"spawn", "subagent",
		"record_correction",
		"memory_search", "memory_get",
	}
	for _, name := range disallowed {
		t.Run(name, func(t *testing.T) {
			if got := bot.agent.Tools.Get(name); got != nil {
				t.Fatalf("direct lookup returned disallowed tool %q", name)
			}
			assertDirectExecutionRejected(t, bot, name)
		})
	}
}

func TestWithToolAllowlistEmptyMeansNoTools(t *testing.T) {
	bot := newPolicyTestBot(t, WithToolAllowlist())

	if got := definitionNames(bot); len(got) != 0 {
		t.Fatalf("empty allowlist exposed definitions: %v", got)
	}
	if got := bot.agent.Tools.ToolNames(); len(got) != 0 {
		t.Fatalf("empty allowlist registered tools: %v", got)
	}
	assertDirectExecutionRejected(t, bot, "read_file")
}

func TestWithToolAllowlistUnknownNameFailsClosed(t *testing.T) {
	const unknown = "not_a_real_luckclaw_tool"
	bot := newPolicyTestBot(t, WithToolAllowlist(unknown))

	if got := definitionNames(bot); len(got) != 0 {
		t.Fatalf("unknown allowlist entry exposed definitions: %v", got)
	}
	if got := bot.agent.Tools.Get(unknown); got != nil {
		t.Fatalf("unknown allowlist entry became directly lookupable: %v", got)
	}
	assertDirectExecutionRejected(t, bot, unknown)
}

func TestWithToolAllowlistRequiresExactToolNames(t *testing.T) {
	bot := newPolicyTestBot(t, WithToolAllowlist(" read_file "))

	if got := definitionNames(bot); len(got) != 0 {
		t.Fatalf("non-exact allowlist entry exposed definitions: %v", got)
	}
	assertDirectExecutionRejected(t, bot, "read_file")
}

func TestWithToolAllowlistRejectsLaterRegistration(t *testing.T) {
	bot := newPolicyTestBot(t, WithToolAllowlist("read_file"))

	bot.agent.Tools.Register(&tools.CronTool{})

	if got := bot.agent.Tools.Get("cron"); got != nil {
		t.Fatalf("late registration bypassed allowlist: %v", got)
	}
	assertDirectExecutionRejected(t, bot, "cron")
}

func TestNewBotWithoutAllowlistPreservesRegisteredTools(t *testing.T) {
	bot := newPolicyTestBot(t)

	for _, name := range []string{
		"read_file", "write_file", "exec", "ssh", "modbus_tcp", "mqtt",
		"web_search", "web_fetch", "browser", "cron", "web_design",
		"record_correction", "memory_search", "memory_get",
	} {
		if bot.agent.Tools.Get(name) == nil {
			t.Errorf("default NewBot omitted legacy tool %q", name)
		}
	}
}

func newPolicyTestBot(t *testing.T, options ...BotOption) *Bot {
	t.Helper()
	t.Setenv("LUCKCLAW_HOME", "")

	cfg := config.Default()
	cfg.Agents.Defaults.Workspace = t.TempDir()
	cfg.Agents.Defaults.Model = "openai/test-model"
	cfg.Agents.Defaults.Provider = "openai"
	cfg.Providers.OpenAI.APIKey = "test-key"
	cfg.Tools.Browser.Enabled = true
	cfg.Tools.Browser.RemoteURL = "ws://127.0.0.1:9222"
	cfg.Tools.MCPServers = nil

	cfgPath := t.TempDir() + "/config.json"
	if err := config.Save(cfgPath, cfg); err != nil {
		t.Fatalf("save test config: %v", err)
	}

	bot, err := NewBot(cfgPath, options...)
	if err != nil {
		t.Fatalf("NewBot: %v", err)
	}
	t.Cleanup(bot.Close)
	return bot
}

func definitionNames(bot *Bot) []string {
	defs := bot.agent.Tools.Definitions()
	names := make([]string, 0, len(defs))
	for _, def := range defs {
		names = append(names, def.Function.Name)
	}
	sort.Strings(names)
	return names
}

func assertDirectExecutionRejected(t *testing.T, bot *Bot, name string) {
	t.Helper()
	_, err := bot.agent.Tools.ExecuteJSON(context.Background(), name, `{}`)
	if err == nil {
		t.Fatalf("direct execution unexpectedly accepted tool %q", name)
	}
	if !strings.Contains(err.Error(), "unknown tool") {
		t.Fatalf("direct execution error for %q = %q, want unknown tool", name, err)
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
