package tools

import (
	"regexp"
	"strings"
	"testing"
)

func TestMCPToolWrapperNameIsProviderSafeWhileOriginalNameStaysDotted(t *testing.T) {
	wrapper := &MCPToolWrapper{
		ServerName:   "swarmboard_authoring",
		OriginalName: "capabilities.list",
	}

	if got, want := wrapper.Name(), "mcp_swarmboard_authoring_capabilities_list"; got != want {
		t.Fatalf("wrapper name = %q, want %q", got, want)
	}
	if !regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`).MatchString(wrapper.Name()) {
		t.Fatalf("wrapper name is not provider-safe: %q", wrapper.Name())
	}
	if wrapper.OriginalName != "capabilities.list" {
		t.Fatalf("original MCP name changed to %q", wrapper.OriginalName)
	}

	long := (&MCPToolWrapper{
		ServerName:   strings.Repeat("server", 20),
		OriginalName: strings.Repeat("tool.", 30),
	}).Name()
	if !regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`).MatchString(long) {
		t.Fatalf("long wrapper name is not provider-safe: %q", long)
	}
}

func TestMCPToolWrapperAliasCollisionFailsClosed(t *testing.T) {
	seen := map[string]string{}
	first := &MCPToolWrapper{ServerName: "server", OriginalName: "a.b"}
	second := &MCPToolWrapper{ServerName: "server", OriginalName: "a_b"}

	if err := claimMCPWrapperName(seen, first); err != nil {
		t.Fatal(err)
	}
	err := claimMCPWrapperName(seen, second)
	if err == nil {
		t.Fatal("provider-safe MCP wrapper collision was accepted")
	}
	if got, want := err.Error(), "MCP tool wrapper alias collision"; got != want {
		t.Fatalf("collision error = %q, want stable %q", got, want)
	}
}
