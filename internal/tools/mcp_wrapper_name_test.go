package tools

import (
	"errors"
	"regexp"
	"strings"
	"testing"
)

type trackingMCPCloser struct {
	closeCalls int
	err        error
}

func (c *trackingMCPCloser) Close() error {
	c.closeCalls++
	return c.err
}

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

func TestCloseMCPConnectionsClosesEverySessionDespiteCloseErrors(t *testing.T) {
	first := &trackingMCPCloser{err: errors.New("first close failed")}
	second := &trackingMCPCloser{}

	closeMCPConnections([]*trackingMCPCloser{first, second})

	if first.closeCalls != 1 || second.closeCalls != 1 {
		t.Fatalf("close calls = %d, %d; want 1, 1", first.closeCalls, second.closeCalls)
	}
}
