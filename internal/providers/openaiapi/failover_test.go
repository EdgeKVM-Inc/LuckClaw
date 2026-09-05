package openaiapi

import (
	"errors"
	"net"
	"net/http"
	"testing"
)

func TestHTTPFailureClassificationSeparatesProductionCategories(t *testing.T) {
	for _, test := range []struct {
		name   string
		status int
		body   string
		want   FailoverReason
	}{
		{name: "rate limit", status: http.StatusTooManyRequests, body: `{"code":"rate_limit_exceeded"}`, want: ReasonRateLimit},
		{name: "quota", status: http.StatusTooManyRequests, body: `{"code":"insufficient_quota"}`, want: ReasonBilling},
		{name: "authentication", status: http.StatusUnauthorized, want: ReasonAuth},
		{name: "permission", status: http.StatusForbidden, want: ReasonAuth},
		{name: "missing model", status: http.StatusNotFound, body: `{"code":"model_not_found"}`, want: ReasonModelNotFound},
		{name: "missing anthropic model", status: http.StatusNotFound, body: `{"type":"error","error":{"type":"not_found_error","message":"model: claude-sonnet-nope"}}`, want: ReasonModelNotFound},
		{name: "context window", status: http.StatusBadRequest, body: `{"message":"maximum context length exceeded"}`, want: ReasonContextWindow},
		{name: "deprecated parameter", status: http.StatusBadRequest, body: "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"`temperature` is deprecated for this model\"}}", want: ReasonBadParameter},
		{name: "unsupported parameter", status: http.StatusBadRequest, body: `{"error":{"message":"Unsupported parameter: 'temperature' is not supported with this model.","code":"unsupported_parameter"}}`, want: ReasonBadParameter},
		{name: "anthropic output cap", status: http.StatusBadRequest, body: `{"error":{"code":"invalid_request_error","message":"max_tokens: 200000 > 128000, which is the maximum allowed number of output tokens for claude-sonnet-5","type":"invalid_request_error"}}`, want: ReasonOutputCap},
		{name: "openai output cap", status: http.StatusBadRequest, body: `{"error":{"message":"max_tokens is too large: 300000. This model supports at most 128000 completion tokens.","type":"invalid_request_error"}}`, want: ReasonOutputCap},
		{name: "incompatible", status: http.StatusBadRequest, body: `{"message":"unsupported request format"}`, want: ReasonFormat},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := ClassifyHTTPError(test.status, test.body).Reason; got != test.want {
				t.Fatalf("reason=%q, want %q", got, test.want)
			}
		})
	}
}

func TestNetworkFailureClassificationPreservesTransportCause(t *testing.T) {
	dns := &net.DNSError{Name: "private.invalid", Err: "no such host"}
	failure := ClassifyNetworkError(dns)
	if failure.Reason != ReasonUnknown || !errors.Is(failure, dns) {
		t.Fatalf("failure=%#v", failure)
	}

	timeout := &net.DNSError{Name: "private.invalid", IsTimeout: true}
	if got := ClassifyNetworkError(timeout).Reason; got != ReasonTimeout {
		t.Fatalf("timeout reason=%q", got)
	}
}
