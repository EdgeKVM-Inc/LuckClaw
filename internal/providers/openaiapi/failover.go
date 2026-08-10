package openaiapi

import (
	"errors"
	"fmt"
	"net"
	"strings"
)

type FailoverReason string

const (
	ReasonRateLimit     FailoverReason = "rate_limit"
	ReasonAuth          FailoverReason = "auth"
	ReasonBilling       FailoverReason = "billing"
	ReasonTimeout       FailoverReason = "timeout"
	ReasonServer        FailoverReason = "server"
	ReasonFormat        FailoverReason = "format"
	ReasonBadParameter  FailoverReason = "bad_parameter"
	ReasonOutputCap     FailoverReason = "output_cap"
	ReasonModelNotFound FailoverReason = "model_not_found"
	ReasonContextWindow FailoverReason = "context_window"
	ReasonUnknown       FailoverReason = "unknown"
)

// FailoverError is a structured error returned by the provider client.
// It classifies API failures so callers can decide whether to retry,
// fallback, or abort immediately.
type FailoverError struct {
	Reason  FailoverReason
	Status  int
	Body    string
	Wrapped error
}

func (e *FailoverError) Error() string {
	if e.Status > 0 {
		return fmt.Sprintf("provider error [%s]: status=%d body=%s", e.Reason, e.Status, e.Body)
	}
	if e.Wrapped != nil {
		return fmt.Sprintf("provider error [%s]: %v", e.Reason, e.Wrapped)
	}
	return fmt.Sprintf("provider error [%s]", e.Reason)
}

// UserMessage returns a user-friendly error message for display in chat.
func (e *FailoverError) UserMessage() string {
	switch e.Reason {
	case ReasonRateLimit:
		return "Usage quota exceeded or rate limit triggered. Please try again later."
	case ReasonAuth:
		return "API authentication failed. Please check your API key."
	case ReasonBilling:
		return "Billing or quota limit exceeded. Please check your account."
	case ReasonTimeout:
		return "Request timed out. Please try again."
	case ReasonServer:
		return "The AI service is temporarily unavailable. Please try again later."
	case ReasonBadParameter:
		return "The provider rejected a request parameter that this model does not support (for example `temperature`). Adjust the request options or switch model."
	case ReasonOutputCap:
		return "The requested output tokens exceed this model's output limit. Lower the configured context window or output reserve."
	case ReasonModelNotFound:
		return "Model not found or not available for this provider. Use /model to switch model, or run `luckclaw models list` to see available models."
	case ReasonContextWindow:
		return "The configured model context window is too small for this request."
	default:
		return e.Error()
	}
}

func (e *FailoverError) Unwrap() error { return e.Wrapped }

// Retryable returns true for transient failures where the same request
// might succeed after a delay (rate-limiting, timeouts, server errors).
func (e *FailoverError) Retryable() bool {
	switch e.Reason {
	case ReasonRateLimit, ReasonTimeout, ReasonServer:
		return true
	default:
		return false
	}
}

// ClassifyHTTPError turns a non-2xx HTTP response into a FailoverError
// with the appropriate reason based on status code and response body.
func ClassifyHTTPError(status int, body string) *FailoverError {
	fe := &FailoverError{Status: status, Body: body}

	switch {
	case status == 429:
		if classifyByBody(body) == ReasonBilling {
			fe.Reason = ReasonBilling
		} else {
			fe.Reason = ReasonRateLimit
		}
	case status == 401 || status == 403:
		fe.Reason = ReasonAuth
	case status == 402:
		fe.Reason = ReasonBilling
	case status == 408:
		fe.Reason = ReasonTimeout
	case status == 400 || status == 404 || status == 413:
		fe.Reason = classifyByBody(body)
	case status >= 500:
		fe.Reason = ReasonServer
	default:
		fe.Reason = ReasonUnknown
	}

	return fe
}

// ClassifyNetworkError wraps a transport-level error (DNS, TCP, TLS,
// timeout) into a FailoverError so retry logic can handle it uniformly.
func ClassifyNetworkError(err error) *FailoverError {
	if err == nil {
		return nil
	}
	fe := &FailoverError{Wrapped: err}

	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		fe.Reason = ReasonTimeout
		return fe
	}

	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "timeout") || strings.Contains(msg, "etimedout"):
		fe.Reason = ReasonTimeout
	case strings.Contains(msg, "connection refused"),
		strings.Contains(msg, "connection reset"),
		strings.Contains(msg, "eof"):
		fe.Reason = ReasonUnknown
	default:
		fe.Reason = ReasonUnknown
	}
	return fe
}

func classifyByBody(body string) FailoverReason {
	lower := strings.ToLower(body)
	switch {
	case strings.Contains(lower, "rate limit") || strings.Contains(lower, "rate_limit"):
		return ReasonRateLimit
	case strings.Contains(lower, "quota") || strings.Contains(lower, "billing") || strings.Contains(lower, "insufficient_quota"):
		return ReasonBilling
	case strings.Contains(lower, "model not exist") || strings.Contains(lower, "model_not_found") ||
		strings.Contains(lower, "model does not exist") || strings.Contains(lower, "model not found") ||
		strings.Contains(lower, "not_found_error"):
		return ReasonModelNotFound
	case strings.Contains(lower, "context length") || strings.Contains(lower, "too long") || strings.Contains(lower, "maximum context"):
		return ReasonContextWindow
	case strings.Contains(lower, "max_tokens") &&
		(strings.Contains(lower, "maximum allowed") || strings.Contains(lower, "at most") ||
			strings.Contains(lower, "too large")):
		// Anthropic: "max_tokens: X > Y, which is the maximum allowed number
		// of output tokens for <model>". OpenAI: "max_tokens is too large:
		// X. This model supports at most Y completion tokens".
		return ReasonOutputCap
	case strings.Contains(lower, "deprecated") || strings.Contains(lower, "unsupported parameter") ||
		strings.Contains(lower, "unsupported_parameter") || strings.Contains(lower, "unsupported value") ||
		strings.Contains(lower, "unknown parameter"):
		return ReasonBadParameter
	default:
		return ReasonFormat
	}
}
