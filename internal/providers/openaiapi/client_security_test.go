package openaiapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

const expectedMaxChatResponseBodyBytes = 512 * 1024

type responseRoundTripper func(*http.Request) (*http.Response, error)

func (roundTrip responseRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	return roundTrip(request)
}

type countingResponseBody struct {
	remaining int
	read      int
}

func (body *countingResponseBody) Read(buffer []byte) (int, error) {
	if body.remaining == 0 {
		return 0, io.EOF
	}
	count := len(buffer)
	if count > body.remaining {
		count = body.remaining
	}
	for index := 0; index < count; index++ {
		buffer[index] = 'x'
	}
	body.remaining -= count
	body.read += count
	return count, nil
}

func (*countingResponseBody) Close() error { return nil }

func TestChatRequestResponseFormatIsOptional(t *testing.T) {
	ordinary, err := (&Client{}).buildRequestBody(ChatRequest{
		Model:    "test-model",
		Messages: []Message{{Role: "user", Content: "ordinary"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var ordinaryBody map[string]any
	if err := json.Unmarshal(ordinary, &ordinaryBody); err != nil {
		t.Fatal(err)
	}
	if _, present := ordinaryBody["response_format"]; present {
		t.Fatalf("ordinary request unexpectedly contains response_format: %s", ordinary)
	}

	var structured ChatRequest
	if err := json.Unmarshal([]byte(`{"model":"test-model","messages":[],"response_format":{"type":"json_object"}}`), &structured); err != nil {
		t.Fatal(err)
	}
	encoded, err := (&Client{}).buildRequestBody(structured)
	if err != nil {
		t.Fatal(err)
	}
	var structuredBody map[string]any
	if err := json.Unmarshal(encoded, &structuredBody); err != nil {
		t.Fatal(err)
	}
	format, ok := structuredBody["response_format"].(map[string]any)
	if !ok || format["type"] != "json_object" {
		t.Fatalf("structured response format missing: %s", encoded)
	}
}

func TestChatRejectsOversizedSuccessAndErrorBodiesAtReadLimit(t *testing.T) {
	for _, status := range []int{http.StatusOK, http.StatusBadRequest} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			body := &countingResponseBody{remaining: 4 * expectedMaxChatResponseBodyBytes}
			client := &Client{
				APIBase: "https://provider.invalid/v1",
				HTTPClient: &http.Client{Transport: responseRoundTripper(func(*http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: status,
						Header:     make(http.Header),
						Body:       body,
					}, nil
				})},
			}
			_, err := client.Chat(context.Background(), ChatRequest{
				Model:    "test-model",
				Messages: []Message{{Role: "user", Content: "bounded"}},
			})
			if err == nil || !strings.Contains(err.Error(), "response body exceeds") {
				t.Fatalf("oversized status %d did not return the bounded-body error", status)
			}
			if body.read > expectedMaxChatResponseBodyBytes+1 {
				t.Fatalf("read %d bytes, want at most %d", body.read, expectedMaxChatResponseBodyBytes+1)
			}
			var providerError *FailoverError
			if !errors.As(err, &providerError) || providerError.Reason != ReasonFormat || providerError.Body != "" {
				if providerError == nil {
					t.Fatal("oversized response did not return a classified provider error")
				}
				t.Fatalf("provider error reason=%q body bytes=%d", providerError.Reason, len(providerError.Body))
			}
		})
	}
}
