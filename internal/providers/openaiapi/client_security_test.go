package openaiapi

import (
	"context"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
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
	if err := json.Unmarshal([]byte(`{"model":"test-model","messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"turn","strict":true,"schema":{"type":"object","properties":{"reply":{"type":"string"}},"required":["reply"],"additionalProperties":false}}}}`), &structured); err != nil {
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
	jsonSchema, _ := format["json_schema"].(map[string]any)
	schema, _ := jsonSchema["schema"].(map[string]any)
	if !ok || format["type"] != "json_schema" || jsonSchema["name"] != "turn" ||
		jsonSchema["strict"] != true || schema["additionalProperties"] != false {
		t.Fatalf("structured response format missing: %s", encoded)
	}
}

func TestCurrentOpenAIModelsUseCompatibleChatFields(t *testing.T) {
	encoded, err := (&Client{Provider: "openai"}).buildRequestBody(ChatRequest{
		Model:           "gpt-5.6-terra",
		Messages:        []Message{{Role: "user", Content: "analyze"}},
		Temperature:     0.1,
		MaxTokens:       8192,
		ReasoningEffort: "medium",
	})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatal(err)
	}
	if _, present := body["max_tokens"]; present {
		t.Fatalf("current OpenAI request contains legacy max_tokens: %s", encoded)
	}
	if _, present := body["temperature"]; present {
		t.Fatalf("current OpenAI request contains unsupported temperature: %s", encoded)
	}
	if body["max_completion_tokens"] != float64(8192) {
		t.Fatalf("max_completion_tokens = %v, want 8192", body["max_completion_tokens"])
	}
	if body["reasoning_effort"] != "medium" {
		t.Fatalf("reasoning_effort = %v, want medium", body["reasoning_effort"])
	}
}

func TestCompatibleProviderModelsKeepLegacyChatFields(t *testing.T) {
	encoded, err := (&Client{Provider: "ollama"}).buildRequestBody(ChatRequest{
		Model:       "qwen2.5",
		Messages:    []Message{{Role: "user", Content: "analyze"}},
		Temperature: 0.1,
		MaxTokens:   8192,
	})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatal(err)
	}
	if body["max_tokens"] != float64(8192) || body["temperature"] != 0.1 {
		t.Fatalf("compatible-provider controls changed: %s", encoded)
	}
	if _, present := body["max_completion_tokens"]; present {
		t.Fatalf("compatible-provider request contains OpenAI-only field: %s", encoded)
	}
}

func TestProviderRootPoolLoadsCertifiFallback(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	t.Cleanup(server.Close)
	certificate := server.TLS.Certificates[0].Certificate[0]
	bundle := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificate})
	path := filepath.Join(t.TempDir(), "cacert.pem")
	if err := os.WriteFile(path, bundle, 0o600); err != nil {
		t.Fatal(err)
	}

	pool := loadProviderRootCAs(
		func() (*x509.CertPool, error) { return x509.NewCertPool(), nil },
		os.ReadFile,
		[]string{filepath.Join(t.TempDir(), "missing.pem"), path},
	)
	if pool == nil {
		t.Fatal("fallback root pool is nil")
	}
	if len(pool.Subjects()) != 1 {
		t.Fatalf("fallback subjects = %d, want 1", len(pool.Subjects()))
	}

	client := newHTTPClientWithProxy(nil, time.Second, pool)
	response, err := client.Get(server.URL)
	if err != nil {
		t.Fatalf("fallback TLS request: %v", err)
	}
	response.Body.Close()
}

func TestProviderCACandidatesHonorExplicitBundle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "explicit-cacert.pem")
	t.Setenv("SSL_CERT_FILE", "  "+path+"  ")
	candidates := providerCACandidates()
	if len(candidates) == 0 {
		t.Fatal("explicit CA candidate was omitted")
	}
	if candidates[0] != path {
		t.Fatalf("first CA candidate = %q, want %q", candidates[0], path)
	}
}

func TestProviderCACandidatesDiscoverLuckfoxCertifiBundle(t *testing.T) {
	const luckfoxBundle = "/usr/lib/python3.11/site-packages/certifi/cacert.pem"
	expectedPatterns := []string{
		"/usr/lib/python*/site-packages/certifi/cacert.pem",
		"/usr/local/lib/python*/site-packages/certifi/cacert.pem",
	}
	var patterns []string
	candidates := providerCACandidatesWithGlob("", func(pattern string) ([]string, error) {
		patterns = append(patterns, pattern)
		if pattern == "/usr/lib/python*/site-packages/certifi/cacert.pem" {
			return []string{luckfoxBundle}, nil
		}
		return nil, nil
	})
	if !slices.Equal(patterns, expectedPatterns) {
		t.Fatalf("glob patterns = %v, want %v", patterns, expectedPatterns)
	}
	if len(candidates) != 1 || candidates[0] != luckfoxBundle {
		t.Fatalf("CA candidates = %v, want [%s]", candidates, luckfoxBundle)
	}
}

func TestProviderRootPoolPreservesHealthySystemTrust(t *testing.T) {
	systemPool := x509.NewCertPool()
	server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	t.Cleanup(server.Close)
	certificate := server.TLS.Certificates[0].Certificate[0]
	systemPool.AddCert(server.Certificate())
	candidate := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certificate})
	read := false

	pool := loadProviderRootCAs(
		func() (*x509.CertPool, error) { return systemPool, nil },
		func(string) ([]byte, error) {
			read = true
			return candidate, nil
		},
		[]string{"unused-certifi.pem"},
	)
	if pool != systemPool || len(pool.Subjects()) != 1 {
		t.Fatal("healthy system trust was replaced")
	}
	if read {
		t.Fatal("certifi fallback was read for a healthy system pool")
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
