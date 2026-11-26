"""A local HTTP server that imitates the gateway in front of OpenAI, Anthropic and Google for one
structured-output call, and records exactly what each SDK sends (method, path, query, headers, body).

Responses are SSE, like the vendors stream them. `buffered=True` sends the whole SSE body at once with a
Content-Length (what the gateway does: it buffers the vendor response and returns it whole);
`buffered=False` writes it event by event with chunked transfer encoding (what the vendors do).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self
from urllib.parse import urlsplit


@dataclass
class Seen:
    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class FakeVendor:
    answer: dict[str, Any]
    buffered: bool = True
    refuse_with: int | None = None  # answer every request with a gateway error of this status
    refuse_code: str = "cap_reached"  # ... and this gateway error code
    reject_temperature: bool = False  # 400 in the vendor's error shape when the body carries temperature
    fail_first: int = 0  # answer the first N requests with a vendor 503
    seen: list[Seen] = field(default_factory=list)
    server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.server is not None
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> Self:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # keep test output clean
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                parts = urlsplit(self.path)
                body = json.loads(raw) if raw else {}
                fake.seen.append(
                    Seen("POST", parts.path, parts.query, {k.lower(): v for k, v in self.headers.items()}, body)
                )
                vendor_error = None
                if fake.reject_temperature and _has_temperature(parts.path, body):
                    vendor_error = _temperature_error(parts.path)
                elif fake.fail_first and len(fake.seen) <= fake.fail_first:
                    vendor_error = (503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
                if vendor_error:
                    status, err = vendor_error
                    payload = json.dumps(err).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if fake.refuse_with:
                    payload = json.dumps(
                        {
                            "error": {
                                "type": "tekti_gateway",
                                "code": fake.refuse_code,
                                "message": "refused by the gateway",
                            }
                        }
                    ).encode()
                    self.send_response(fake.refuse_with)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if parts.path.startswith("/openai/"):
                    events = openai_events(body, fake.answer)
                elif parts.path.startswith("/anthropic/"):
                    events = anthropic_events(body, fake.answer)
                elif parts.path.startswith("/google/"):
                    events = google_events(body, fake.answer)
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if fake.buffered:
                    data = b"".join(events)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for chunk in events:
                        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self.server is not None
        self.server.shutdown()
        self.server.server_close()


def _has_temperature(path: str, body: dict[str, Any]) -> bool:
    if path.startswith("/google/"):
        return "temperature" in (body.get("generationConfig") or {})
    return "temperature" in body


def _temperature_error(path: str) -> tuple[int, dict[str, Any]]:
    """Refusals in each vendor's error shape. OpenAI's and Anthropic's wording is the published one;
    Google ignores temperature on current models, so its message here is illustrative."""
    if path.startswith("/openai/"):
        return 400, {
            "error": {
                "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "unsupported_parameter",
            }
        }
    if path.startswith("/anthropic/"):
        return 400, {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "temperature: Extra inputs are not permitted"},
        }
    return 400, {
        "error": {"code": 400, "message": "temperature is not supported by this model", "status": "INVALID_ARGUMENT"}
    }


def _sse(event: str | None, data: dict[str, Any]) -> bytes:
    head = f"event: {event}\n" if event else ""
    return (head + "data: " + json.dumps(data) + "\n\n").encode()


def openai_events(body: dict[str, Any], answer: dict[str, Any]) -> list[bytes]:
    """Responses API stream: one function call to the structured-output tool, then response.completed."""
    tool = body["tools"][0]["name"]
    args = json.dumps(answer)
    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": tool,
        "arguments": "",
        "status": "in_progress",
    }
    usage = {
        "input_tokens": 120,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 30,
        "output_tokens_details": {"reasoning_tokens": 10},
        "total_tokens": 150,
    }
    resp = {"id": "resp_1", "object": "response", "status": "in_progress", "model": body["model"], "output": []}
    done = {
        **resp,
        "status": "completed",
        "output": [{**item, "arguments": args, "status": "completed"}],
        "usage": usage,
    }
    return [
        _sse("response.created", {"type": "response.created", "sequence_number": 0, "response": resp}),
        _sse(
            "response.output_item.added",
            {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": item},
        ),
        _sse(
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2,
                "item_id": "fc_1",
                "output_index": 0,
                "delta": args,
            },
        ),
        _sse(
            "response.function_call_arguments.done",
            {
                "type": "response.function_call_arguments.done",
                "sequence_number": 3,
                "item_id": "fc_1",
                "output_index": 0,
                "arguments": args,
                "name": tool,
            },
        ),
        _sse("response.completed", {"type": "response.completed", "sequence_number": 4, "response": done}),
    ]


def anthropic_events(body: dict[str, Any], answer: dict[str, Any]) -> list[bytes]:
    """Messages stream: JSON text when native output was requested, else a tool_use block."""
    text = json.dumps(answer)
    native = "format" in (body.get("output_config") or {})
    start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 200,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }
    if native:
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": text}
        stop = "end_turn"
    else:
        name = (body.get("tool_choice") or {}).get("name") or body["tools"][0]["name"]
        block = {"type": "tool_use", "id": "toolu_1", "name": name, "input": {}}
        delta = {"type": "input_json_delta", "partial_json": text}
        stop = "tool_use"
    return [
        _sse("message_start", start),
        _sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 40},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]


def google_events(body: dict[str, Any], answer: dict[str, Any]) -> list[bytes]:
    """streamGenerateContent?alt=sse: one chunk with a functionCall to the structured-output tool."""
    name = body["tools"][0]["functionDeclarations"][0]["name"]
    chunk = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"functionCall": {"name": name, "args": answer}}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 300,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 15,
            "totalTokenCount": 335,
        },
        "modelVersion": "fake",
    }
    return [_sse(None, chunk)]
