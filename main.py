# main.py
import json
import uuid
from typing import Any

import httpx
import yaml
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# --- Config ---

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
SERVER = CONFIG["server"]
UPSTREAM = CONFIG["upstream"]

# --- Request Conversion (Anthropic -> OpenAI) ---

def convert_content_block(block: dict) -> dict:
    """Convert a single Anthropic content block to OpenAI format.

    - text blocks: pass through as-is
    - image blocks (base64): convert to OpenAI image_url format with data URL
    """
    block_type = block.get("type")

    if block_type == "text":
        return {"type": "text", "text": block["text"]}

    if block_type == "image":
        source = block["source"]
        media_type = source["media_type"]
        data = source["data"]
        data_url = f"data:{media_type};base64,{data}"
        return {
            "type": "image_url",
            "image_url": {"url": data_url},
        }

    # Unknown block type: pass through
    return block


def convert_messages(messages: list) -> list:
    """Convert Anthropic messages array to OpenAI format.

    Handles:
    - Simple string content (pass through)
    - Content block arrays (convert each block via convert_content_block)
    - tool_result blocks in user messages (split into separate tool-role messages)
    - Assistant messages with tool_use blocks (convert to tool_calls format)
    - Assistant messages with thinking blocks (skip thinking, keep text)
    """
    result = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        # Simple string content: pass through directly
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        # Content is a list of blocks
        if isinstance(content, list):
            # --- User message: check for tool_result blocks ---
            if role == "user":
                has_tool_result = any(
                    b.get("type") == "tool_result" for b in content
                )
                if has_tool_result:
                    # Each tool_result becomes a separate tool-role message
                    for block in content:
                        if block.get("type") == "tool_result":
                            tool_content = block.get("content", "")
                            # content can be string or list of blocks
                            if isinstance(tool_content, list):
                                parts = []
                                for sub in tool_content:
                                    if sub.get("type") == "text":
                                        parts.append(sub["text"])
                                tool_content = "\n".join(parts)
                            result.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": tool_content,
                            })
                        else:
                            # Non-tool_result blocks in a mixed message:
                            # convert normally and emit as user message
                            converted = convert_content_block(block)
                            result.append({
                                "role": "user",
                                "content": [converted],
                            })
                else:
                    # Normal user content block array
                    converted_blocks = [
                        convert_content_block(b) for b in content
                    ]
                    result.append({
                        "role": "user",
                        "content": converted_blocks,
                    })
                continue

            # --- Assistant message: handle tool_use and thinking blocks ---
            if role == "assistant":
                text_parts = []
                tool_calls = []

                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block["text"])
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block["input"]),
                            },
                        })
                    elif btype == "thinking":
                        # Skip thinking blocks in outgoing conversion
                        pass

                out_msg: dict[str, Any] = {"role": "assistant"}
                # Set content: joined text or None if only tool calls
                if text_parts:
                    out_msg["content"] = "\n".join(text_parts) if len(text_parts) > 1 else text_parts[0]
                else:
                    out_msg["content"] = None
                if tool_calls:
                    out_msg["tool_calls"] = tool_calls

                result.append(out_msg)
                continue

            # Other roles with list content: convert blocks
            converted_blocks = [convert_content_block(b) for b in content]
            result.append({"role": role, "content": converted_blocks})

    return result


def convert_tools(tools: list) -> list:
    """Convert Anthropic tool definitions to OpenAI format.

    Anthropic: {name, description, input_schema}
    OpenAI:    {type: "function", function: {name, description, parameters}}
    """
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def convert_request(anthropic_req: dict) -> dict:
    """Convert an Anthropic Messages API request to OpenAI Chat Completions format.

    - model: pass through
    - max_tokens: fixed to 16384 (ignore request value)
    - system: move into messages as first system message (string or list)
    - messages: convert via convert_messages()
    - temperature, top_p: pass through
    - stop_sequences -> stop
    - top_k: drop
    - stream: pass through; when True add stream_options
    - tools: convert via convert_tools()
    """
    openai_req: dict[str, Any] = {}

    # Model: pass through
    openai_req["model"] = anthropic_req["model"]

    # max_tokens: pass through
    if "max_tokens" in anthropic_req:
        openai_req["max_tokens"] = anthropic_req["max_tokens"]

    # Build messages list
    converted_messages = []

    # System prompt -> first system message
    system = anthropic_req.get("system")
    if system is not None:
        if isinstance(system, str):
            converted_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # List of content blocks: join text fields
            parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
            converted_messages.append({
                "role": "system",
                "content": "\n".join(parts),
            })

    # Convert and append user/assistant messages
    converted_messages.extend(
        convert_messages(anthropic_req.get("messages", []))
    )
    openai_req["messages"] = converted_messages

    # temperature: pass through
    if "temperature" in anthropic_req:
        openai_req["temperature"] = anthropic_req["temperature"]

    # top_p: pass through
    if "top_p" in anthropic_req:
        openai_req["top_p"] = anthropic_req["top_p"]

    # stop_sequences -> stop
    if "stop_sequences" in anthropic_req:
        openai_req["stop"] = anthropic_req["stop_sequences"]

    # top_k: intentionally dropped (OpenAI doesn't support it)

    # stream: pass through; add stream_options when True
    if "stream" in anthropic_req:
        openai_req["stream"] = anthropic_req["stream"]
        if anthropic_req["stream"]:
            openai_req["stream_options"] = {"include_usage": True}

    # tools: convert if present
    if "tools" in anthropic_req:
        openai_req["tools"] = convert_tools(anthropic_req["tools"])

    return openai_req


# --- Response Conversion (OpenAI -> Anthropic) ---

FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def generate_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def convert_response(openai_resp: dict, model: str) -> dict:
    """Convert OpenAI Chat Completions response to Anthropic Messages format."""
    choice = openai_resp["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason", "stop")
    usage = openai_resp.get("usage", {})

    content = []

    # Add reasoning/thinking block if present
    reasoning = message.get("reasoning_content")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

    # Add text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # Add tool_use blocks
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc["function"]
        try:
            input_data = json.loads(func["arguments"])
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        content.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": func["name"],
            "input": input_data,
        })

    if not content:
        content.append({"type": "text", "text": ""})

    return {
        "id": generate_msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": FINISH_REASON_MAP.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# --- Streaming Event Builders ---

def sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

def build_message_start_event(model: str, msg_id: str | None = None) -> str:
    return sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id or generate_msg_id(),
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

def build_content_block_start_event(index: int, block_type: str, tool_id: str | None = None, tool_name: str | None = None) -> str:
    if block_type == "text":
        block = {"type": "text", "text": ""}
    elif block_type == "thinking":
        block = {"type": "thinking", "thinking": ""}
    elif block_type == "tool_use":
        block = {"type": "tool_use", "id": tool_id or "", "name": tool_name or "", "input": {}}
    else:
        block = {"type": block_type}
    return sse_event("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": block,
    })

def build_content_block_delta_event(index: int, delta_type: str, text: str = "", partial_json: str = "") -> str:
    if delta_type == "text_delta":
        delta = {"type": "text_delta", "text": text}
    elif delta_type == "thinking_delta":
        delta = {"type": "thinking_delta", "thinking": text}
    elif delta_type == "input_json_delta":
        delta = {"type": "input_json_delta", "partial_json": partial_json}
    else:
        delta = {"type": delta_type}
    return sse_event("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": delta,
    })

def build_content_block_stop_event(index: int) -> str:
    return sse_event("content_block_stop", {"type": "content_block_stop", "index": index})

def build_message_delta_event(stop_reason: str, output_tokens: int = 0) -> str:
    return sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

def build_message_stop_event() -> str:
    return sse_event("message_stop", {"type": "message_stop"})


# --- Proxy Helpers ---

def get_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=UPSTREAM["base_url"],
        timeout=httpx.Timeout(UPSTREAM.get("timeout", 300)),
    )


def get_upstream_headers() -> dict:
    return {
        "Authorization": f"Bearer {UPSTREAM['api_key']}",
        "Content-Type": "application/json",
    }


def convert_error(status_code: int, openai_error: dict) -> tuple[int, dict]:
    error_body = openai_error.get("error", {})
    error_type = error_body.get("type", "api_error")
    type_map = {
        "invalid_request_error": "invalid_request_error",
        "authentication_error": "authentication_error",
        "rate_limit_error": "rate_limit_error",
        "not_found_error": "not_found_error",
    }
    return status_code, {
        "type": "error",
        "error": {
            "type": type_map.get(error_type, "api_error"),
            "message": error_body.get("message", "Unknown error"),
        },
    }


# --- FastAPI App & Endpoints ---

app = FastAPI(title="cc-proxy")


@app.get("/")
async def health():
    return {"status": "ok", "service": "cc-proxy"}


async def handle_streaming(client: httpx.AsyncClient, openai_req: dict, model: str):
    """Handle streaming proxy: read OpenAI SSE -> convert -> emit Anthropic SSE."""

    async def stream_generator():
        msg_id = generate_msg_id()
        yield build_message_start_event(model=model, msg_id=msg_id)

        block_index = 0
        current_block_type = None  # "text" | "thinking" | "tool_use"
        tool_call_states: dict[int, dict] = {}  # OpenAI tool call index -> state
        finish_reason = "end_turn"
        output_tokens = 0

        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=openai_req,
            headers=get_upstream_headers(),
        ) as resp:
            if resp.status_code != 200:
                error_text = ""
                async for chunk in resp.aiter_text():
                    error_text += chunk
                try:
                    error_body = json.loads(error_text)
                except Exception:
                    error_body = {"error": {"message": error_text, "type": "api_error"}}
                _, err = convert_error(resp.status_code, error_body)
                yield sse_event("error", err)
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    usage = chunk.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", 0)
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                chunk_finish = choice.get("finish_reason")

                if chunk_finish:
                    finish_reason = FINISH_REASON_MAP.get(chunk_finish, "end_turn")

                usage = chunk.get("usage")
                if usage:
                    output_tokens = usage.get("completion_tokens", 0)

                # Handle reasoning_content (thinking)
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    if current_block_type != "thinking":
                        if current_block_type is not None:
                            yield build_content_block_stop_event(block_index)
                            block_index += 1
                        yield build_content_block_start_event(block_index, "thinking")
                        current_block_type = "thinking"
                    yield build_content_block_delta_event(block_index, "thinking_delta", text=reasoning)
                    continue

                # Handle text content
                text_content = delta.get("content")
                if text_content:
                    if current_block_type != "text":
                        if current_block_type is not None:
                            yield build_content_block_stop_event(block_index)
                            block_index += 1
                        yield build_content_block_start_event(block_index, "text")
                        current_block_type = "text"
                    yield build_content_block_delta_event(block_index, "text_delta", text=text_content)
                    continue

                # Handle tool calls
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        tc_index = tc.get("index", 0)
                        if tc_index not in tool_call_states:
                            if current_block_type is not None:
                                yield build_content_block_stop_event(block_index)
                                block_index += 1
                            tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                            tc_name = tc.get("function", {}).get("name", "")
                            tool_call_states[tc_index] = {
                                "id": tc_id,
                                "name": tc_name,
                                "block_index": block_index,
                            }
                            yield build_content_block_start_event(
                                block_index, "tool_use", tool_id=tc_id, tool_name=tc_name
                            )
                            current_block_type = "tool_use"

                        args_delta = tc.get("function", {}).get("arguments", "")
                        if args_delta:
                            bi = tool_call_states[tc_index]["block_index"]
                            yield build_content_block_delta_event(
                                bi, "input_json_delta", partial_json=args_delta
                            )

        # Close the last content block
        if current_block_type is not None:
            yield build_content_block_stop_event(block_index)

        yield build_message_delta_event(finish_reason, output_tokens)
        yield build_message_stop_event()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def handle_non_streaming(client: httpx.AsyncClient, openai_req: dict, model: str):
    resp = await client.post(
        "/v1/chat/completions", json=openai_req, headers=get_upstream_headers()
    )
    if resp.status_code != 200:
        try:
            error_body = resp.json()
        except Exception:
            error_body = {"error": {"message": resp.text, "type": "api_error"}}
        status, body = convert_error(resp.status_code, error_body)
        return JSONResponse(status_code=status, content=body)
    openai_resp = resp.json()
    anthropic_resp = convert_response(openai_resp, model=model)
    return JSONResponse(content=anthropic_resp)


@app.post("/v1/messages")
async def messages_endpoint(request: Request):
    body = await request.json()
    model = body.get("model", "unknown")
    is_stream = body.get("stream", False)
    openai_req = convert_request(body)

    try:
        async with get_http_client() as client:
            if is_stream:
                return await handle_streaming(client, openai_req, model)
            else:
                return await handle_non_streaming(client, openai_req, model)
    except httpx.ConnectError:
        return JSONResponse(status_code=529, content={
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Failed to connect to upstream"},
        })
    except httpx.TimeoutException:
        return JSONResponse(status_code=529, content={
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Upstream request timed out"},
        })


def run():
    uvicorn.run(app, host=SERVER["host"], port=SERVER["port"])

if __name__ == "__main__":
    run()
