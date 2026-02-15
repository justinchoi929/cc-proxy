# main.py
import json
import time
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

    # max_tokens: fixed
    openai_req["max_tokens"] = 16384

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


app = FastAPI(title="cc-proxy")

if __name__ == "__main__":
    uvicorn.run(app, host=SERVER["host"], port=SERVER["port"])
