from fastapi import Request as FastAPIRequest
from fastapi_offline import FastAPIOffline
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any, Literal
import time, json, os, uuid, re
from datetime import datetime
from common.api import DeepSeekAPI
from common.config import DEEPSEEK_API_KEY

app = FastAPIOffline()

api = DeepSeekAPI(DEEPSEEK_API_KEY)

sessions: Dict[str, dict] = {}

AVAILABLE_MODELS = [
    {"id": "thinking_not_search", "object": "model1", "created": 1677610602, "owned_by": "you"},
    {"id": "thinking_search", "object": "model2", "created": 1677610602, "owned_by": "you"},
    {"id": "not_thinking_not_search", "object": "model3", "created": 1677610602, "owned_by": "you"},
    {"id": "not_thinking_search", "object": "model4", "created": 1677610602, "owned_by": "you"},
    {"id": "ds/thinking_not_search", "object": "model5", "created": 1677610602, "owned_by": "you"},
    {"id": "ds/thinking_search", "object": "model6", "created": 1677610602, "owned_by": "you"},
    {"id": "ds/not_thinking_not_search", "object": "model7", "created": 1677610602, "owned_by": "you"},
    {"id": "ds/not_thinking_search", "object": "model8", "created": 1677610602, "owned_by": "you"},
]

# ---------- Models ----------
class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None

class Message(BaseModel):
    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatRequest(BaseModel):
    model_config = {"extra": "ignore"}
    
    messages: List[Message]
    model: str = "thinking_not_search"
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

# ---------- Helper ----------
def extract_content(content: Optional[Union[str, List[ContentPart]]]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text or "" for part in content if part.type == "text")

def build_tools_system_instruction(tools: List[Tool]) -> str:
    tools_list = []
    for tool in tools:
        tools_list.append({
            "name": tool.function.name,
            "description": tool.function.description or "",
            "parameters": tool.function.parameters or {}
        })
    
    tools_json = json.dumps(tools_list, indent=2, ensure_ascii=False)
    
    instruction = (
        "You have access to the following tools/functions:\n"
        f"{tools_json}\n\n"
        "To invoke a tool, you MUST reply ONLY with a valid JSON block in this format:\n"
        "```json\n"
        "{\n"
        '  "tool_calls": [\n'
        "    {\n"
        '      "name": "<function_name>",\n'
        '      "arguments": { <argument_key_value_pairs> }\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Do NOT add any extra text or conversational filler before or after the JSON when invoking a tool.\n"
        "If no tool invocation is needed, respond normally with plain text."
    )
    return instruction

def messages_to_api_format(messages: List[Message], tools: Optional[List[Tool]] = None, tool_choice: Optional[Union[str, Dict[str, Any]]] = None) -> str:
    parts = []
    if tool_choice == "none":
        tools = None

    system_instruction = build_tools_system_instruction(tools) if tools else ""
    
    has_system = False
    for msg in messages:
        if msg.role == "system":
            has_system = True
            content = extract_content(msg.content)
            if system_instruction:
                content = f"{content}\n\n[SYSTEM TOOL INSTRUCTIONS]\n{system_instruction}"
            parts.append(f"[SYSTEM]\n{content}")
        elif msg.role in ("tool", "function"):
            content = extract_content(msg.content)
            tool_id_info = f" (call_id: {msg.tool_call_id})" if msg.tool_call_id else ""
            name_info = f" (name: {msg.name})" if msg.name else ""
            parts.append(f"[TOOL RESULT{tool_id_info}{name_info}]\n{content}")
        elif msg.role == "assistant":
            content = extract_content(msg.content)
            if msg.tool_calls:
                tc_list = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = tc.function.arguments
                    tc_list.append({
                        "name": tc.function.name,
                        "arguments": args
                    })
                tc_json = json.dumps({"tool_calls": tc_list}, ensure_ascii=False)
                if content:
                    content = f"{content}\n\n```json\n{tc_json}\n```"
                else:
                    content = f"```json\n{tc_json}\n```"
            parts.append(f"[ASSISTANT]\n{content}")
        else:
            content = extract_content(msg.content)
            parts.append(f"[{msg.role.upper()}]\n{content}")

    if tools and not has_system:
        parts.insert(0, f"[SYSTEM]\n{system_instruction}")

    return "\n\n".join(parts)

def parse_tool_calls(text: str):
    """
    Parses text to extract tool calls if present.
    Returns tuple: (cleaned_content, tool_calls_list)
    If no tool calls found, returns (text, None)
    """
    if not text:
        return text, None

    json_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.IGNORECASE)
    
    raw_json_str = None
    match_span = None
    if json_block_match:
        raw_json_str = json_block_match.group(1).strip()
        match_span = json_block_match.span()
    else:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            raw_json_str = text[start_idx:end_idx+1].strip()
            match_span = (start_idx, end_idx + 1)
            
    if not raw_json_str:
        return text, None

    try:
        data = json.loads(raw_json_str)
    except json.JSONDecodeError:
        return text, None

    if not isinstance(data, dict):
        return text, None

    raw_calls = []
    if "tool_calls" in data and isinstance(data["tool_calls"], list):
        raw_calls = data["tool_calls"]
    elif "name" in data and ("arguments" in data or "parameters" in data):
        raw_calls = [data]
    elif "function" in data:
        raw_calls = [data]
    else:
        return text, None

    parsed_tool_calls = []
    for item in raw_calls:
        func_name = None
        func_args = {}
        
        if "name" in item:
            func_name = item["name"]
        elif "function" in item:
            if isinstance(item["function"], str):
                func_name = item["function"]
            elif isinstance(item["function"], dict):
                func_name = item["function"].get("name")
                
        if "arguments" in item:
            func_args = item["arguments"]
        elif "parameters" in item:
            func_args = item["parameters"]

        if not func_name:
            continue

        if isinstance(func_args, dict):
            args_str = json.dumps(func_args, ensure_ascii=False)
        elif isinstance(func_args, str):
            args_str = func_args
        else:
            args_str = json.dumps(func_args, ensure_ascii=False)

        call_id = f"call_{uuid.uuid4().hex[:8]}"
        parsed_tool_calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": args_str
            }
        })

    if not parsed_tool_calls:
        return text, None

    content_before = text[:match_span[0]].strip() if match_span else ""
    content_after = text[match_span[1]:].strip() if match_span else ""
    cleaned_content = f"{content_before}\n{content_after}".strip()
    if not cleaned_content:
        cleaned_content = None

    return cleaned_content, parsed_tool_calls

# ---------- Middleware برای لاگ ----------
@app.middleware("http")
async def log_time(request: FastAPIRequest, call_next):
    start = datetime.now()
    print(f"[{start.strftime('%H:%M:%S.%f')[:-3]}] --> {request.method} {request.url.path}")
    response = await call_next(request)
    end = datetime.now()
    print(f"[{end.strftime('%H:%M:%S.%f')[:-3]}] <-- {response.status_code} (took {(end-start).total_seconds():.2f}s)")
    return response

# ---------- Endpoints ----------
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    chat_id = api.create_chat_session()
    
    prompt = messages_to_api_format(request.messages, tools=request.tools, tool_choice=request.tool_choice)

    model_name = request.model.split("/")[-1]
    if model_name == "not_thinking_not_search":
        thinking = False
        search = False
    elif model_name == "thinking_not_search":
        thinking = True
        search = False
    elif model_name == "thinking_search":
        thinking = True
        search = True
    elif model_name == "not_thinking_search":
        thinking = False
        search = True
    else:
        thinking, search = True, False  # default fallback

    if request.stream:
        def generate():
            if not request.tools:
                for chunk in api.chat_completion(
                    chat_id, 
                    prompt,  # کل messages به صورت prompt
                    thinking_enabled=thinking,
                    search_enabled=search
                ):
                    chunk_type = chunk.get("type")
                    
                    if chunk_type == 'thinking':
                        # ارسال thinking به عنوان reasoning_content (طبق استاندارد OpenAI)
                        delta = {"reasoning_content": chunk.get("delta", "")}
                    elif chunk_type == 'content':
                        delta = {"content": chunk.get("delta", "")}
                    elif chunk_type == 'finished':
                        break  # خروج از حلقه برای ارسال final chunk
                    else:
                        continue  # نوع ناشناخته، نادیده بگیر
                    
                    response_chunk = {
                        "id": f"chatcmpl-{chat_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(response_chunk)}\n\n"

                final_chunk = {
                    "id": f"chatcmpl-{chat_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                # Accumulate stream when tools are passed to parse tool calls cleanly
                full_text = ""
                for chunk in api.chat_completion(
                    chat_id, 
                    prompt,
                    thinking_enabled=thinking,
                    search_enabled=search
                ):
                    chunk_type = chunk.get("type")
                    if chunk_type == 'thinking':
                        delta = {"reasoning_content": chunk.get("delta", "")}
                        response_chunk = {
                            "id": f"chatcmpl-{chat_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"
                    elif chunk_type == 'content':
                        full_text += chunk.get("delta", "")
                    elif chunk_type == 'finished':
                        break

                content_output, tool_calls = parse_tool_calls(full_text)
                if tool_calls:
                    tool_call_deltas = []
                    for idx, tc in enumerate(tool_calls):
                        tool_call_deltas.append({
                            "index": idx,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        })
                    delta_obj = {"tool_calls": tool_call_deltas}
                    if content_output:
                        delta_obj["content"] = content_output
                    
                    response_chunk = {
                        "id": f"chatcmpl-{chat_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": delta_obj, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(response_chunk)}\n\n"
                    
                    final_chunk = {
                        "id": f"chatcmpl-{chat_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                else:
                    if full_text:
                        response_chunk = {
                            "id": f"chatcmpl-{chat_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [{"index": 0, "delta": {"content": full_text}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(response_chunk)}\n\n"
                    
                    final_chunk = {
                        "id": f"chatcmpl-{chat_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # حالت غیر-استریم
    full_text = ""
    full_thinking = ""
    
    for chunk in api.chat_completion(
        chat_id, 
        prompt,  # کل messages
        thinking_enabled=thinking,
        search_enabled=search
    ):
        chunk_type = chunk.get("type")
            
        if chunk_type == 'content':
            full_text += chunk.get("delta", "")
        elif chunk_type == 'thinking':
            full_thinking += chunk.get("delta", "")
        elif chunk_type == 'finished':
            break

    content_output = full_text
    tool_calls = None
    finish_reason = "stop"

    if request.tools:
        content_output, tool_calls = parse_tool_calls(full_text)
        if tool_calls:
            finish_reason = "tool_calls"

    message_data = {
        "role": "assistant",
        "content": content_output,
        "reasoning_content": full_thinking if full_thinking else None
    }
    if tool_calls:
        message_data["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{chat_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message_data,
                "finish_reason": finish_reason
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": AVAILABLE_MODELS}
