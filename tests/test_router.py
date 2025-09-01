import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

spec = importlib.util.spec_from_file_location(
    "prompt_router", Path(__file__).resolve().parents[1] / "prompt-router.py"
)
pr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr)


@pytest.mark.parametrize("prompt_text", ["hello"])
def test_build_classifier_prompt_lists_all_categories(prompt_text):
    router = pr.CategoryRouter({})
    prompt = pr.build_classifier_prompt(prompt_text, router.categories)
    for name in router.categories:
        assert f"- {name}:" in prompt
    assert prompt.endswith(prompt_text)


@pytest.mark.parametrize(
    "category,model_id",
    [
        ("default", "azure.gpt-4o-mini"),
        ("coding", "eu.anthropic.claude-sonnet-4-20250514-v1:0"),
        ("deep-reasoning", "azure.gpt-4o"),
        ("structured-analysis", "eu.anthropic.claude-sonnet-4-20250514-v1:0"),
        ("content-generation", "azure.gpt-4o"),
        ("vision", "eu.mistral.pixtral-large-2502-v1:0"),
    ],
)
def test_category_router_model_for_returns_model(category, model_id):
    router = pr.CategoryRouter({})
    assert router.model_for(category) == model_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_text,category,expected_model",
    [
        ("What is the capital of France?", "default", "azure.gpt-4o-mini"),
        (
            "Why does `len(5)` raise a TypeError in Python?",
            "coding",
            "eu.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        (
            "How would you plan a mission to Mars using current technology?",
            "deep-reasoning",
            "azure.gpt-4o",
        ),
        (
            "Generate a structured report summarizing last quarter's sales trends with sections for key metrics, regional performance, and recommendations.",
            "structured-analysis",
            "eu.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        (
            "Write a short email requesting vacation time.",
            "content-generation",
            "azure.gpt-4o",
        ),
        (
            "What is happening in this picture?",
            "vision",
            "eu.mistral.pixtral-large-2502-v1:0",
        ),
    ],
)
async def test_pipe_routes_prompt_to_model(
    monkeypatch, prompt_text, category, expected_model
):
    fastapi_stub = types.ModuleType("fastapi")

    class Request: ...

    fastapi_stub.Request = Request
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)

    ow_stub = types.ModuleType("open_webui")
    models_stub = types.ModuleType("open_webui.models")
    users_stub = types.ModuleType("open_webui.models.users")

    class Users:
        @staticmethod
        def get_user_by_id(_):
            return {"id": "u"}

    users_stub.Users = Users
    models_stub.users = users_stub
    ow_stub.models = models_stub

    utils_stub = types.ModuleType("open_webui.utils")
    chat_stub = types.ModuleType("open_webui.utils.chat")

    async def fake_generate(request, body, user):
        if body["model"] == pipe.valves.CLASSIFIER_MODEL_ID:
            return {"choices": [{"message": {"content": category}}]}
        assert body["model"] == expected_model
        return {"ok": True}

    chat_stub.generate_chat_completion = fake_generate
    utils_stub.chat = chat_stub
    ow_stub.utils = utils_stub
    monkeypatch.setitem(sys.modules, "open_webui", ow_stub)
    monkeypatch.setitem(sys.modules, "open_webui.models", models_stub)
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_stub)
    monkeypatch.setitem(sys.modules, "open_webui.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "open_webui.utils.chat", chat_stub)

    pyd_stub = types.ModuleType("pydantic")

    class BaseModel:
        def model_dump(self):
            return self.__dict__

    def Field(default, description=""):
        return default

    pyd_stub.BaseModel = BaseModel
    pyd_stub.Field = Field
    monkeypatch.setitem(sys.modules, "pydantic", pyd_stub)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipe = module.Pipe()
    body = {"messages": [{"role": "user", "content": prompt_text}]}
    events = []

    async def emitter(event):
        events.append(event)

    result = await pipe.pipe(body, {"id": "u"}, Request(), emitter)
    result = await result
    assert body["model"] == expected_model
    assert result == {"ok": True}
    assert any(category in e["data"]["description"] for e in events)


def test_classify_with_bedrock_calls_boto(monkeypatch):
    router = pr.CategoryRouter({})

    class StubbedClient:
        def invoke_model(self, **_):
            data = {"output": {"message": {"content": [{"text": "coding"}]}}}
            return {
                "body": types.SimpleNamespace(read=lambda: json.dumps(data).encode())
            }

    def fake_client(*_args, **_kwargs):
        return StubbedClient()

    monkeypatch.setattr("boto3.client", fake_client)
    label = pr.classify_with_bedrock(router, "prompt")
    assert label == "coding"
