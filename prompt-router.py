"""
title: Prompt Router Pipe
author: sunborn23
author_url: https://github.com/sunborn23
repo_url: https://github.com/sunborn23/prompt-router
version: 0.4
"""

import json
import logging
import os
import sys
from typing import Any, AsyncIterator, Dict

try:  # OpenWebUI runtime (Pipe mode)
    from fastapi import Request
    from open_webui.models.users import Users
    from open_webui.utils.chat import generate_chat_completion
    from pydantic import BaseModel, Field
    from starlette.responses import StreamingResponse

    OPENWEBUI = True
except ImportError:  # Local CLI runtime
    import boto3

    OPENWEBUI = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core router
# ---------------------------------------------------------------------------


class CategoryRouter:
    """Registry of categories and associated model IDs."""

    def __init__(self, valves: Dict[str, str]):
        v = valves or {}
        self.categories = {
            "default": {
                "model": v.get("MODEL_DEFAULT", "azure.gpt-4o-mini"),
                "description": (
                    "For everyday, relatively simple requests with short to medium "
                    "length (1–3 paragraphs). Covers small talk, short explanations, "
                    "summaries, or general questions that do not require deep reasoning."
                ),
            },
            "coding": {
                "model": v.get(
                    "MODEL_CODING", "eu.anthropic.claude-sonnet-4-20250514-v1:0"
                ),
                "description": (
                    "For technical requests related to programming, debugging, "
                    "architecture, or IT tools. Includes code snippets, error messages, "
                    "and in-depth technical explanations."
                ),
            },
            "deep-reasoning": {
                "model": v.get("MODEL_DEEP", "azure.gpt-4o"),
                "description": (
                    "For complex, multi-layered tasks requiring high accuracy, creative "
                    "solutions, or multimodality (e.g. images)."
                ),
            },
            "structured-analysis": {
                "model": v.get(
                    "MODEL_STRUCT", "eu.anthropic.claude-sonnet-4-20250514-v1:0"
                ),
                "description": (
                    "For tasks that need detailed, well-structured, text-heavy "
                    "explanations. Useful for longer documents or strategic analyses."
                ),
            },
            "content-generation": {
                "model": v.get("MODEL_CONTENT", "azure.gpt-4o"),
                "description": (
                    "For creating polished or longer-form text such as emails, blog "
                    "posts, or presentations."
                ),
            },
            "vision": {
                "model": v.get("MODEL_VISION", "eu.mistral.pixtral-large-2502-v1:0"),
                "description": (
                    "For requests involving images or visual content such as image "
                    "description or multimodal tasks."
                ),
            },
        }

    def classifier_prompt(self, user_prompt: str) -> str:
        return build_classifier_prompt(user_prompt, self.categories)

    def model_for(self, category: str) -> str:
        return self.categories[category]["model"]


# ---------------------------------------------------------------------------
# OpenWebUI Pipe mode
# ---------------------------------------------------------------------------


if OPENWEBUI:

    class Pipe:
        class Valves(BaseModel):
            CLASSIFIER_MODEL_ID: str = Field(
                "eu.amazon.nova-micro-v1:0",
                description="Model ID used for classification for routing decisions.",
            )
            MODEL_DEFAULT: str = Field(
                "azure.gpt-4o-mini",
                description="Model ID for default / smalltalk prompts",
            )
            MODEL_CODING: str = Field(
                "eu.anthropic.claude-sonnet-4-20250514-v1:0",
                description="Model ID for coding / tech prompts",
            )
            MODEL_DEEP: str = Field(
                "azure.gpt-4o",
                description="Model ID for deep reasoning / complex queries prompts",
            )
            MODEL_STRUCT: str = Field(
                "eu.anthropic.claude-sonnet-4-20250514-v1:0",
                description="Model ID for structured analysis prompts",
            )
            MODEL_CONTENT: str = Field(
                "azure.gpt-4o",
                description="Model ID for content generation / long-form writing prompts",
            )
            MODEL_VISION: str = Field(
                "eu.mistral.pixtral-large-2502-v1:0",
                description="Model ID for vision / multimodal prompts",
            )
            PREFACE_ENABLED: bool = Field(
                True, description="If false, routing preface will be omitted."
            )

        def __init__(self) -> None:
            self.valves = self.Valves()
            self.router = CategoryRouter(self.valves.model_dump())

        async def pipe(
            self, body: Dict[str, Any], __user__: Dict[str, Any], __request__: Request
        ):
            """Route the prompt to a model chosen by a classifier."""

            user = Users.get_user_by_id((__user__ or {}).get("id"))

            prompt = self._extract_last_user_message(body)
            classifier_request = self._build_classifier_request(prompt)

            clf_response = await generate_chat_completion(
                __request__, classifier_request, user
            )
            raw_label = self._parse_classifier_label(clf_response)

            category = raw_label.strip().lower()
            model_id = self.router.model_for(category)
            logger.info("Routing category '%s' to model '%s'", category, model_id)

            body["model"] = model_id
            response = await generate_chat_completion(__request__, body, user)

            if not self.valves.PREFACE_ENABLED:
                return response

            preface = build_preface(category, model_id)
            if isinstance(response, StreamingResponse):
                return wrap_stream_with_preface(response, preface)

            self._prepend_non_streaming_preface(response, preface)
            return response

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _extract_last_user_message(self, body: Dict[str, Any]) -> str:
            messages = (body or {}).get("messages")
            if not isinstance(messages, list) or not messages:
                return ""
            last = messages[-1] or {}
            return str(last.get("content", "")).strip()

        def _build_classifier_request(self, prompt: str) -> Dict[str, Any]:
            return {
                "model": self.valves.CLASSIFIER_MODEL_ID,
                "messages": [
                    {"role": "user", "content": self.router.classifier_prompt(prompt)}
                ],
                "stream": False,
            }

        def _parse_classifier_label(self, response: Dict[str, Any]) -> str:
            return response["choices"][0]["message"]["content"]  # type: ignore[index]

        def _prepend_non_streaming_preface(
            self, resp: Dict[str, Any], preface: str
        ) -> None:
            choices = resp["choices"]
            message = choices[0]["message"]
            content = message.get("content", "")
            message["content"] = preface + str(content)

    def pipes() -> list[dict[str, object]]:
        return [
            {
                "id": "prompt-router",
                "name": "Auto Prompt Router",
                "description": (
                    "Automatically select the right model based on your prompt. "
                    "Chooses between GPT-4o, GPT-4o-mini, Claude 4 Sonnet and Pixtral Large."
                ),
                "pipe": Pipe,
            }
        ]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def build_classifier_prompt(
    user_prompt: str, categories: Dict[str, Dict[str, str]]
) -> str:
    """Create a deterministic classification prompt for Nova Micro."""

    descriptions = "\n".join(
        f"- {name}: {cfg['description']}" for name, cfg in categories.items()
    )
    return (
        "You are a classification assistant.\n"
        "Your task is to read the user's prompt and assign it to exactly one "
        "category from the list below.\n"
        "Return only the category name in lowercase without explanations or "
        "additional text.\n\n"
        f"Categories:\n{descriptions}\n\n---\n\n"
        "Example input:\n"
        "Can you help me debug this Python error?\n"
        "Example output:\n"
        "coding\n\n---\n\n"
        "Now classify the following user prompt:\n"
        f"{user_prompt}"
    )


def build_preface(category: str, model_id: str) -> str:
    """Return the markdown preface for routing information."""

    return f"_(detected {category} prompt, routing to {model_id})_\n\n---\n\n"


def wrap_stream_with_preface(
    upstream: StreamingResponse, preface_text: str
) -> StreamingResponse:
    """Prepend a preface chunk to a streaming response."""

    async def stream() -> AsyncIterator[bytes]:
        chunk = {
            "id": "router-preface",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": preface_text}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

        async for part in upstream.body_iterator:
            yield part

    headers = dict(getattr(upstream, "headers", {}) or {})
    headers.pop("content-length", None)
    return StreamingResponse(stream(), media_type=upstream.media_type, headers=headers)


# ---------------------------------------------------------------------------
# Local CLI test mode
# ---------------------------------------------------------------------------


if not OPENWEBUI:

    def classify_with_bedrock(router: CategoryRouter, user_prompt: str) -> str:
        """Classify a prompt using AWS Bedrock (Nova Micro)."""

        model_id = os.getenv("BEDROCK_CLASSIFIER_MODEL_ID", "eu.amazon.nova-micro-v1:0")
        region = os.getenv("AWS_REGION", "eu-central-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": router.classifier_prompt(user_prompt)}],
                }
            ],
            "inferenceConfig": {"maxTokens": 50, "temperature": 0.0, "topP": 1.0},
        }

        resp = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(resp["body"].read())
        return result["output"]["message"]["content"][0]["text"].strip().lower()

    def main() -> None:
        router = CategoryRouter({})
        print("=== Local Router Test (Bedrock Nova Micro) ===")
        print(f"Region: {os.getenv('AWS_REGION', 'eu-central-1')}")
        print(
            f"Model : {os.getenv('BEDROCK_CLASSIFIER_MODEL_ID', 'eu.amazon.nova-micro-v1:0')}"
        )
        print("Type 'exit' or 'quit' to leave.\n")

        while True:
            try:
                prompt = input("Prompt> ")
            except KeyboardInterrupt:
                print("\nExiting.")
                sys.exit(0)
            if prompt.strip().lower() in {"exit", "quit"}:
                break

            raw = classify_with_bedrock(router, prompt)
            category = raw.strip().lower()
            model_id = router.model_for(category)
            print(f"Category     : {category}")
            print(f"Target model : {model_id}\n")

    if __name__ == "__main__":
        main()
