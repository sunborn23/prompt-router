"""
title: Prompt Router Pipe
author: sunborn23
author_url: https://github.com/sunborn23
repo_url: https://github.com/sunborn23/prompt-router
version: 0.6
"""

import json
import os
import sys
from typing import Any, Dict

try:  # OpenWebUI runtime (Pipe mode)
    from fastapi import Request
    from open_webui.models.users import Users
    from open_webui.utils.chat import generate_chat_completion
    from pydantic import BaseModel, Field

    OPENWEBUI = True
except ImportError:  # Local CLI runtime
    import boto3

    OPENWEBUI = False


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

    def classifier_messages(self, user_prompt: str) -> list[Dict[str, str]]:
        return build_classifier_messages(user_prompt, self.categories)

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
            ROUTING_STATUS_ENABLED: bool = Field(
                True, description="If false, routing status events will be omitted."
            )

        def __init__(self) -> None:
            self.valves = self.Valves()
            self.router = CategoryRouter(self.valves.model_dump())

        async def pipe(
                self,
                body: Dict[str, Any],
                __user__: Dict[str, Any],
                __request__: Request,
                __event_emitter__=None,
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

            if self.valves.ROUTING_STATUS_ENABLED and __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f'Detected prompt category "{category}", routing to "{model_id}"',
                            "done": True,
                            "hidden": False,
                        },
                    }
                )

            body["model"] = model_id
            return await generate_chat_completion(__request__, body, user)

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
                "messages": self.router.classifier_messages(prompt),
                "stream": False,
            }

        def _parse_classifier_label(self, response: Any) -> str:
            if not isinstance(response, dict):
                response = json.loads(response.body)
            return response["choices"][0]["message"]["content"]  # type: ignore[index]


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


def build_classifier_messages(
        user_prompt: str, categories: Dict[str, Dict[str, str]]
) -> list[Dict[str, str]]:
    descriptions = "\n".join(
        f"- {name}: {cfg['description']}" for name, cfg in categories.items()
    )
    valid_labels = ", ".join(categories.keys())
    system = (
        "You are a strict classification assistant. "
        "Choose EXACTLY ONE category from the allowed list. "
        "Return ONLY the category token in lowercase, without punctuation, "
        "quotes or explanation. If uncertain, return 'default'.\n\n"
        f"Valid categories: {valid_labels}\n\n"
        f"Categories:\n{descriptions}"
    )
    user = f"Classify the following user prompt:\n{user_prompt}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_bedrock_body(router_messages: list[Dict[str, str]],
                       max_tokens: int = 6,
                       temperature: float = 0.0,
                       top_p: float = 0.0) -> Dict[str, Any]:
    system_chunks: list[str] = []
    converted: list[Dict[str, Any]] = []

    for msg in router_messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))

        if role == "system":
            if content:
                system_chunks.append(content)
            continue

        if role == "user":
            converted.append({
                "role": "user",
                "content": [{"text": content}]
            })

    body: Dict[str, Any] = {
        "messages": converted,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }
    if system_chunks:
        body["system"] = [{"text": "\n\n".join(system_chunks)}]

    return body


# ---------------------------------------------------------------------------
# Local CLI test mode
# ---------------------------------------------------------------------------


if not OPENWEBUI:

    def classify_with_bedrock(router: CategoryRouter, user_prompt: str) -> str:
        """Classify a prompt using AWS Bedrock (Nova Micro)."""

        model_id = os.getenv("BEDROCK_CLASSIFIER_MODEL_ID", "eu.amazon.nova-micro-v1:0")
        region = os.getenv("AWS_REGION", "eu-central-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        # Hardcodiertes Mapping: Wir erwarten zwei Messages in fester Reihenfolge
        # 0: system, 1: user
        msgs = router.classifier_messages(user_prompt)
        system_msg = str(msgs[0]["content"])
        user_msg = str(msgs[1]["content"])

        body = {
            "system": [{"text": system_msg}],
            "messages": [
                {"role": "user", "content": [{"text": user_msg}]},
            ],
            "inferenceConfig": {"maxTokens": 6, "temperature": 0.0, "topP": 0.0},
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
