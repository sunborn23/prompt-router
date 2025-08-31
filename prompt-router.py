"""
title: Prompt Router Pipe
author: sunborn23
author_url: https://github.com/sunborn23
repo_url: https://github.com/sunborn23/prompt-router
version: 1.0
"""

import os
import sys
import json

try:
    # -------------------------------
    # OpenWebUI runtime (Pipe mode)
    # -------------------------------
    from pydantic import BaseModel, Field
    from fastapi import Request
    from open_webui.models.users import Users
    from open_webui.utils.chat import generate_chat_completion
    from starlette.responses import StreamingResponse

    OPENWEBUI = True
except ImportError:
    # -------------------------------
    # Local CLI runtime (no OpenWebUI)
    # -------------------------------
    import boto3  # Only needed locally

    OPENWEBUI = False


class Router:
    """
    Core classification prompt builder and category registry.
    OpenWebUI and CLI modes both use this class to keep logic DRY.
    """

    def __init__(self, valves=None):
        # ---------------------------------------------------------------------
        # Hardcoded categories and model mapping
        #
        # Use valves to inject the actual target model IDs from the Admin UI.
        # Fallbacks are just examples – please copy the exact IDs from Admin → Models.
        # ---------------------------------------------------------------------
        v = valves or {}
        self.categories = {
            "default": {
                "model": v.get("MODEL_ID_DEFAULT", "azure.gpt-4o-mini-example"),
                "description": "For everyday, relatively simple requests with short to medium length (1–3 paragraphs). Covers small talk, short explanations, summaries, or general questions that do not require deep reasoning."
            },
            "coding": {
                "model": v.get("MODEL_ID_CODING", "eu.anthropic.claude-sonnet-4.1-example"),
                "description": "For technical requests related to programming, debugging, architecture, or IT tools. Includes code snippets, error messages, best practices, and in-depth technical explanations."
            },
            "deep-reasoning": {
                "model": v.get("MODEL_ID_DEEP", "azure.gpt-4o-example"),
                "description": "For complex, multi-layered tasks requiring high accuracy, creative solutions, or multimodality (e.g. images). Best suited for prompts with longer text (several paragraphs to full pages) or high logical complexity."
            },
            "structured-analysis": {
                "model": v.get("MODEL_ID_STRUCT", "eu.anthropic.claude-sonnet-4.1-example"),
                "description": "For tasks that need detailed, well-structured, text-heavy explanations. Useful for longer documents, strategic analyses, or conceptual requests requiring clarity and structure."
            },
            "content-generation": {
                "model": v.get("MODEL_ID_CONTENT", "azure.gpt-4o-example"),
                "description": "For creating polished or longer-form text such as emails, blog posts, marketing copy, reports, or presentations. Emphasis on tone, coherence, and style control. Typically multiple paragraphs or more."
            },
            "vision": {
                "model": v.get("MODEL_ID_VISION", "eu.mistral.pixtral-large-2502-v1:0-example"),
                "description": "For requests involving images or visual content – such as image description, visual analysis, diagram interpretation, or multimodal tasks."
            },
        }

    def build_classifier_prompt(self, user_prompt: str) -> str:
        """
        Build a strict classification prompt for Nova Micro.

        Contract:
        - Return only one of the known category names (lowercase).
        - No extra text. No explanations.
        """
        cat_desc = "\n".join(
            [f"- {name}: {cfg['description']}" for name, cfg in self.categories.items()]
        )
        return f"""
You are a classification assistant.
Your task is to read the user's prompt and assign it to exactly one category from the list below.
Return only the category *name* as plain text in lowercase. Do not explain your choice. Do not output anything else.

Categories:
{cat_desc}

---

Example input:
"Can you help me debug this Python error?"
Example output:
coding

---

Now classify the following user prompt:
{user_prompt}
""".strip()


# =============================================================================
# OpenWebUI Pipe mode
# =============================================================================
if OPENWEBUI:

    class Pipe:
        """
        OpenWebUI Pipe that:
        1) Calls a small classifier model (e.g., Nova Micro) to label the intent.
        2) Rewrites the request body with the target model based on that label.
        3) Forwards the original conversation to the chosen model.
        """

        class Valves(BaseModel):
            """
            Admin-configurable knobs exposed in the WebUI.
            """
            # Classifier (make sure the prefix matches your environment, e.g. 'eu.amazon...')
            MODEL_ID_CLASSIFIER: str = Field(
                default="eu.amazon.nova-micro-v1:0",
                description="Model ID used for classification for routing decisions.",
            )
            # Target models – update with exact IDs from Admin → Models
            MODEL_ID_DEFAULT: str = Field(
                "azure.gpt-4o-mini",
                description="Model ID for default / smalltalk prompts"
            )
            MODEL_ID_CODING: str = Field(
                "eu.anthropic.claude-sonnet-4-20250514-v1:0",
                description="Model ID for coding / tech prompts"
            )
            MODEL_ID_DEEP: str = Field(
                "azure.gpt-4o",
                description="Model ID for deep reasoning / complex queries prompts"
            )
            MODEL_ID_STRUCT: str = Field(
                "eu.anthropic.claude-sonnet-4-20250514-v1:0",
                description="Model ID for structured analysis prompts"
            )
            MODEL_ID_CONTENT: str = Field(
                "azure.gpt-4o",
                description="Modle ID for content generation / long-form writing prompts"
            )
            MODEL_ID_VISION: str = Field(
                "eu.mistral.pixtral-large-2502-v1:0",
                description="Model ID for vision / multimodal prompts"
            )

        def __init__(self):
            # Initialize configurable valves and the router registry once.
            self.valves = self.Valves()
            self.router = Router(valves=self.valves.model_dump())

        def pipes(self):
            return [
                {
                    "id": "PROMPT_ROUTER",
                    "name": "Auto Prompt Router",
                    "description": (
                        "Automatically select the right model based on your prompt. "
                        "Chooses between GPT-4o, GPT-4o-mini, Claude 4 Sonnet and Pixtral Large."
                    ),
                    "pipe": self.pipe,
                }
            ]

        async def pipe(self, body: dict, __user__: dict, __request__: Request):
            """
            Main entry point for routing a user prompt to the most suitable model.
            Steps:
            1) Extract user prompt
            2) Classify intent via small classifier model
            3) Resolve target model (fallback to 'default' if unknown)
            4) Forward the original request to the resolved model
            5) Prepend routing info to the response (streaming or non-streaming)
            """
            # Resolve OpenWebUI user for provider invocation
            user = Users.get_user_by_id((__user__ or {}).get("id"))

            # 1) Extract last user message content (defensive access)
            user_prompt = self._extract_last_user_message(body)

            # 2) Build and call classifier request
            classifier_body = self._build_classifier_request(user_prompt)
            classifier_response = await generate_chat_completion(__request__, classifier_body, user)

            # 3) Parse and normalize category; fallback to 'default'
            raw_category = self._parse_category_from_response(classifier_response)
            category = self._normalize_category(raw_category)

            # 4) Route to target model and forward original request
            target_model_id = self._resolve_target_model(category)
            body["model"] = target_model_id
            routed_response = await generate_chat_completion(__request__, body, user)

            # 5) Prepend routing info (works for streaming and non-streaming responses)
            if isinstance(routed_response, StreamingResponse):
                return await self._prepend_streaming_preface(routed_response, target_model_id)
            else:
                self._prepend_non_streaming_preface(routed_response, target_model_id)
                return routed_response

        # -----------------------------
        # Internal helper methods
        # -----------------------------

        def _extract_last_user_message(self, body: dict) -> str:
            """Safely extract the latest user message content from the request body."""
            messages = (body or {}).get("messages")
            if not isinstance(messages, list) or not messages:
                return ""
            last = messages[-1] or {}
            return str(last.get("content", "")).strip()

        def _build_classifier_request(self, user_prompt: str) -> dict:
            """Create request body for the small classifier model."""
            return {
                "model": self.valves.MODEL_ID_CLASSIFIER,
                "messages": [
                    {"role": "user", "content": self.router.build_classifier_prompt(user_prompt)}
                ],
                "stream": False,
            }

        def _parse_category_from_response(self, clf_resp: dict) -> str:
            """Parse raw category string from classifier response."""
            choices = (clf_resp or {}).get("choices") or []
            first = (choices[0] if choices else {}) or {}
            message = first.get("message") or {}
            raw = message.get("content", "")
            return str(raw).strip().lower()

        def _normalize_category(self, category: str) -> str:
            """Normalize to a known category or fallback to 'default'."""
            return category if category in self.router.categories else "default"

        def _resolve_target_model(self, category: str) -> str:
            """Resolve model id for a category, fallback to MODEL_ID_DEFAULT."""
            entry = self.router.categories.get(category, {})
            return entry.get("model", getattr(self.valves, "MODEL_ID_DEFAULT", "default"))

        async def _prepend_streaming_preface(self, upstream: StreamingResponse, model_id: str) -> StreamingResponse:
            """
            Wrap StreamingResponse and inject a small preface chunk before
            forwarding the original upstream stream untouched.
            """
            async def generator():
                # Send one OpenAI-style SSE chunk with a delta containing the preface.
                preface_text = f"_(routing to: {model_id})_\n\n---\n\n"
                preface_chunk = {
                    "id": "router-info",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": preface_text}}],
                }
                yield f"data: {json.dumps(preface_chunk)}\n\n".encode("utf-8")

                # Forward original stream without modification.
                async for chunk in upstream.body_iterator:
                    yield chunk

            # Preserve media type and headers, avoid content-length mismatch.
            headers = dict(getattr(upstream, "headers", {}) or {})
            headers.pop("content-length", None)
            return StreamingResponse(
                generator(),
                media_type=getattr(upstream, "media_type", "text/event-stream"),
                headers=headers,
            )

        def _prepend_non_streaming_preface(self, resp: dict, model_id: str) -> None:
            """Prepend routing preface to the assistant message for non-stream responses."""
            try:
                choices = (resp or {}).get("choices")
                if not isinstance(choices, list) or not choices:
                    return
                first = choices[0] or {}
                message = first.get("message")
                if not isinstance(message, dict):
                    return
                original = message.get("content") or ""
                preface = f"_(routing to: {model_id})_\n\n---\n\n"
                message["content"] = preface + str(original)
            except Exception:
                # Keep silent in production flow; preface is non-critical.
                pass


# =============================================================================
# Local CLI test mode
# =============================================================================
else:
    def classify_with_bedrock(router: Router, user_prompt: str) -> str:
        """
        Local classifier call using boto3 and Amazon Bedrock (Nova Micro), mainly used for testing.
        """
        MODEL_ID = os.getenv("BEDROCK_CLASSIFIER_MODEL_ID", "eu.amazon.nova-micro-v1:0")
        REGION = os.getenv("AWS_REGION", "eu-central-1")

        client = boto3.client("bedrock-runtime", region_name=REGION)

        # Nova (Messages API) request body
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": router.build_classifier_prompt(user_prompt)}],
                }
            ],
            "inferenceConfig": {"maxTokens": 50, "temperature": 0.0, "topP": 1.0},
        }

        resp = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(resp["body"].read())

        # Try Messages output first; fall back to legacy key if present.
        try:
            return result["output"]["message"]["content"][0]["text"].strip().lower()
        except Exception:
            return (result.get("outputText", "") or "").strip().lower()


    def main():
        """
        Simple REPL for local testing:
        - Runs the same classifier prompt locally against Bedrock Nova Micro.
        - Prints the resolved category and mapped target model.
        """
        router = Router()
        print("=== Local Router Test (Bedrock Nova Micro) ===")
        print(f"Region: {os.getenv('AWS_REGION', 'eu-central-1')}")
        print(f"Model : {os.getenv('BEDROCK_CLASSIFIER_MODEL_ID', 'eu.amazon.nova-micro-v1:0')}")
        print("Type 'exit' or 'quit' to leave.\n")

        while True:
            try:
                prompt = input("Prompt> ")
                if prompt.strip().lower() in ("exit", "quit"):
                    break

                category = classify_with_bedrock(router, prompt)
                if category not in router.categories:
                    category = "default"

                print(f"Category     : {category}")
                print(f"Target model : {router.categories[category]['model']}\n")

            except KeyboardInterrupt:
                print("\nExiting.")
                sys.exit(0)
            except Exception as e:
                print("Error:", e)


    if __name__ == "__main__":
        main()
