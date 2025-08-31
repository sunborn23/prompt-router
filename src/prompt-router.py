import os, sys, json

try:
    # try OpenWebUI imports
    from pydantic import BaseModel, Field
    from fastapi import Request
    from open_webui.models.users import Users
    from open_webui.utils.chat import generate_chat_completion
    OPENWEBUI = True
except ImportError:
    # fallback to local imports
    import boto3
    OPENWEBUI = False

class Router:
    def __init__(self):
        # hardcoded categories + model mapping
        self.categories = {
            "default": {
                "model": "openai/gpt-4o-mini",
                "description": "For everyday, relatively simple requests with short to medium length (1–3 paragraphs). Covers small talk, short explanations, summaries, or general questions that do not require deep reasoning."
            },
            "coding": {
                "model": "anthropic/claude-4-sonnet",
                "description": "For technical requests related to programming, debugging, architecture, or IT tools. Includes code snippets, error messages, best practices, and in-depth technical explanations."
            },
            "deep-reasoning": {
                "model": "openai/gpt-4o",
                "description": "For complex, multi-layered tasks requiring high accuracy, creative solutions, or multimodality (e.g. images). Best suited for prompts with longer text (several paragraphs to full pages) or high logical complexity."
            },
            "structured-analysis": {
                "model": "anthropic/claude-4-sonnet",
                "description": "For tasks that need detailed, well-structured, text-heavy explanations. Useful for longer documents, strategic analyses, or conceptual requests requiring clarity and structure."
            },
            "content-generation": {
                "model": "openai/gpt-4o",
                "description": "For creative or business writing tasks such as blog posts, emails, marketing copy, or presentations. Focused on high-quality, coherent, and stylistically appropriate text generation."
            },
            "vision": {
                "model": "mistral/pixtral-large-2502",
                "description": "For requests involving images or visual content – such as image description, visual analysis, diagram interpretation, or multimodal tasks."
            }
        }

    def build_classifier_prompt(self, user_prompt: str) -> str:
        cat_desc = "\n".join(
            [f"- {name}: {cfg['description']}" for name, cfg in self.categories.items()]
        )
        return f"""
You are a classification assistant.
Your task is to read the user's prompt and assign it to exactly one category from the list below.
Return only the category *name* as plain text. Do not explain your choice. Do not output anything else.

Categories:
{cat_desc}

---

Example input:
"Can you help me debug this Python error?"
Example output:
Coding

---

Now classify the following user prompt:
{user_prompt}
"""


# -------------------
# OpenWebUI pipe mode
# -------------------
if OPENWEBUI:
    class Pipe:
        class Valves(BaseModel):
            CLASSIFIER_MODEL_ID: str = Field("amazon.nova-micro-v1:0", description="Classifier model in OpenWebUI")

        def __init__(self):
            self.valves = self.Valves()
            self.router = Router()

        def pipes(self):
            return [{
                "id": "AUTO_ROUTER",
                "name": "Auto-select model",
                "description": "Automatically select the right model based on your prompt. Chooses between GPT-4o, GPT-4o-mini, Claude 4 and Pixtral Large.",
                "pipe": self.pipe
            }]

        async def pipe(self, body: dict, __user__: dict, __request__: Request):
            user = Users.get_user_by_id(__user__["id"])
            user_prompt = body["messages"][-1]["content"]

            # Ask Nova Micro (via OpenWebUI generate_chat_completion)
            clf_body = {
                "model": self.valves.CLASSIFIER_MODEL_ID,
                "messages": [{"role": "user", "content": self.router.build_classifier_prompt(user_prompt)}],
                "stream": False
            }
            clf_resp = await generate_chat_completion(__request__, clf_body, user)
            category = clf_resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            if category not in self.router.categories:
                category = "Default"

            body["model"] = self.router.categories[category]["model"]
            return await generate_chat_completion(__request__, body, user)


# -------------------
# Local CLI test mode
# -------------------
else:
    def classify_with_bedrock(router: Router, user_prompt: str) -> str:
        MODEL_ID = "eu.amazon.nova-micro-v1:0"
        REGION = os.getenv("AWS_REGION", "eu-central-1")
        client = boto3.client("bedrock-runtime", region_name=REGION)

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
        # Nova Messages response shape
        try:
            return (
                result["output"]["message"]["content"][0]["text"]
                .strip()
            )
        except Exception:
            # Fallback for any alternate shape
            return result.get("outputText", "").strip()

    def main():
        router = Router()
        print("=== Local Router Test ===")
        while True:
            try:
                prompt = input("Prompt> ")
                if prompt.lower() in ["exit", "quit"]:
                    break
                category_raw = classify_with_bedrock(router, prompt)
                category = (category_raw or "").strip().lower()
                if category not in router.categories:
                    category = "default"
                print(f"Category: {category}")
                print(f"→ Target model: {router.categories[category]['model']}\n")
            except KeyboardInterrupt:
                sys.exit(0)

    if __name__ == "__main__":
        main()
