from pydantic import BaseModel, Field
from fastapi import Request
from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion

class Pipe:
    class Valves(BaseModel):
        CLASSIFIER_MODEL_ID: str = Field("amazon.nova-micro-v1:0", description="Classifier model in OpenWebUI")

    def __init__(self):
        self.valves = self.Valves()

        # Hardcoded categories + model mapping
        self.categories = {
            "Default": {
                "model": "openai/gpt-4o-mini",
                "description": "For everyday, relatively simple requests with short to medium length (1–3 paragraphs). Covers small talk, short explanations, summaries, or general questions that do not require deep reasoning."
            },
            "Coding": {
                "model": "anthropic/claude-4-sonnet",
                "description": "For technical requests related to programming, debugging, architecture, or IT tools. Includes code snippets, error messages, best practices, and in-depth technical explanations."
            },
            "DeepReasoning": {
                "model": "openai/gpt-4o",
                "description": "For complex, multi-layered tasks requiring high accuracy, creative solutions, or multimodality (e.g. images). Best suited for prompts with longer text (several paragraphs to full pages) or high logical complexity."
            },
            "StructuredAnalysis": {
                "model": "anthropic/claude-4-sonnet",
                "description": "For tasks that need detailed, well-structured, text-heavy explanations. Useful for longer documents, strategic analyses, or conceptual requests requiring clarity and structure."
            },
            "Vision": {
                "model": "mistral/pixtral-large-2502",
                "description": "For requests involving images or visual content – such as image description, visual analysis, diagram interpretation, or multimodal tasks."
            }
        }

    def pipes(self):
        return [{"id": "AUTO_ROUTER", "name": "Auto-select model", "description": "Automatically select the right model based on your prompt. Chooses between GPT-4o, GPT-4o-mini, Claude 4 and Mixtral Large 2502.", "pipe": self.pipe}]

    async def pipe(self, body: dict, __user__: dict, __request__: Request):
        user = Users.get_user_by_id(__user__["id"])

        # Get last user message
        user_prompt = body["messages"][-1]["content"]

        # Build classification prompt
        classifier_prompt = self.build_classifier_prompt(user_prompt)

        # Ask Nova Micro inside OpenWebUI
        clf_body = {
            "model": self.valves.CLASSIFIER_MODEL_ID,
            "messages": [{"role": "user", "content": classifier_prompt}],
            "stream": False
        }
        clf_resp = await generate_chat_completion(__request__, clf_body, user)
        category = clf_resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if category not in self.categories:
            category = "Default"

        body["model"] = self.categories[category]["model"]
        return await generate_chat_completion(__request__, body, user)

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

    def classify_with_nova(self, user_prompt: str) -> str:
        body = {
            "inputText": self.build_classifier_prompt(user_prompt),
            "textGenerationConfig": {"maxTokenCount": 50, "temperature": 0.0, "topP": 1.0}
        }
        response = self.client.invoke_model(
            modelId=self.valves.CLASSIFIER_MODEL_ID,
            body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        return result["outputText"].strip()
