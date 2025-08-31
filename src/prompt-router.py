from pydantic import BaseModel, Field
from fastapi import Request
from open_webui.models.users import Users
from open_webui.utils.chat import generate_chat_completion
import boto3, json

class Pipe:
    class Valves(BaseModel):
        AWS_REGION: str = Field("us-east-1", description="AWS Region for Bedrock")
        CLASSIFIER_MODEL_ID: str = Field("amazon.nova-micro-v1:0", description="Nova Micro model ID")

    def __init__(self):
        self.valves = self.Valves()

        # Hardcoded categories + model mapping
        self.categories = {
            "Default": {
                "model": "openai/gpt-4o-mini",
                "description": "For everyday, relatively simple requests with short to medium length (1–3 paragraphs). Covers small talk, short explanations, summaries, or general questions that do not require deep reasoning."
            },
            "Coding": {
                "model": "openai/gpt-4o-mini",
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

        # AWS Bedrock client
        self.client = boto3.client("bedrock-runtime", region_name=self.valves.AWS_REGION)

    def pipes(self):
        return [{"id": "AUTO_ROUTER", "name": "AUTO Router"}]

    async def pipe(self, body: dict, __user__: dict, __request__: Request):
        user = Users.get_user_by_id(__user__["id"])

        # Extract last user message
        user_prompt = body["messages"][-1]["content"]

        # Classify with Nova Micro
        category = self.classify_with_nova(user_prompt)
        if category not in self.categories:
            category = "Default"

        # Route to mapped model
        body["model"] = self.categories[category]["model"]

        return await generate_chat_completion(__request__, body, user)

    def build_classifier_prompt(self, user_prompt: str) -> str:
        cat_desc = "\n".join(
            [f"- {name}: {cfg['description']}" for name, cfg in self.categories.items()]
        )
        return f"""
You are a classification assistant. Your task is to assign the user's prompt to exactly one category.
Return only the category name as plain text.

Categories:
{cat_desc}

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
