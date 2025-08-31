import os
import boto3
import json
from prompt_router import Pipe

MODEL_ID = "amazon.nova-micro-v1:0"
REGION = os.getenv("AWS_REGION", "us-east-1")

client = boto3.client("bedrock-runtime", region_name=REGION)

def classify_with_bedrock(pipe: Pipe, user_prompt: str) -> str:
    body = {
        "inputText": pipe.build_classifier_prompt(user_prompt),
        "textGenerationConfig": {"maxTokenCount": 50, "temperature": 0.0, "topP": 1.0}
    }
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(body))
    result = json.loads(response["body"].read())
    return result["outputText"].strip()

def main():
    pipe = Pipe()
    print("=== Local Router Test ===")
    while True:
        user_prompt = input("Prompt> ")
        if user_prompt.lower() in ["exit", "quit"]:
            break
        category = classify_with_bedrock(pipe, user_prompt)
        if category not in pipe.categories:
            category = "Default"
        target_model = pipe.categories[category]["model"]
        print(f"Category: {category}")
        print(f"→ Target model: {target_model}\n")

if __name__ == "__main__":
    main()
