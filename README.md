# Prompt Router

This project provides a **single-file Prompt Router** (`prompt-router.py`) that can be used:

1. As an **OpenWebUI Pipe** to automatically route user prompts to the most suitable model.
2. As a **standalone CLI tool** for local testing with AWS Bedrock (Nova Micro).

The router uses **Amazon Nova Micro** to classify prompts into categories and maps them to the appropriate target model (GPT-4o, Claude 4, Pixtral, etc.).

---

## Categories

The router currently supports these categories:

- **default** → `openai/gpt-4o-mini`  
  Everyday, simple requests (1–3 paragraphs). Small talk, summaries, general questions.

- **coding** → `anthropic/claude-4-sonnet`  
  Technical requests: programming, debugging, error messages, best practices.

- **deep-reasoning** → `openai/gpt-4o`  
  Complex, multi-layered tasks requiring high accuracy, creativity or multimodality. Suitable for long texts or complex logic.

- **structured-analysis** → `anthropic/claude-4-sonnet`  
  Detailed, well-structured, text-heavy explanations (long documents, strategies, conceptual tasks).

- **content-generation** → `openai/gpt-4o`  
  Creative or business writing: blog posts, emails, marketing copy, or presentations. Focused on high-quality and stylistically coherent text generation.

- **vision** → `mistral/pixtral-large-2502`  
  Prompts involving images or visual content (image description, diagram analysis, multimodal tasks).

---

## Usage in OpenWebUI

1. Go to Admin → Functions → New in the OpenWebUI interface.
2. Create a new Pipe Function, give it a meaningful name and description.
3. Copy the entire Python code from `prompt-router.py` into the editor and save.
4. Enable the function using the toggle.
5. In the model selector, you will see a new entry: **Auto Prompt Router**.
6. Select it, and the router will automatically classify and forward prompts.

Check the Valves values to configure the router. Available options include:

- `CLASSIFIER_MODEL_ID` – model ID used for classification.
- `MODEL_DEFAULT` – model ID for default/smalltalk prompts.
- `MODEL_CODING` – model ID for coding/tech prompts.
- `MODEL_DEEP` – model ID for deep reasoning/complex queries.
- `MODEL_STRUCT` – model ID for structured analysis.
- `MODEL_CONTENT` – model ID for content generation.
- `MODEL_VISION` – model ID for vision/multimodal prompts.
- `PREFACE_ENABLED` – toggle to include or omit the routing prefaces.

## Logging

The router logs the detected category, the requested model, and the model reported in the response at the INFO level.

---

## Local CLI Test with AWS Bedrock

For local testing, the same file can be executed directly:

    python prompt-router.py

This will start an interactive loop:

    === Local Router Test ===
    Prompt> Can you help me debug this Python error?
    Category: coding
    → Target model: anthropic/claude-4-sonnet

### Requirements

Install dependencies:

    pip install boto3

### AWS Credentials

The CLI mode uses `boto3` to access **Amazon Bedrock**.  
Make sure the following environment variables are set:

Linux/macOS (bash):

    export AWS_ACCESS_KEY_ID=your-access-key-id
    export AWS_SECRET_ACCESS_KEY=your-secret-access-key
    export AWS_REGION=eu-central-1

Windows PowerShell:

    $env:AWS_ACCESS_KEY_ID="your-access-key-id"
    $env:AWS_SECRET_ACCESS_KEY="your-secret-access-key"
    $env:AWS_REGION="eu-central-1"

---

## Notes

- The router assumes the classifier returns a valid category; unknown categories raise an error.
- In OpenWebUI mode, Bedrock is called via the integrated `generate_chat_completion`.
- In CLI mode, Bedrock is accessed directly via `boto3`.
- If a model request fails, the router returns an error so you can manually select a model.

## Manual test steps

- Non-stream test and verify preface renders correctly (no Markdown heading).
- Stream test and verify preface chunk arrives first.
- Unknown category → error.

## License

see LICENSE file
