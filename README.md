# prompt-router

tbd

## Requirements

- Python 3.12+
- Dependencies listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```

### AWS credentials

The script uses AWS Bedrock for AI tasks. Set the following environment
variables before running the script:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...      # optional
AWS_REGION=us-east-1       # or use AWS_DEFAULT_REGION
```