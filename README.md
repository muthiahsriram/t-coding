# AuraWealth

Consumer wealth management app — an AI advisor over your portfolio, goals and
net worth. Chat UI in Chainlit, Claude served through Databricks Foundation
Model APIs.

## Layout

```
app/
  config.py        credential resolution + model settings
  llm.py           async Databricks client (streaming + non-streaming)
  conversation.py  multi-turn history
  prompts.py       system prompts
  main.py          Chainlit entrypoint
app.yaml           Databricks Apps deployment descriptor
tests/             unit tests (no workspace needed)
```

## Deploying to Databricks Apps

1. Commit and push.
2. Sync the repo into the workspace (Repos / Git folder).
3. Point a Databricks App at the repo folder and deploy.

`app.yaml` binds `0.0.0.0:$DATABRICKS_APP_PORT` and Databricks installs
`requirements.txt` automatically.

**No secrets are needed.** The app authenticates as its own OAuth service
principal from the injected `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`,
resolved by `databricks-sdk`. Grant that service principal **CAN QUERY** on the
`databricks-claude-sonnet-4-5` serving endpoint or every call returns 403.

## Running elsewhere

```bash
pip install -r requirements.txt
cp .env.example .env          # DATABRICKS_HOST + DATABRICKS_TOKEN
chainlit run app/main.py -w
```

A personal access token in the environment takes precedence and bypasses the
SDK credential chain — that chain probes cloud metadata services and hangs on
networks that blackhole them.

## Tests

```bash
python -m pytest
```

The suite mocks the serving endpoint, so it needs no workspace and no network.
