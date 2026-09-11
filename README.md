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
  market.py        instrument universe + FX rates
  domain.py        holdings, accounts, property, liabilities, goals, clients
  seed.py          the deterministic book of 8 demo clients
  portfolio.py     analytics + the pre-computed briefing handed to the model
  identity.py      which client is on screen (SSO + profile picker)
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

## Who the app thinks you are

Two independent things:

- **Who is operating the app** — Databricks Apps authenticates the workspace
  user and forwards them in `X-Forwarded-Email`. Shown in the welcome message.
- **Whose portfolio is on screen** — the chat-profile dropdown at the top of the
  UI, one entry per seeded client. This wins over SSO, because access is
  restricted to a few corporate IDs and a demo needs to switch between clients.

To point a real workspace ID at a seeded client (so it's the default when no
profile is picked), set `AURA_CLIENT_MAP` in `app.yaml`:

```yaml
- name: "AURA_CLIENT_MAP"
  value: "jane.doe@fedex.com:C002,john.roe@fedex.com:C005"
```

An unknown ID falls back to `DEFAULT_CLIENT_ID`; a mapping to a client that
doesn't exist fails at startup rather than silently serving the wrong person.

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
