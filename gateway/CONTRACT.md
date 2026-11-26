# Tekti gateway: contract

Owned by the main orchestrator. The gateway code and the Tekti client are built against this file.

## Purpose

Let another person run Tekti on the same vendor models as the author without holding the author's vendor keys. The gateway is a thin HTTPS pass-through: it checks a gateway token, enforces limits, swaps the token for the real vendor key, forwards the request unchanged and returns the vendor's response unchanged. It does no translation between vendor formats.

## Address and routes

Base: the Lambda function URL, `https://<id>.lambda-url.<region>.on.aws`.

| Prefix | Forwarded to | Allowed (POST only) |
| :--- | :--- | :--- |
| `/openai/` | `https://api.openai.com/` | `/v1/chat/completions`, `/v1/responses` |
| `/anthropic/` | `https://api.anthropic.com/` | `/v1/messages` |
| `/google/` | `https://generativelanguage.googleapis.com/` | `/v1beta/models/<model>:generateContent`, `/v1beta/models/<model>:streamGenerateContent`, `/v1beta/openai/chat/completions` |

`GET /health` returns `{"ok": true}` without a token. Everything else is 404.

## Authentication

The client puts the **gateway token** wherever that vendor's SDK normally puts its API key, so SDKs work unmodified with `base_url = <gateway>/<prefix>` and `api_key = <gateway token>`:

- `Authorization: Bearer <token>` (OpenAI SDK),
- `x-api-key: <token>` (Anthropic SDK),
- `x-goog-api-key: <token>` or `?key=<token>` (Google SDK).

The gateway accepts the token from any of these, removes all of them, and sets the vendor's real credential in that vendor's own place. Tokens are compared in constant time. A missing or wrong token is 401 with a JSON body; nothing is forwarded.

There are two named tokens, `owner` and `jury`, each with its own request cap.

## Limits

- **Model allowlist:** the model named in the request (JSON body `model`, or the path for Google) must be in the configured list, else 403. This stops a leaked token being used for other models.
- **Request cap per token:** a counter in DynamoDB, incremented atomically before forwarding; at the cap the gateway answers 429 with a clear JSON message. Caps are deployment parameters.
- **Expiry:** an optional date after which the `jury` token is refused (403).
- **Body size:** requests over 1 MB are refused (413).
- **Concurrency:** reserved concurrency on the function.

## What is kept

One log line per request: time, token name (never the token), vendor, model, status, duration, and token usage if the vendor's response reports it. No prompts, no responses, no headers.

## Errors

Vendor errors pass through with the vendor's status and body. Gateway errors use `{"error": {"type": "tekti_gateway", "code": "<code>", "message": "<text>"}}`.

## Client side (Tekti)

`.env` for someone using the gateway:

```
TEKTI_GATEWAY_URL=https://<id>.lambda-url.<region>.on.aws
TEKTI_GATEWAY_TOKEN=<token>
```

With these set, every `openai`, `anthropic` and `google` model is called through the gateway. Without them, Tekti calls the vendors directly using `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY`.

## Sign-in with GitHub (replaces the `jury` token)

Decided 26 November 2026: no credential is ever sent to the jury. A jury member proves who they are with their own GitHub account, and the gateway gives that person a short-lived session. The `jury` token and its secret are removed. The `owner` token stays.

### Flow

1. The person runs `tekti login`. The CLI starts GitHub's device flow with the project's GitHub OAuth app **client id** (public, not a secret; no client secret is used) and **no scopes**, prints the code and address, and waits for approval in the browser.
2. The CLI sends the resulting GitHub access token once to the gateway: `POST /auth/github` with JSON `{"github_token": "..."}`.
3. The gateway calls `GET https://api.github.com/user` with that token to learn the login and numeric id, checks the allowlist, and answers `{"session": "<token>", "login": "<login>", "expires_at": "<ISO time>"}`. The GitHub token is not stored or logged by the gateway.
4. The CLI stores the session in the user's config directory with owner-only permissions and discards the GitHub token.
5. Every model request carries the session where the vendor SDK puts its API key, exactly like the owner token.

### Session token

`tks1.<base64url payload>.<base64url HMAC-SHA256>`; the payload is JSON with `login` (lower-cased), `id`, `exp` (Unix seconds). The signing key is a generated Secrets Manager secret read by the function. Lifetime 24 hours. Verified locally with a constant-time comparison; an expired or malformed session is 401 with code `session_expired` or `invalid_token`, and the CLI tells the person to run `tekti login` again.

### Who is allowed

- `AllowedGithubLogins` (template parameter, comma-separated, case-insensitive). When it is non-empty, only those logins get a session (others: 403 `not_on_allowlist`). When empty, any GitHub user gets one.
- The allowlist is checked again on every request, so removing a login takes effect at the next deploy without waiting for sessions to expire.
- The date parameter that used to expire the jury token now applies to all GitHub sessions (`GithubAccessExpiryDate`).

### Limits for signed-in users

- **Per person:** `GithubUserRequestCap` requests in total per login (default 400).
- **All signed-in users together:** `GithubTotalRequestCap` (default 4000). Both counters are incremented atomically; a request that would pass one but not the other must not consume either.
- `/auth/github` itself is limited to a small number of calls per login per day and does not count against the model caps.
- The model allowlist, body size limit and logging rules are unchanged. The log's `token` field is `owner` or `gh:<login>`.

### Client side

- `tekti login`, `tekti logout`, `tekti whoami` (shows login, expiry, and never the session).
- With `TEKTI_GATEWAY_URL` set: `TEKTI_GATEWAY_TOKEN` (the owner's) is used if present; otherwise the stored session; otherwise the run stops with a one-line instruction to run `tekti login`.
- `TEKTI_GITHUB_CLIENT_ID` overrides the built-in client id.
