# Tekti gateway

> **Status, 27 November 2026:** not deployed. The stack was deleted on 27 November 2026, and the gateway
> is not a way for the jury to run Tekti: use your own vendor keys or an OpenRouter key (see the root
> README). The code stays in the repository; this page describes how it works if someone deploys it.

A thin HTTPS pass-through on AWS Lambda. It lets other people (the hackathon jury) run Tekti against
your OpenAI, Anthropic and Google models without ever holding your vendor keys or any credential of
yours. They prove who they are with their own GitHub account; you call it with your `owner` token.
The contract it implements is in [CONTRACT.md](CONTRACT.md).

What one deployment (one CloudFormation stack) contains:

| Resource | Purpose |
| :--- | :--- |
| Lambda function `<stack>-gateway` (Python, standard library + the runtime's boto3) | checks the token or session, enforces limits, forwards to the vendor, returns the vendor's answer unchanged |
| Lambda function URL, auth type `NONE` | the public HTTPS address; the checks in code are the only authentication |
| Secret `<stack>/vendor-keys` | your vendor API keys as JSON; created with `REPLACE_ME` placeholders that you replace |
| Secret `<stack>/token-owner` | your own gateway token, 40 random letters and digits |
| Secret `<stack>/session-signing-key` | the key that signs GitHub sessions, 64 random letters and digits |
| DynamoDB table | request counters (owner, each GitHub login, all GitHub logins together, daily sign-ins) |
| CloudWatch log group `/aws/lambda/<stack>-gateway` | one JSON line per request, kept 30 days |
| IAM role | read exactly those three secrets, update items in that one table, write that one log group |

Nothing in the template, the stack outputs or the logs contains a key, token or session. Only someone
who can read the secrets in your AWS account (you) can see them.

## How the jury gets access

1. You put their GitHub logins in `AllowedGithubLogins` and deploy (see below).
2. They set `TEKTI_GATEWAY_URL=https://<id>.lambda-url.<region>.on.aws` and run `tekti login`.
   The CLI starts GitHub's device flow (no scopes), they approve in the browser, and the CLI sends the
   resulting GitHub token once to `POST /auth/github`.
3. The gateway asks GitHub (`GET https://api.github.com/user`) who owns that token, checks the login
   against `AllowedGithubLogins`, and answers with a session valid for 24 hours (or until the end of
   `GithubAccessExpiryDate`, if that is sooner). The GitHub token is neither stored nor logged.
4. The CLI keeps the session in the user's config directory and sends it with every model request.
   After 24 hours, they run `tekti login` again.

You send them nothing but the gateway URL.

## Deploy

Requirements: AWS CLI v2 with credentials for the target account, and permission to create the
resources above (CloudFormation needs `CAPABILITY_IAM` for the role).

```bash
gateway/deploy.sh <region> <stack-name> [--bucket <artifact-bucket>] [ParamKey=Value ...]

# example
gateway/deploy.sh us-east-1 tekti-gateway \
  AllowedModels=gpt-5,claude-sonnet-4-5,gemini-2.5-pro \
  AllowedGithubLogins=alice,bob-the-judge GithubAccessExpiryDate=2026-12-05
```

The script stages `handler.py`, runs `aws cloudformation package` (zips and uploads the code) and
`aws cloudformation deploy`, then prints the stack outputs. There is no build step. On an update,
any parameter you do not pass keeps its current value.

Without `--bucket`, the code is uploaded to a bucket named `<stack-name>-artifacts-<account-id>-<region>`
which the script creates if it is missing (public access blocked, encrypted, TLS only, objects expire
after 90 days). That bucket is outside the stack; delete it yourself after tearing down.

### Parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `AllowedModels` | `gpt-5,claude-sonnet-4-5,gemini-2.5-pro` (**placeholder**) | Comma-separated model names that may be called by anyone. Exact match; for Google a leading `models/` is ignored. Everything else gets 403. |
| `OwnerRequestCap` | `3000` | Total requests the `owner` token may make. `0` blocks it. |
| `AllowedGithubLogins` | empty | Comma-separated GitHub logins (case-insensitive) that may sign in. Checked at sign-in **and on every request**. **Empty means any GitHub user in the world may sign in**, limited only by the caps. |
| `GithubUserRequestCap` | `400` | Total requests each signed-in login may make. One Tekti model call can take up to 3 requests; a full run is roughly 150 to 250. |
| `GithubTotalRequestCap` | `4000` | Total requests all signed-in logins together may make. `0` blocks every GitHub user. |
| `GithubAccessExpiryDate` | empty (no end date) | Last day (UTC, `YYYY-MM-DD`) on which GitHub sign-in and sessions work. From the next day: 403. |
| `ReservedConcurrency` | `-1` (reserve nothing) | `-1` is required when the account's total Lambda concurrency is 10. A positive number reserves that many; `0` stops all traffic. |
| `PythonRuntime` | `python3.15` | Lambda runtime. Pass `python3.14` if 3.15 is not offered in your region. |
| `SecretCacheSeconds` | `120` | How long the function keeps keys and the signing key in memory before reading them again. |

Sign-ins (`POST /auth/github`) are limited to 10 per login per UTC day and do not count against the
request caps.

### Outputs

| Output | What it is |
| :--- | :--- |
| `GatewayUrl` | the function URL, `https://<id>.lambda-url.<region>.on.aws/`. Without the trailing slash it is `TEKTI_GATEWAY_URL`. |
| `VendorKeysSecretName` / `VendorKeysSecretArn` | the secret to put your vendor keys in |
| `OwnerTokenSecretName` | the secret holding your own gateway token |
| `SessionSigningKeySecretName` | the secret whose value signs GitHub sessions |
| `FunctionName`, `CounterTableName`, `LogGroupName` | for operations |

Check that it is up (no token needed):

```bash
curl -s https://<id>.lambda-url.<region>.on.aws/health      # {"ok": true}
```

## Put your vendor keys in

Until you do, every call answers 503 `vendor_not_configured`.

1. AWS console, in the stack's region: **Secrets Manager** -> secret `<stack>/vendor-keys`.
2. **Retrieve secret value** -> **Edit** -> **Plaintext**, and replace the placeholders:
   ```json
   {"openai": "sk-...", "anthropic": "sk-ant-...", "google": "AIza..."}
   ```
   A vendor you do not want to offer can keep `REPLACE_ME`; calls to it get 503.
3. **Save**. The function picks the new value up within `SecretCacheSeconds` (2 minutes by default).

Redeploying the stack does not overwrite this: CloudFormation only rewrites the secret if the
`SecretString` in the template changes, and it must never be changed.

## Your own token

Secrets Manager -> secret `<stack>/token-owner` -> **Retrieve secret value**. Put it in your own
`.env` as `TEKTI_GATEWAY_TOKEN`; with it set, Tekti uses it instead of a GitHub session.

## Change who may sign in, the caps or the end date

Redeploy with the new values; no code change:

```bash
gateway/deploy.sh us-east-1 tekti-gateway AllowedGithubLogins=alice,bob-the-judge,carol
gateway/deploy.sh us-east-1 tekti-gateway GithubUserRequestCap=600 GithubTotalRequestCap=6000
gateway/deploy.sh us-east-1 tekti-gateway GithubAccessExpiryDate=2026-12-12
```

These are environment variables, so they apply to the next request once the update completes.

Counters never reset by themselves; the caps are totals, not daily limits. To see usage or reset a
counter, open the DynamoDB table (`CounterTableName`) -> **Explore items**: the items are `owner`,
`gh:<login>` (one per GitHub user), `gh-total` (all GitHub users) and `signin:<login>:<date>`.
Delete an item, or edit its `count`, to reset it.

## Revoke access

- **One person:** remove their login from `AllowedGithubLogins` and redeploy. Their session stops
  working on the next request, because the allowlist is checked every time. (This only works while the
  allowlist is non-empty; an empty allowlist lets everyone in.)
- **Every GitHub user, immediately:** redeploy with `GithubTotalRequestCap=0`, or with a
  `GithubAccessExpiryDate` in the past.
- **Every existing session:** give the signing key a new random value. All sessions become invalid
  within `SecretCacheSeconds`, and people on the allowlist simply run `tekti login` again.
  - Console: secret `<stack>/session-signing-key` -> **Retrieve secret value** -> **Edit** -> paste a
    new random value of at least 32 letters and digits -> **Save**.
  - Or from a shell, without the value ever being printed:
    ```bash
    aws secretsmanager put-secret-value --region <region> --secret-id <stack>/session-signing-key \
      --secret-string "$(aws secretsmanager get-random-password --region <region> \
          --password-length 64 --exclude-punctuation --query RandomPassword --output text)" \
      --query VersionId --output text
    ```
- **Your owner token:** the same command with `--secret-id <stack>/token-owner` and
  `--password-length 40`.
- **Everything at once:** redeploy with `ReservedConcurrency=0`.

## What the gateway does with a request

`GET /health` answers `{"ok": true}`. `POST /auth/github` signs a GitHub user in (above). Every other
request must be `POST` to one of the model routes in CONTRACT.md (`/openai/...`, `/anthropic/...`,
`/google/...`), otherwise 404 (unknown path) or 405 (wrong method). Then, in order:

1. Credential: taken from `Authorization: Bearer`, `x-api-key`, `x-goog-api-key` or `?key=`; if it
   appears in several places it must be the same.
   - A value starting with `tks1.` is a GitHub session. Its HMAC-SHA256 signature is checked in
     constant time. A bad one gives 401 `invalid_token`; an expired one gives 401 `session_expired`.
   - Anything else is compared in constant time with the owner token, and is 401 if it does not match.
2. GitHub users only: login not on a non-empty `AllowedGithubLogins` gives 403 `not_on_allowlist`;
   past `GithubAccessExpiryDate` gives 403 `token_expired`.
3. Body over 1 MB: 413.
4. Model (JSON `model`, or the model in the Google path) not in `AllowedModels`: 403.
5. Vendor key still `REPLACE_ME`: 503.
6. Counters, checked before anything is forwarded:
   - Owner: one counter, incremented atomically.
   - GitHub user: the per-login counter and the all-users counter are incremented in a single
     DynamoDB transaction. Either both move or neither does.
   - At a cap: 429 `request_cap_reached` (per login, or the owner) or `total_cap_reached`.
7. Forwarded with the vendor key in the vendor's own header (`Authorization: Bearer` for OpenAI and
   Google's OpenAI-compatible route, `x-api-key` for Anthropic, `x-goog-api-key` for Google).
   - Only these request headers are forwarded: `content-type`, `accept`, `user-agent`,
     `anthropic-version`, `anthropic-beta`, `openai-beta`, `idempotency-key`, `x-goog-api-client`,
     `x-stainless-*`. In particular, `openai-organization` and `openai-project` are not.
   - The query string is kept, except `key`.
8. The vendor's status, body and headers come back unchanged (except hop-by-hop headers,
   `set-cookie` and `openai-organization`). Server-sent events are buffered and returned whole.

Gateway errors look like
`{"error": {"type": "tekti_gateway", "code": "<code>", "message": "<text>"}}`. Gateway 4xx answers
carry `x-should-retry: false`, so the vendor SDKs do not resend them.

Codes:
- `not_found`, `method_not_allowed`
- `missing_token`, `invalid_token`, `session_expired`, `token_expired`, `not_on_allowlist`
- `github_token_rejected` (401), `github_unavailable` (502), `signin_limit_reached` (429)
- `body_too_large`, `invalid_body`, `model_not_allowed`, `vendor_not_configured`
- `request_cap_reached`, `total_cap_reached`, `gateway_busy` (503)
- `upstream_timeout` (504), `upstream_unreachable` (502), `upstream_response_too_large` (502; the
  function URL cannot return more than 6 MB)
- `gateway_misconfigured`, `gateway_unavailable`, `internal_error` (500)

## Logs

One JSON line per request in `/aws/lambda/<stack>-gateway`, for example:

```json
{"duration_ms":2310,"model":"claude-sonnet-4-5","request_id":"...","route":"/v1/messages","status":200,
 "token":"gh:alice","ts":"2026-11-26T10:00:00.000+00:00","upstream_status":200,"usage":{"input_tokens":812,"output_tokens":95},"vendor":"anthropic"}
```

`token` is `owner` or `gh:<login>`. No token, session, GitHub token, key, header, query string,
prompt or response is logged. A CloudWatch Logs Insights query for usage per person:

```
fields token, vendor, model, status
| filter ispresent(status)
| stats count() as requests by token, vendor, model, status
```

## Tear down

```bash
aws cloudformation delete-stack --region <region> --stack-name <stack>
aws cloudformation wait stack-delete-complete --region <region> --stack-name <stack>
# then, if deploy.sh created it:
aws s3 rb --region <region> --force s3://<stack>-artifacts-<account-id>-<region>
```

This removes the function, URL, secrets, table and log group. Secrets Manager may keep deleted secrets
in a recovery window; if a new stack with the same name then fails because a secret name is still
taken, wait for the window to pass or use another stack name.

## Known limits

- **The URL is public.** Anyone on the internet can reach the function URL. There is no WAF and no
  per-IP rate limiting. The checks in code, the caps and the concurrency limit are the only
  protections.
  - Requests with bad tokens still invoke (and bill) the function, though they never reach a vendor.
  - `POST /auth/github` makes one call to GitHub per attempt, even for a bogus token.
- **An empty `AllowedGithubLogins` admits every GitHub account.** Then only `GithubTotalRequestCap`
  limits the cost. Keep the list filled in.
- **A session is a bearer credential for up to 24 hours.** Removing the login from the allowlist,
  rotating the signing key, or setting the total cap to 0 ends it sooner.
- **Logins can be renamed on GitHub.** The allowlist matches the login, not GitHub's numeric id. If
  someone renames their account and another person takes over the freed login, that person passes
  the allowlist.
- **The caps count requests, not tokens or money.** Keep `AllowedModels` short, and set spending
  limits in the vendors' own consoles as well.
- **Responses are buffered.** Streaming clients get the whole event stream at once. A single answer
  is limited to 6 MB and to the 300-second function timeout.
- **Key and signing-key rotations take effect after up to `SecretCacheSeconds`.**

## Tests

```bash
uv run pytest tests/gateway
```

The tests run offline: AWS clients and outbound HTTP are replaced by fakes. One end-to-end test runs
the real HTTP client against a local server standing in for the vendors and for GitHub. The upstream
base URLs can be overridden with `UPSTREAM_OPENAI_URL`, `UPSTREAM_ANTHROPIC_URL`,
`UPSTREAM_GOOGLE_URL` and `GITHUB_API_URL`; the template never sets them.
