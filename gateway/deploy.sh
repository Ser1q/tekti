#!/usr/bin/env bash
# Deploy (or update) the Tekti gateway stack.
#
#   gateway/deploy.sh <region> <stack-name> [--bucket <artifact-bucket>] [--dry-run] [ParamKey=Value ...]
#
# --dry-run creates the change set but does not execute it, so you can review it first.
#
# Examples:
#   gateway/deploy.sh eu-central-1 tekti-gateway
#   gateway/deploy.sh eu-central-1 tekti-gateway AllowedModels=gpt-5,claude-sonnet-4-5 GithubAccessExpiryDate=2026-12-20
#
# Parameters you do not pass keep their current value on an update (defaults on a first deploy).
# The code is a single file (handler.py, standard library + the runtime's boto3), so there is no
# build step: it is staged into a temporary directory, zipped and uploaded by
# `aws cloudformation package`, then `aws cloudformation deploy` creates or updates the stack.
#
# Without --bucket, the script uses an artifact bucket named
# <stack-name>-artifacts-<account-id>-<region> and creates it (private, encrypted, TLS only) if it
# does not exist. That bucket is not part of the stack; delete it yourself after tearing down.
#
# This script never reads or prints a secret value.
set -euo pipefail

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[[ $# -ge 2 ]] || usage
REGION="$1"
STACK="$2"
shift 2

BUCKET=""
DRY_RUN=0
OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      [[ $# -ge 2 ]] || usage
      BUCKET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help) usage ;;
    *=*)
      OVERRIDES+=("$1")
      shift
      ;;
    *) usage ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text --region "$REGION")"

if [[ -z "$BUCKET" ]]; then
  BUCKET="$(printf '%s-artifacts-%s-%s' "$STACK" "$ACCOUNT" "$REGION" | tr '[:upper:]' '[:lower:]')"
fi

if ! aws s3api head-bucket --bucket "$BUCKET" --expected-bucket-owner "$ACCOUNT" --region "$REGION" >/dev/null 2>&1; then
  echo "Creating artifact bucket $BUCKET in $REGION"
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --region "$REGION" \
    --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "$BUCKET" --region "$REGION" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-policy --bucket "$BUCKET" --region "$REGION" --policy "$(cat <<JSON
{"Version":"2012-10-17","Statement":[{"Sid":"DenyInsecureTransport","Effect":"Deny","Principal":"*",
"Action":"s3:*","Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"],
"Condition":{"Bool":{"aws:SecureTransport":"false"}}}]}
JSON
)"
  aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --region "$REGION" \
    --lifecycle-configuration \
    '{"Rules":[{"ID":"expire-old-artifacts","Status":"Enabled","Filter":{},"Expiration":{"Days":90}}]}'
fi

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD/function"
cp "$HERE/template.yaml" "$BUILD/template.yaml"
cp -p "$HERE/handler.py" "$BUILD/function/handler.py"  # -p keeps mtime so the zip hash is stable

aws cloudformation package \
  --region "$REGION" \
  --template-file "$BUILD/template.yaml" \
  --s3-bucket "$BUCKET" \
  --s3-prefix "$STACK" \
  --output-template-file "$BUILD/packaged.yaml" >/dev/null

DEPLOY_ARGS=(
  --region "$REGION"
  --stack-name "$STACK"
  --template-file "$BUILD/packaged.yaml"
  --capabilities CAPABILITY_IAM
  --no-fail-on-empty-changeset
)
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  DEPLOY_ARGS+=(--parameter-overrides "${OVERRIDES[@]}")
fi
if [[ $DRY_RUN -eq 1 ]]; then
  DEPLOY_ARGS+=(--no-execute-changeset)
fi
aws cloudformation deploy "${DEPLOY_ARGS[@]}"
[[ $DRY_RUN -eq 1 ]] && exit 0

echo
echo "Stack outputs (names and ARNs only; secret values stay in Secrets Manager):"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table
