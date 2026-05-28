#!/usr/bin/env bash
set -euo pipefail

: "${ARTIFACT_BUCKET:?ARTIFACT_BUCKET is required}"
: "${ARTIFACT_PREFIX:=rag-fine-tuning}"
: "${AWS_REGION:=us-east-1}"

aws s3 sync "s3://${ARTIFACT_BUCKET}/${ARTIFACT_PREFIX}/artifacts" artifacts --region "${AWS_REGION}"
aws s3 sync "s3://${ARTIFACT_BUCKET}/${ARTIFACT_PREFIX}/outputs" outputs --region "${AWS_REGION}"
