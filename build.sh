#!/bin/bash
# build-and-push.sh

# Variables
REGISTRY="gitlab-registry.dev.anansecloud.com"
PROJECT="ml"
IMAGE="smoke-test"
TAG="${1:-latest}"
FULL_IMAGE="${REGISTRY}/${PROJECT}/${IMAGE}:${TAG}"

# 1. Build the image
echo "Building ${FULL_IMAGE}..."
docker build -t ${FULL_IMAGE} .

# 2. Tag with additional version (optional)
docker tag ${FULL_IMAGE} ${REGISTRY}/${PROJECT}/${IMAGE}:$(date +%Y%m%d-%H%M%S)

# 3. Login to registry
echo "Logging into ${REGISTRY}..."
# docker login ${REGISTRY}

# 4. Push the image
echo "Pushing ${FULL_IMAGE}..."
docker push ${FULL_IMAGE}

# 5. Push additional tags
docker push ${REGISTRY}/${PROJECT}/${IMAGE}:$(date +%Y%m%d-%H%M%S)

echo "✅ Push complete!"