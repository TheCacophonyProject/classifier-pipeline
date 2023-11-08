#!/bin/bash
set -xe
hub_repo="cacophonyproject/classifier"
docker build -t $hub_repo . -f docker/Dockerfile

docker tag $hub_repo $hub_repo:$TRAVIS_TAG

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker push $hub_repo:$TRAVIS_TAG
docker push $hub_repo:latest
