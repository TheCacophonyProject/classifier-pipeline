#!/bin/bash
hub_repo="cacophonyproject/classifier"
docker build -t $hub_repo .

docker tag $hub_repo $hub_repo:$TRAVIS_TAG

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker push $hub_repo:$TRAVIS_TAG
