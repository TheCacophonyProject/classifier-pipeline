#!/bin/bash
hub_repo="cacophonyproject/classifier"
TRAVIS_TAG="0.1"
DOCKER_USERNAME="giampaolof"
DOCKER_PASSWORD="%8!%*scU7sVe_;^"
docker build -t $hub_repo .

docker tag $hub_repo $hub_repo:$TRAVIS_TAG

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker push $hub_repo:$TRAVIS_TAG
