#!/bin/bash
python3 -m build

set -xe
TAG=$TRAVIS_TAG

sed -i "s/NEWVERSIONHERE/${TAG}/g" pyproject.toml

hub_repo="cacophonyproject/classifier"
docker build -t $hub_repo . -f docker/DockerFile

docker tag $hub_repo $hub_repo:$TRAVIS_TAG

echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker push $hub_repo:$TRAVIS_TAG
