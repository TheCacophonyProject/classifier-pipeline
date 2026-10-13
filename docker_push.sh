#!/bin/bash
set -xe
hub_repo="cacophonyproject/classifier"
docker login
docker build -t $hub_repo . -f docker/Dockerfile

docker tag $hub_repo $hub_repo:v2.45.3

docker push $hub_repo:v2.45.3
docker push $hub_repo:latest
