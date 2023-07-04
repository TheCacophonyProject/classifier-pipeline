#!/bin/bash

docker run -v /tmp/cacophony:/tmp/cacophony --name classifier -d classifier-docker /usr/bin/supervisord
