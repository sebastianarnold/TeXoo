#!/bin/sh
docker run --oom-kill-disable -m 50G -v "$(pwd)":/usr/src -it texoo /bin/bash
