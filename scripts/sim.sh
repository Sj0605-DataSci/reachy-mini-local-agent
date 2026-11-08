#!/usr/bin/env bash
# Reachy Mini MuJoCo sim daemon on :8000. Add --headless when there is no display (use viewer/web page instead).
exec ${PYTHON:-python} -m reachy_mini.daemon.app.main --sim "$@"
