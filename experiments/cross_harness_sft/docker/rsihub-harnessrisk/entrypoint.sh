#!/bin/sh
set -eu

if [ "$#" -eq 0 ]; then
  set -- evolve --help
fi

case "$1" in
  adapt-receipt)
    shift
    exec python /opt/rsihub_harnessrisk_adapter.py "$@" --rsihub-src "$RSIHUB_SRC"
    ;;
  download-model)
    shift
    exec python /opt/download_hsrl_models.py "$@"
    ;;
  versions)
    cat /opt/rsihub_harnessrisk_source_pins.json
    ;;
  *)
    exec "$@"
    ;;
esac
