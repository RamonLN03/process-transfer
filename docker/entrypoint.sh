#!/bin/sh
# Entry point of the image (docs/docker.md): generate one data set with the pipeline of
# the repository, and refuse to start unless PT_DATA_DIR is a directory mounted from
# outside the container.
#
# Why the refusal. Files written anywhere else live in the container's own writable
# layer and are deleted with it, by `docker rm` or by the --rm of the run itself. A data
# set generated there would be lost without a word, so the run stops before generating
# anything. A folder of the host or a Docker volume mounted on PT_DATA_DIR survives the
# container.
#
# Exit codes: those of `python -m process_transfer.generation` (0 every mandatory check
# passed, 1 a check failed or an error ended the run, 2 no definition file or wrong
# arguments), and 3 when PT_DATA_DIR is not a mounted directory.
set -eu

case "${1-}" in
    -h | --help)
        exec python -m process_transfer.generation --help
        ;;
esac

if ! mountpoint -q "$PT_DATA_DIR"; then
    echo "entrypoint: PT_DATA_DIR=$PT_DATA_DIR is not a mounted directory, so nothing was generated." >&2
    echo "entrypoint: mount a folder of the host there, for example -v \"<host folder>:$PT_DATA_DIR\"." >&2
    echo "entrypoint: whatever a run writes inside the container is lost when the container is removed." >&2
    exit 3
fi

exec python -m process_transfer.generation "$@"
