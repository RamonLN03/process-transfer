#!/bin/sh
# Entry point of the image (docs/docker.md): generate one data set with the pipeline of
# the repository, and refuse to start when nothing is mounted on PT_DATA_DIR.
#
# What the refusal catches. With nothing mounted there, the run would write into the
# container's own writable layer, which `docker rm`, or the --rm of the run itself,
# deletes with the container; the data set would be lost without a word, so the run stops
# before generating anything.
#
# What it does not catch. The check only asks whether PT_DATA_DIR is a mount point. It
# cannot tell a mount that keeps the results from one that does not, and it is not a
# guarantee that they are kept. A folder of the host or a named volume keeps them. A tmpfs
# mount lives in memory and ends with the container, and an anonymous volume (-v /data)
# is deleted with a container run with --rm: both pass the check, and both lose the
# results. Codex reproduced both cases in its audit of 2026-09-24.
#
# Exit codes: those of `python -m process_transfer.generation` (0 every mandatory check
# passed, 1 a check failed or an error ended the run, 2 no definition file or wrong
# arguments), and 3 when nothing is mounted on PT_DATA_DIR.
set -eu

case "${1-}" in
    -h | --help)
        exec python -m process_transfer.generation --help
        ;;
esac

if ! mountpoint -q "$PT_DATA_DIR"; then
    echo "entrypoint: nothing is mounted on PT_DATA_DIR=$PT_DATA_DIR, so nothing was generated." >&2
    echo "entrypoint: to keep the results, mount a folder of the host, -v \"<host folder>:$PT_DATA_DIR\", or a named volume, -v <volume name>:$PT_DATA_DIR." >&2
    echo "entrypoint: a tmpfs mount, or an anonymous volume with --rm, would pass this check and still lose the results (docs/docker.md)." >&2
    exit 3
fi

exec python -m process_transfer.generation "$@"
