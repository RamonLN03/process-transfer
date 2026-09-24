# syntax=docker/dockerfile:1
#
# The data path of M0 in a Linux container: the virtual plants, the generator, Parquet,
# DuckDB, the SQL checks and the export, run from the files of this repository exactly as
# a checkout runs them. docs/docker.md explains the choices and how to build and run it.

# Debian 13 with CPython 3.13.7, the interpreter of the registered environment of M0. The
# tag says what the image is; the digest fixes its bytes, so a rebuild starts from the same
# base even if the tag is later moved.
FROM python:3.13.7-slim-trixie@sha256:5f55cdf0c5d9dc1a415637a5ccc4a9e18663ad203673173b8cda8f8dcacef689

# PT_DATA_DIR is where every generated file goes, and the entry point refuses to run
# unless a directory from outside the container is mounted there. MPLCONFIGDIR gives
# matplotlib a cache it can write as any user. Python writes no bytecode at run time, the
# code being read-only for the user that runs it, and pip keeps no cache in the image.
ENV PT_DATA_DIR=/data \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# The dependencies first. They change rarely, so this layer stays in the build cache while
# the code changes. pip installs exactly the wheels of the lock, verified by their hashes,
# and resolves nothing; pip check confirms that together they are consistent.
COPY docker/requirements.lock.txt docker/requirements.lock.txt
RUN pip install --require-hashes --no-deps --requirement docker/requirements.lock.txt \
    && pip check

# The project: the parts of the checkout that the data path reads, and nothing else
# (.dockerignore). Installed in editable mode, as in development and in CI, so that the
# code finds the SQL and the configurations next to pyproject.toml. setuptools comes from
# the lock, so the install needs nothing from the network.
COPY pyproject.toml README.md ./
COPY src/ src/
COPY sql/ sql/
COPY configs/ configs/
COPY experiments/ experiments/
COPY docker/entrypoint.sh docker/entrypoint.sh
RUN pip install --no-deps --no-build-isolation --editable . \
    && pip check

# A user without administrator rights runs everything from here on. The code belongs to
# root and this user cannot change it; the mount point of PT_DATA_DIR belongs to it.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin pt \
    && mkdir "$PT_DATA_DIR" \
    && chown pt "$PT_DATA_DIR"
USER pt

ENTRYPOINT ["/bin/sh", "/app/docker/entrypoint.sh"]
CMD ["--help"]
