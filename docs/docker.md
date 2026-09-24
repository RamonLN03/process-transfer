# The data path in a Linux container

The image built from `Dockerfile` runs the data path of M0, `python -m process_transfer.generation`, from the files of this repository, on Linux and on the CPU. It is the same code that a checkout runs: the virtual plants, their verified starting points, the acceptance checks of the truth, the sensors, Parquet, DuckDB, the SQL quality queries, the export and the scan for hidden information. Nothing of M1 is in it yet. The decision is D-031.

The image contains the simulator and the configuration files of the true plants, hidden physics included, exactly as the checkout does. It belongs to the truth side of the project, like the generator itself. What a model may read is still only the available branch that a run writes under `PT_DATA_DIR/available/`, never the image.

The image is built locally and is not published anywhere.

## The files

| File | What it does |
|---|---|
| `Dockerfile` | the recipe of the image: the base image, the dependencies, the project, the user and the entry point |
| `.dockerignore` | which files of the repository the build may see: everything is left out and only what the image needs is let back in |
| `docker/requirements.lock.txt` | the Python packages of the image, each pinned to one version and to the hash of one wheel |
| `docker/entrypoint.sh` | what a container runs: it refuses to start when nothing is mounted on `PT_DATA_DIR`, then runs the generator; the check does not guarantee that the results are kept (see "Keeping the results") |

## Why it is built this way

* Base image: `python:3.13.7-slim-trixie`, pinned by its digest. CPython 3.13.7 is the interpreter of the registered environment of M0 (`docs/reference_environment.md`). The slim Debian images have the C library, glibc, for which every package of the lock publishes a ready wheel, so nothing is compiled. Alpine uses another one, musl, and PyPI has no wheel of DuckDB 1.5.5 for it (checked on 2026-09-24), so DuckDB would have to be built from source. Trixie is the current Debian release. The tag says what the image is; the digest fixes its bytes, so a rebuild starts from the same base even if the tag is moved later.
* Dependencies: `docker/requirements.lock.txt`, not `requirements-reference.lock.txt`. The reference lock describes a Windows environment and holds tools the image does not need (pytest, ruff) and colorama, which only Windows uses; it stays as it is. The container's lock has the same versions of every runtime package, adds setuptools for the install of the project, and pins the hash of the Linux wheel of each package. The Dockerfile installs it with `--require-hashes --no-deps`: exactly those files, nothing resolved at build time, and `pip check` confirms they fit together.
* The project is installed from the checkout, in editable mode, as in development and in CI. The database reads its SQL from `sql/` next to `pyproject.toml`, and definitions are found under `configs/`, so the image needs those directories where the code expects them. Installing the package as a wheel, without the checkout, has not been built or tested, and nothing here claims it works.
* `PT_DATA_DIR` is `/data` inside the container. With nothing mounted there, a run would write into the container's own layer and lose everything with it, so the entry point refuses that case (exit code 3). The check only asks whether something is mounted on `/data`: it guards against a forgotten mount, and it is not a guarantee that the results are kept, since a mount that loses them passes it too (see "Keeping the results"). A `VOLUME` instruction in the Dockerfile was set aside: it would give every run an anonymous volume, which silences the check, and `docker run --rm` deletes that volume with the container.
* A user without administrator rights, `pt` (UID 1000), runs everything. The code belongs to root and `pt` cannot change it; `/data` belongs to `pt`. On Windows, Docker Desktop lets the container write to any mounted folder. On Linux the mounted folder must be writable by UID 1000, which is what the CI job arranges with `chmod`.
* There is no git in the image and `.git` is not in the build context. See "Provenance" below.

## Build the image

From the root of the repository, in PowerShell:

```powershell
docker build -t process-transfer:local .
```

`docker build` reads `Dockerfile` and produces an image. `-t process-transfer:local` names it: `process-transfer` is the repository of images on this machine and `local` the tag, which says that it was built here and never pulled or pushed. The final `.` is the build context, the directory whose files the build may copy, filtered by `.dockerignore`.

The first build downloads the base image and the wheels and takes one or two minutes. Every instruction of the Dockerfile produces a layer, and a layer is reused from the cache when neither its instruction nor the files it copies have changed. The dependencies are installed before the code is copied, so a change to the code rebuilds only the last layers, in a few seconds; a change to the lock rebuilds the dependencies as well. Rebuild whenever the code, the configurations or the lock change: a container runs the files that were copied into its image, not the files of the checkout.

## Run a data set

```powershell
$out = "$env:USERPROFILE\pt-docker\run-1"
New-Item -ItemType Directory -Force $out | Out-Null
docker run --rm -v "${out}:/data" process-transfer:local configs/datasets/m0_e06.yaml
$LASTEXITCODE
```

* `$out` is a new folder of the host for the results. Keep it outside OneDrive and outside the repository: a DuckDB file that a synchronisation client touches while it is open can be damaged.
* `docker run` creates a container from the image and starts it; `--rm` removes the container when it ends. The results are not in the container, so nothing is lost with it.
* `-v "${out}:/data"` is a bind mount: the folder of the host appears at `/data` inside the container, which is `PT_DATA_DIR`. What the run writes there is written to the host.
* `configs/datasets/m0_e06.yaml` is passed to the generator. It is a path inside the image, relative to `/app`, where the configurations were copied when the image was built.
* `$LASTEXITCODE` is the exit code of the container, which is the exit code of the generator.

What a run leaves in `$out`:

```
available\datasets\<dataset_id>\       manifest.json and the Parquet files of the data set
available\databases\<dataset_id>.duckdb
available\exports\<dataset_id>\        export.json and the aligned series
private\datasets\<dataset_id>\<attempt>\   provenance.json, pipeline_report.json, figures, the configurations used
```

Running the same definition again into the same folder changes nothing of the data set or of the export and reports them as already present; DuckDB may still grow its file, as recorded in M0. A new attempt directory is added under `private/`. Running it into a folder that holds a different data set of the same name, the one registered on Windows for instance, is refused as a conflict and nothing is overwritten.

## Keeping the results

What keeps the results is where `/data` is mounted from, not the check of the entry point, which only refuses a run with nothing mounted there.

Keep them in one of these:

* A folder of the host, a bind mount, as in the commands above: `-v "${out}:/data"`. The files are on the disk of the host and nothing Docker does to the container removes them. On Windows, Docker Desktop lets the container write to any folder. On a Linux host the folder must be writable by UID 1000, the user of the image: make it so with `chmod a+rwx`, as the CI job does, or give it to that UID. Without that, the run ends with exit code 1 and `PermissionError`.
* A named volume, a storage area that Docker manages and keeps until it is removed by name:

  ```powershell
  docker run --rm -v pt-data:/data process-transfer:local configs/datasets/m0_e06.yaml
  docker volume ls
  docker run --rm -v pt-data:/data -v "${out}:/out" --entrypoint cp process-transfer:local -r /data/available /out/
  ```

  The first command creates the volume `pt-data` if it does not exist; `--rm` removes the container and not the volume. The volume lives in Docker's storage, inside the Linux machine of Docker Desktop, not in a folder of Windows, so the third command copies the results out, with a throwaway container that mounts the volume and a folder of the host. A new volume takes the owner of `/data` in the image, UID 1000, the first time it is mounted, so the user of the image can write to it with no change of permissions; a volume created earlier by another container keeps the owner it was given. `docker volume rm pt-data` deletes the volume and everything in it.

Do not use these for results that should be kept, although both pass the check of the entry point:

* A tmpfs mount (`--tmpfs /data`, or `--mount type=tmpfs,destination=/data`): it is a file system in memory, and it ends with the container.
* An anonymous volume (`-v /data`, a volume without a name): with `--rm` Docker deletes it when the container ends, after the run has finished with exit code 0. Without `--rm` it outlives the container under a generated name, until `docker rm -v` or `docker volume prune` removes it, and it is easy to lose track of.

Codex reproduced both in its audit of 2026-09-24. On a tmpfs mount the pipeline ran, and its results were temporary. On an anonymous volume with `--rm` the run generated the data set and ended with exit code 0; Docker then removed the volume, and Codex verified that it was gone.

## Exit codes, and where to read errors

| Code | Meaning |
|---|---|
| 0 | every mandatory check of the generator passed |
| 1 | a mandatory check failed, or an error ended the run: the traceback is on the screen |
| 2 | the definition file does not exist, or the arguments are wrong |
| 3 | nothing is mounted on `PT_DATA_DIR`; nothing was generated |
| 125, 126, 127 | Docker itself failed before the program ran: no such image, a wrong option, a command that cannot be run or found |

The container writes its messages to the screen of `docker run`. After a run, the report of the generator, with every check and the time of every stage, is `pipeline_report.json` in the attempt directory under `private/`. Without `--rm`, the ended container stays: `docker ps -a` shows its exit status and `docker logs <name>` repeats what it printed.

Two habits of Windows PowerShell 5.1 are worth knowing. Lines that a program writes to its error stream appear in red as `NativeCommandError` when they are redirected with `2>&1`; that is PowerShell's wrapping of the program's own message, and `$LASTEXITCODE` still has the real code. And piping `docker run` into `Select-Object -First` stops the Docker client early, so its exit code is lost; filter the output after the run instead. PowerShell 5.1 also mangles double quotes nested inside an argument to an external program, so keep such arguments free of inner quotes.

## A definition that is not in the image

The image holds the configurations of the build. To run a definition written since, mount the configurations of the checkout over them, read-only:

```powershell
docker run --rm -v "${out}:/data" -v "${PWD}\configs:/app/configs:ro" process-transfer:local configs/datasets/<definition>.yaml
```

`:ro` makes the mount read-only, so the container can read the definitions and cannot change them. The whole folder is mounted, not a single file, because a definition names its plants and sensors by paths relative to itself, such as `../source_cstr.yaml`.

## An experiment script

The scripts of `experiments/` are in the image too. Experiments 05 to 07 run the generator and add the checks registered for their data sets; experiments 00 to 04 and 08 generate nothing and write only diagnostics.

```powershell
docker run --rm --entrypoint python -v "${out}:/data" process-transfer:local experiments/06_steady_operation.py
```

`--entrypoint python` replaces the entry point of the image for this run, so the check of the mount does not happen: mount a folder of the host or a named volume on `/data` yourself ("Keeping the results").

## Removing containers and images

```powershell
docker ps -a
docker rm <container name>
docker image ls process-transfer
docker image rm process-transfer:local
docker volume ls
docker volume rm <volume name>
```

Removing a container or an image does not touch the folders of the host that were mounted, nor named volumes. `docker volume rm` deletes a volume with everything in it, and `docker rm -v` deletes the anonymous volumes of the container it removes.

## Provenance inside the container

Neither `.git` nor git is in the image, so every run records that the code could not be identified: in `provenance.json` the git block says "the git executable was not found", the commit is null and `code_identified` is false, the manifest of the data set says the same, and attempt directories end in `_nogit`. The environment is still recorded: Linux, CPython and the version of every package.

No commit is written into the results because nothing inside the container can verify one. The build context may hold changes that were never committed, and a commit passed in from outside would be a claim, not a check. To tie a run to the code, build from a clean checkout and keep the commit, and the ID of the image (`docker image inspect process-transfer:local --format '{{.Id}}'`), next to the output folder. That ID names one build: with the image store of Docker Desktop it covers the attestation that each build writes, so two builds of the same files get different IDs. Whether two images hold the same files is answered by their layers, `docker image inspect <image> --format '{{json .RootFS.Layers}}'`, which were equal across two such builds on 2026-09-25. Putting `.git` and git into the image was set aside: the files in the image differ on purpose from the commit (`.dockerignore`), so git would report the tree as changed, and its ownership check refuses a repository owned by another user.

## Reproducibility: four claims kept apart

1. The environment can be rebuilt. The base image is pinned by digest and every package by version and wheel hash, and nothing is resolved at build time, so a rebuild on linux/amd64 installs the same interpreter and the same files, or fails. On another architecture the lock has no wheels and the build fails; that was not tried.
2. Results repeat inside the container environment. The definition of M0-E06, generated twice into two new folders, gave the same content hashes, before and after the image was rebuilt with a change that did not touch the data.
3. The numbers agree with the registered environment on Windows, with the same versions of every library. For M0-E06 the source run was identical bit for bit; the readings of the target run differed in 1580 of 2402 values, by at most 3.7e-10 mol/m^3 and 2.6e-11 K. For M0-E05 every input was identical and the readings of all six runs differed by at most 2.6e-6 mol/m^3 and 4.3e-7 K, of the order of the tolerance of the integrator and six orders of magnitude below the noise of the sensors. The registered checks of both experiments held in the container.
4. Content hashes are not equal across platforms. Of the eight runs compared, one, the steady run of the source, had the hash registered on Windows; the other seven did not. A data set generated in the container is therefore other content than the registered one, and the policy on repetition refuses to put it in place of the registered data set.

## What was run on 2026-09-24

On Docker Desktop 4.92.0 (engine 29.8.0, WSL 2, linux/amd64), into new folders outside the repository and outside OneDrive:

| Check | Result |
|---|---|
| build from the repository root | 73 s the first time, the base image already pulled; 8 s after a change to the code, the dependency layer taken from the cache |
| content of `/app` | only the files `.dockerignore` lets in; no bytecode, data or build metadata of the host |
| packages | exactly the lock, plus pip of the base image and the project in editable mode; `pip check` clean |
| user | `pt`, UID 1000; the code not writable by it, `/data` writable |
| `configs/datasets/m0_e06.yaml` | exit 0, the ten checks of the generator pass; Parquet, DuckDB, export and private record written to the host |
| the container removed with `docker rm` | every file still in the folder of the host |
| the same definition again, same folder | exit 0, data set and export reported as already present and not touched |
| without a mount | exit 3, nothing generated |
| a definition that does not exist | exit 2 |
| the folder mounted read-only | exit 1, `OSError: Read-only file system`, nothing written |
| the configurations of the checkout mounted read-only over `/app/configs` | exit 0; a write there refused with `Read-only file system`; nothing in the checkout changed |
| the data set registered on Windows already in the folder | exit 1, `DatasetConflictError`, the registered copy untouched |
| `experiments/06_steady_operation.py` | exit 0, the five hypotheses of M0-E06 hold |
| `experiments/05_full_data_path.py` | exit 0, the nine hypotheses of M0-E05 hold, 63 s against about 16 s natively on Windows |
| a fresh `git clone` of the commit, outside OneDrive, built and run | exit 0 with the same definition; the files in `/app` identical to those of the image built from the working tree |
| a Linux folder owned by root, mode 755, mounted as on a CI runner (inside the Linux machine of Docker Desktop) | exit 1, `PermissionError`; after `chmod a+rwx`, as the CI job does, exit 0 and the files owned by UID 1000 |
| the test suite, run once in the image with pytest added to a throwaway container | 727 tests: 722 pass; 4 skipped, those that need git; 1 fails by design, `test_paths` expecting `AGENTS.md` next to `pyproject.toml`, which the image leaves out; with `AGENTS.md` mounted as a fixture it passes, as Codex's audit found (below) |

The first run in the container showed a defect of the figures of M0: they asked matplotlib for Segoe UI, a font Linux does not have, and matplotlib logged 1005 warnings for the figures of one data set, burying the output of the run. The figures now use the first font of a short list that the machine has, and a regression test covers it.

Folders of Windows reach the container through a network file system, 9p, which is slower than a native disk for many small files; that is why experiment 05 took four times as long.

The cache is keyed by what is copied, file metadata included. Folders of this repository that carry the read-only attribute of Windows arrive in the image without their write bit, so a build from the working tree and one from a fresh clone hold the same files but do not share the layers from the copy of `src/` on.

## Independent audit of `01c6422` by Codex (2026-09-24)

Codex audited the repository at `01c6422` on 2026-09-24, independently and without modifying it. What follows is its report, recorded here; these checks were not repeated in the iteration that recorded them. Commits after `01c6422` are not covered by this audit.

| Check | Result reported by Codex |
|---|---|
| test suite in the reference environment on Windows | 727 tests pass |
| test suite in the container | 723 pass, 4 skipped because git is absent. `tests/` and `AGENTS.md` were mounted read-only as fixtures and pytest was added to a throwaway container; the failure reported earlier for the missing `AGENTS.md` was not a failure of the data path |
| ruff and `git diff --check` | clean |
| image built with the cache and with `--no-cache` | both correct |
| dependencies | verified by their hashes and by `pip check` |
| files of `src/`, `configs/`, `sql/` and `experiments/` | 69 files identical byte for byte between the image and the checkout |
| M0-E05, M0-E06, M0-E07 and M0-E08 | every criterion holds on Windows and in the container; the runs of these experiments in the container were made without network |
| user | the code runs as `pt`, UID 1000, with no write access to the code |
| help and exit codes | as documented |
| persistence in a folder of the host, and a repetition of M0-E06 | the data set and the export kept their content and their modification times |
| configurations mounted read-only | the run is correct |

Conservation of M0. Codex rebuilt `91206b2`, the audited reference of M0, temporarily, generated its data sets in the reference environment on Windows, and compared them with those of the current code in the same environment: every content hash was equal, 6 of 6 runs of M0-E05, 2 of 2 of M0-E06 and 16 of 16 of M0-E07, 24 of 24. Since `91206b2` the only change to the production code of M0 is the choice of font in `generation/figures.py`; equations, parameters, protocols, integration, SQL and data contracts are unchanged.

Windows against the container. The inputs were identical, and the largest differences of the readings were:

| Data set | C_A, mol/m^3 | T, K |
|---|---|---|
| M0-E05 | 2.5831683103660907e-6 | 4.25626126343559e-7 |
| M0-E06 | 3.6646952139562927e-10 | 2.5920599000528455e-11 |
| M0-E07 | 1.2840075669373618e-6 | 1.4216107047104742e-7 |

That is numerical agreement between the platforms, not a general equality of content hashes.

What the audit also recorded:

* The check of the entry point is not a guarantee that results are kept: a tmpfs mount, and an anonymous volume with `--rm`, pass it and lose the results. The text of the entry point, of the Dockerfile, of this file and of D-031 was corrected on 2026-09-25 (see "Keeping the results"); the check and its exit code did not change.
* GitHub Actions had not been verified by the audit.
* Runs in the container record the commit as unknown, and no provenance is invented for them.
* The audit modified nothing in the repository and did not reopen M0.

## Checked on 2026-09-25

After the correction of the persistence claims, which changed the comments and messages of `docker/entrypoint.sh`, a comment of the `Dockerfile` and the documentation, and nothing of the check, of the exit codes or of the data path, an image built from the corrected files was run again on Docker Desktop:

| Check | Result |
|---|---|
| `--help` with nothing mounted | exit 0 |
| a definition with nothing mounted | exit 3, with the new message |
| a named volume that did not exist, `-v pt-check-volume:/data` | exit 0; the volume still there after `--rm`; its root owned by UID 1000 with no change of permissions; the results copied out to a folder of Windows with a throwaway container |
| the same definition again into that volume | exit 0, the data set reported as already present |
| the image rebuilt from the same files | all 12 steps from the cache and the same layers; a different image ID, since the ID covers the attestation of each build |
| `sh -n` on the entry point, in the image | no syntax error |

## Continuous integration

The `docker` job of `.github/workflows/ci.yml` builds the image on every push, generates the data set of M0-E06 into a folder of the runner, checks that the data set, the database and the export are there after the container has ended, and checks the exit codes of a run without a mount (3), of a definition that does not exist (2) and of a data directory mounted read-only (1). The folder of the runner is made writable for UID 1000 first, as any Linux host needs. The job does not run the experiments or any benchmark. It was added on 2026-09-24. Its result for a given commit is on the Actions page of the repository; this file does not copy the results of CI.

## Adding the training dependencies of M1

The framework for the learned models is chosen in I3 (`docs/m1_plan.md`, section 8.6). When it is:

1. Declare it where the project declares its dependencies, `pyproject.toml`, as a dependency or an optional group such as `train`.
2. Add its wheels to `docker/requirements.lock.txt` in the same way: one version, the hash of the Linux wheel for CPython 3.13, produced in a throwaway container of the base image. For PyTorch that is the CPU-only wheel from its own index, which avoids the CUDA libraries.
3. Rebuild. The dependency layer comes before the code, so it is rebuilt once and then cached.

If the training packages make the image much larger than data generation needs, the Dockerfile can build two targets from one base, one for the data path and one for training, and `docker build --target` chooses. GPUs, Docker Compose, registries and cloud deployment remain out of scope.
