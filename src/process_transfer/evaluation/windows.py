"""Windows, phases and contexts: the index contract of M1 (``docs/m1_plan.md``, section 4.3).

Ticks count the rows of a run from its start: tick k is the row at k * 6 s after it. A
window is where one prediction starts and is scored. Around the onset tick o of an
excursion of protocol P3:

    context    o - 9 ... o          10 readings in (t_o - 60 s, t_o]; their mean is the
                                    initial state of the prediction; never scored in it
    excursion  o + 1 ... o + 20     scored
    return     o + 21 ... o + 70    scored
    settled    o + 71 ... o + 110   scored

and the ten readings after it, o + 111 ... o + 120, are the context of the next window. A
run of single-input steps has one window: its onset at the step and 200 scored readings in
four phases of 50. The row at an onset carries the new inputs and the reading of a state
that has not yet responded to them (the right-continuous convention of the data contract),
so that reading belongs to the context. A prediction with L scored readings is driven by
the inputs of rows o ... o + L - 1; the input of row o + L applies after its last instant.

How windows are found. From the known inputs and the known nominal inputs, nothing else.
An onset is a row whose inputs are not the nominal ones while the row before carries them,
or the first row of a run when its inputs are not nominal. The inputs are exact (D-020) and
the nominal inputs of an export are the same floating-point numbers, so they are compared
exactly; a tolerance would need a scale that nothing here provides. An excursion forms a
window only if the ten rows before its onset carry the nominal inputs, so that its context
is read under constant nominal inputs, and the run lasts until its last scored reading. The
first excursion of a P3 run of M0 starts at tick 0, has no context and forms no window. An
excursion without a window is recorded with its reason.

What a layout requires of a run that forms windows is checked, and a run that breaks it is
refused rather than read with wrong phases: the moved inputs are held for exactly the rows
of the layout, the nominal inputs hold for the rest of the window, and the context of a
window never reaches into the scored readings of the window before.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from process_transfer.cstr_variables import (
    INPUT_NAMES,
    INPUT_UNITS,
    STATE_NAMES,
    STATE_UNITS,
    FloatArray,
)
from process_transfer.measurement.observations import Observations
from process_transfer.sampling_clock import nearest_ticks

CONTEXT_READINGS = 10
SAMPLE_PERIOD = 6.0  # s, the sensor clock the contract is written for


@dataclass(frozen=True)
class Phase:
    name: str
    readings: int


@dataclass(frozen=True)
class WindowLayout:
    """The shape of the windows of one protocol, in ticks of the sensor clock."""

    protocol: str  # the operating mode of the runs it applies to
    hold: int  # rows, from the onset on, at which the moved inputs are held
    phases: tuple[Phase, ...]  # the scored readings, in order

    def __post_init__(self) -> None:
        if not self.phases or any(phase.readings < 1 for phase in self.phases):
            raise ValueError(f"{self.protocol}: every phase needs at least one reading")
        names = [phase.name for phase in self.phases]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.protocol}: phase names must be distinct, got {names}")
        if not 1 <= self.hold <= self.scored_readings:
            raise ValueError(
                f"{self.protocol}: the hold of {self.hold} rows must end inside the window of "
                f"{self.scored_readings} scored readings"
            )

    @property
    def scored_readings(self) -> int:
        return sum(phase.readings for phase in self.phases)

    def phase_slices(self) -> tuple[tuple[str, slice], ...]:
        """Where each phase lies among the scored readings of a window."""
        slices, start = [], 0
        for phase in self.phases:
            slices.append((phase.name, slice(start, start + phase.readings)))
            start += phase.readings
        return tuple(slices)


P3_LAYOUT = WindowLayout(
    protocol="p3",
    hold=20,
    phases=(Phase("excursion", 20), Phase("return", 50), Phase("settled", 40)),
)
STEP_LAYOUT = WindowLayout(
    protocol="step",
    hold=100,
    phases=(
        Phase("hold transient", 50),
        Phase("hold settled", 50),
        Phase("recovery transient", 50),
        Phase("recovery settled", 50),
    ),
)
_LAYOUTS = {layout.protocol: layout for layout in (P3_LAYOUT, STEP_LAYOUT)}


def layout_for(operating_mode: str) -> WindowLayout:
    """The layout of the windows of a run, from the operating mode its export states."""
    if operating_mode not in _LAYOUTS:
        raise ValueError(
            f"no window layout for runs of mode {operating_mode!r}; there are layouts for "
            f"{sorted(_LAYOUTS)}. A steady run has no excursion and so no window"
        )
    return _LAYOUTS[operating_mode]


@dataclass(frozen=True)
class Window:
    """One window of a run: which rows are its context, its scored readings and its inputs."""

    run_id: str
    excursion: int  # 1-based position of its excursion among those of the run
    onset: int  # tick
    layout: WindowLayout

    @property
    def key(self) -> tuple[str, int]:
        return (self.run_id, self.excursion)

    @property
    def context_ticks(self) -> range:
        return range(self.onset - CONTEXT_READINGS + 1, self.onset + 1)

    @property
    def scored_ticks(self) -> range:
        return range(self.onset + 1, self.onset + 1 + self.layout.scored_readings)

    @property
    def input_ticks(self) -> range:
        """The rows whose inputs drive the prediction of the window."""
        return range(self.onset, self.onset + self.layout.scored_readings)

    @property
    def first_tick(self) -> int:
        return self.context_ticks[0]

    @property
    def last_tick(self) -> int:
        return self.scored_ticks[-1]


@dataclass(frozen=True)
class SkippedExcursion:
    """An excursion that forms no window, and why."""

    excursion: int
    onset: int
    reason: str


@dataclass(frozen=True)
class RunWindows:
    """The excursions of one run and the windows they form."""

    run_id: str
    layout: WindowLayout
    n_ticks: int
    onsets: tuple[int, ...]  # of every excursion, with or without a window
    windows: tuple[Window, ...]
    skipped: tuple[SkippedExcursion, ...]


def require_cstr_channels(observations: Observations) -> None:
    """The readings and inputs must be the channels of the CSTR, in the order and the SI
    units in which the models hold their states and inputs."""
    found = (
        observations.measured_names,
        observations.measured_units,
        observations.input_names,
        observations.input_units,
    )
    expected = (STATE_NAMES, STATE_UNITS, INPUT_NAMES, INPUT_UNITS)
    if found != expected:
        raise ValueError(
            f"run {observations.run!r} has channels {found}; the evaluation contract is "
            f"written for {expected}"
        )


def run_ticks(observations: Observations) -> int:
    """The number of rows of a run, after checking that row k is tick k of the sensor
    clock of 6 s, with nothing missing."""
    if observations.sample_period != SAMPLE_PERIOD:
        raise ValueError(
            f"run {observations.run!r} is sampled every {observations.sample_period!r} s; the "
            f"index contract is written for readings every {SAMPLE_PERIOD} s"
        )
    ticks, on_clock = nearest_ticks(
        observations.times, float(observations.times[0]), observations.sample_period
    )
    n = observations.n_samples
    if not np.all(on_clock) or not np.array_equal(ticks, np.arange(n)):
        raise ValueError(
            f"the rows of run {observations.run!r} are not the consecutive ticks of its sensor "
            "clock; a window is defined on ticks, and none is formed on a run with gaps"
        )
    return n


def _nominal_vector(nominal_inputs: FloatArray) -> FloatArray:
    nominal = np.asarray(nominal_inputs, dtype=np.float64)
    if nominal.shape != (len(INPUT_NAMES),) or not np.all(np.isfinite(nominal)):
        raise ValueError(f"the nominal inputs must be {len(INPUT_NAMES)} finite values")
    return nominal


def find_windows(
    observations: Observations, nominal_inputs: FloatArray, layout: WindowLayout
) -> RunWindows:
    """The windows of a run, found from its known inputs and the known nominal inputs."""
    require_cstr_channels(observations)
    n = run_ticks(observations)
    nominal = _nominal_vector(nominal_inputs)
    inputs = observations.inputs
    run = observations.run
    at_nominal = np.all(inputs == nominal, axis=1)
    if not np.any(at_nominal):
        # Every run of these protocols rests at the nominal inputs. Without this check a
        # disagreement in the last bit between the export's nominal inputs and the rows
        # would leave one excursion at tick 0 and no window, without a word.
        raise ValueError(
            f"no row of run {observations.run!r} carries the nominal inputs "
            f"{nominal.tolist()}; the known nominal inputs and the run disagree"
        )
    before_at_nominal = np.concatenate(([True], at_nominal[:-1]))
    onsets = tuple(int(tick) for tick in np.flatnonzero(~at_nominal & before_at_nominal))

    length = layout.scored_readings
    windows: list[Window] = []
    skipped: list[SkippedExcursion] = []
    for excursion, onset in enumerate(onsets, start=1):
        if onset < CONTEXT_READINGS or not np.all(at_nominal[onset - CONTEXT_READINGS : onset]):
            skipped.append(
                SkippedExcursion(
                    excursion,
                    onset,
                    f"no context: the {CONTEXT_READINGS} rows before the onset at tick {onset} "
                    "are not all rows at the nominal inputs",
                )
            )
            continue
        if onset + length > n - 1:
            skipped.append(
                SkippedExcursion(
                    excursion,
                    onset,
                    f"the run ends at tick {n - 1}, before the last scored reading of the "
                    f"window, tick {onset + length}",
                )
            )
            continue
        held = inputs[onset : onset + layout.hold]
        if not np.all(held == inputs[onset]):
            raise ValueError(
                f"run {run!r}: the inputs moved at tick {onset} change again before tick "
                f"{onset + layout.hold}; the {layout.protocol} layout holds them for "
                f"{layout.hold} rows"
            )
        if not np.all(at_nominal[onset + layout.hold : onset + length]):
            raise ValueError(
                f"run {run!r}: the inputs are not nominal from tick {onset + layout.hold} to "
                f"tick {onset + length - 1}, inside the window of the excursion at tick "
                f"{onset}; its phases would be read wrongly"
            )
        window = Window(run, excursion, onset, layout)
        if windows and window.first_tick <= windows[-1].last_tick:
            raise ValueError(
                f"run {run!r}: the context of the window at tick {onset} starts at tick "
                f"{window.first_tick}, inside the scored readings of the window before, which "
                f"end at tick {windows[-1].last_tick}; no reading may be both"
            )
        windows.append(window)
    return RunWindows(run, layout, n, onsets, tuple(windows), tuple(skipped))


def _read_only(values: FloatArray) -> FloatArray:
    copy = np.array(values, dtype=np.float64)
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class WindowData:
    """What may be read of one window: its context readings, its scored readings and the
    inputs that drive its prediction. The initial state is the mean of the context
    readings, computed here and nowhere else.

    It is the unit in which data are handed over. A fit is given the ``WindowData`` of its
    part of a budget and nothing else, so it cannot read a row outside that part.
    """

    window: Window
    sample_period: float  # s
    noise_std: FloatArray  # per measured channel, from the data sheet
    onset_time: float  # s
    context: FloatArray  # (10, 2) readings at the context ticks
    scored: FloatArray  # (L, 2) readings at the scored ticks
    inputs: FloatArray  # (L, 4) inputs of the rows that drive the prediction
    initial_state: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        length = self.window.layout.scored_readings
        shapes = {
            "context": (CONTEXT_READINGS, len(STATE_NAMES)),
            "scored": (length, len(STATE_NAMES)),
            "inputs": (length, len(INPUT_NAMES)),
            "noise_std": (len(STATE_NAMES),),
        }
        for name, shape in shapes.items():
            values = _read_only(getattr(self, name))
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, values)
        if np.any(self.noise_std < 0.0):
            raise ValueError(f"noise_std must not be negative, got {self.noise_std.tolist()}")
        if not (np.isfinite(self.onset_time) and self.sample_period > 0.0):
            raise ValueError("the onset time must be finite and the sample period positive")
        object.__setattr__(self, "initial_state", _read_only(np.mean(self.context, axis=0)))

    @property
    def key(self) -> tuple[str, int]:
        return self.window.key

    @property
    def scored_offsets(self) -> FloatArray:
        """The instants of the scored readings, in seconds after the onset."""
        return self.sample_period * np.arange(
            1, self.window.layout.scored_readings + 1, dtype=np.float64
        )

    @property
    def ticks_read(self) -> frozenset[int]:
        """Every row of the run that this window holds something of."""
        window = self.window
        return frozenset((*window.context_ticks, *window.scored_ticks, *window.input_ticks))


def window_data(observations: Observations, windows: Sequence[Window]) -> tuple[WindowData, ...]:
    """The readings and inputs of ``windows``, and of nothing else, from the run they belong
    to."""
    require_cstr_channels(observations)
    n = run_ticks(observations)
    extracted = []
    for window in windows:
        if window.run_id != observations.run:
            raise ValueError(
                f"window {window.key} belongs to run {window.run_id!r}, not to {observations.run!r}"
            )
        if window.first_tick < 0 or window.last_tick > n - 1:
            raise ValueError(f"window {window.key} reaches beyond the {n} rows of its run")
        context, scored, rows = window.context_ticks, window.scored_ticks, window.input_ticks
        extracted.append(
            WindowData(
                window=window,
                sample_period=observations.sample_period,
                noise_std=np.array(observations.noise_std, dtype=np.float64),
                onset_time=float(observations.times[window.onset]),
                context=observations.measured[context.start : context.stop],
                scored=observations.measured[scored.start : scored.stop],
                inputs=observations.inputs[rows.start : rows.stop],
            )
        )
    return tuple(extracted)
