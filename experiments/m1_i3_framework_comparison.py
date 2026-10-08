"""I3 of M1: a short comparison of two training frameworks on development data
(``docs/m1_plan.md``, section 8.6). Exploratory: it chooses software, not a model.

The same task is written in each framework, JAX (with diffrax for its adaptive solver) and
PyTorch (with torchdiffeq), in double precision on one CPU thread, since the fits of the
benchmark will run one per process:

* the nine windows of the target run ``target.p3.e0.x10.n0`` of the export ``m0-e05``,
  each started from the mean of its context and driven by its inputs, scored on its 110
  readings with the loss J^2 of section 9.1 of the plan;
* two right-hand sides. HK: the modeller's balances with the rate multiplied by
  exp(N(z)), N a network of one hidden layer of 16 tanh units on the normalised C_A and T,
  its output layer zero so that the factor is one. BN: dx/dt = D N(normalised x, u), two
  hidden layers of 32 tanh units. The mechanistic parameters of HK are those of MR fitted
  on the same windows; the scales come from the same windows. The weights are drawn once
  with numpy from a declared seed and handed to both frameworks, so that both compute the
  same function;
* two integrations. A fixed-step fourth-order Runge-Kutta scheme written here, with two
  steps per row of 6 s, whose steps never cross a change of the inputs, since each row is
  integrated with its own inputs; and each library's adaptive solver (diffrax Tsit5,
  torchdiffeq dopri5) at rtol = atol = 1e-8, told where the inputs change.

Measured: the agreement of each integration with the reference rollout of the repository
(LSODA, ``models.rollout``) on HK at the identity, which is MR; the gradient of the loss
against central finite differences; whether a training of 200 steps of Adam gives the same
numbers twice; the time of a loss and its gradient, and of the training. The time and
size of the installation are measured outside this script and recorded with its result.

Usage, in an environment with the framework installed, with PT_DATA_DIR outside the
repository:

    python experiments/m1_i3_framework_comparison.py --framework jax --exports <exports>
    python experiments/m1_i3_framework_comparison.py --framework torch --exports <exports>

Outputs go to PT_DATA_DIR/experiments/m1_i3_framework/<run id>/summary_<framework>.json; the
second framework is given the run directory of the first with --run-directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

# one thread per process, before either framework is imported
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from process_transfer.data.export import open_export_directory  # noqa: E402
from process_transfer.data.paths import data_dir, repository_root  # noqa: E402
from process_transfer.data.provenance import environment, git_state, new_run_directory  # noqa: E402
from process_transfer.evaluation.plant import read_known_plant  # noqa: E402
from process_transfer.evaluation.windows import find_windows, layout_for, window_data  # noqa: E402
from process_transfer.models.fitting import fit_mechanistic  # noqa: E402
from process_transfer.models.mechanistic import (  # noqa: E402
    REFERENCE_TEMPERATURE,
    MechanisticModel,
    modeller_values,
)
from process_transfer.models.rollout import rollout  # noqa: E402

RUN = "target.p3.e0.x10.n0"
SEED = 20260928  # the weights of both networks
SUBSTEPS = 2  # fixed steps per row of 6 s
ADAPTIVE_TOLERANCE = 1e-8
ADAM_STEPS = 200
ADAM_RATE = 1e-3
REPEATS = 20  # timings of a loss and its gradient
FD_STEP = 1e-6  # relative step of the central differences
HK_WIDTH = 16
BN_WIDTH = 32


def glorot(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def problem(exports: Path) -> dict[str, object]:
    """The windows, the scales, MR, and the initial weights, as numpy arrays."""
    export = open_export_directory(exports / "m0-e05")
    known = read_known_plant(export, "target")
    modeller = modeller_values()
    observations = export.observations(RUN)
    windows = window_data(
        observations, find_windows(observations, known.nominal_inputs, layout_for("p3")).windows
    )
    mr = fit_mechanistic("MR", windows, known, modeller).parameters
    readings = np.concatenate([np.vstack([d.context, d.scored]) for d in windows])
    inputs = np.concatenate([d.inputs for d in windows])
    rng = np.random.default_rng(SEED)
    hk = {
        "w1": glorot(rng, 2, HK_WIDTH),
        "b1": np.zeros(HK_WIDTH),
        "w2": np.zeros((HK_WIDTH, 1)),
        "b2": np.zeros(1),
    }
    bn = {
        "w1": glorot(rng, 6, BN_WIDTH),
        "b1": np.zeros(BN_WIDTH),
        "w2": glorot(rng, BN_WIDTH, BN_WIDTH),
        "b2": np.zeros(BN_WIDTH),
        "w3": glorot(rng, BN_WIDTH, 2),
        "b3": np.zeros(2),
    }
    state_scale = readings.std(axis=0)
    return {
        "known": known,
        "windows": windows,
        "mr": mr,
        "x0": np.array([d.initial_state for d in windows]),
        "inputs": np.array([d.inputs for d in windows]),
        "scored": np.array([d.scored for d in windows]),
        "sigma": np.array(windows[0].noise_std),
        "state_mean": readings.mean(axis=0),
        "state_scale": state_scale,
        "input_mean": inputs.mean(axis=0),
        "input_scale": inputs.std(axis=0),
        # a derivative scale: the state's spread over a time constant of a minute
        "derivative_scale": state_scale / 60.0,
        "theta": np.array(
            [math.log(mr.k_350), mr.activation_temperature / REFERENCE_TEMPERATURE, math.log(mr.ua)]
        ),
        "hk": hk,
        "bn": bn,
        "sample_period": windows[0].sample_period,
    }


def reference_states(p: dict[str, object]) -> np.ndarray:
    """MR's rollouts by the reference integration of the repository."""
    model = MechanisticModel("MR", p["known"], p["mr"])
    return np.array(
        [rollout(model, d.initial_state, d.inputs, d.sample_period).states for d in p["windows"]]
    )


def digest(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# JAX
# --------------------------------------------------------------------------- #


def run_jax(p: dict[str, object]) -> dict[str, object]:
    import jax
    import jax.flatten_util
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import diffrax

    k = p["known"]
    dilution_volume = k.volume
    heat = k.heat_release_per_mole
    thermal_mass = k.thermal_mass
    h = p["sample_period"] / SUBSTEPS
    sm, ss = jnp.asarray(p["state_mean"]), jnp.asarray(p["state_scale"])
    um, us = jnp.asarray(p["input_mean"]), jnp.asarray(p["input_scale"])
    dscale = jnp.asarray(p["derivative_scale"])
    sigma = jnp.asarray(p["sigma"])

    def hk_rhs(params, x, u):
        theta, net = params
        c, t = x[..., 0], x[..., 1]
        z = (x - sm) / ss
        g = jnp.exp((jnp.tanh(z @ net["w1"] + net["b1"]) @ net["w2"] + net["b2"])[..., 0])
        k_t = jnp.exp(
            theta[0] - REFERENCE_TEMPERATURE * theta[1] * (1.0 / t - 1.0 / REFERENCE_TEMPERATURE)
        )
        rate = k_t * c * g
        d = u[..., 0] / dilution_volume
        ua = jnp.exp(theta[2])
        dc = d * (u[..., 1] - c) - rate
        dt = d * (u[..., 2] - t) + heat * rate - ua / thermal_mass * (t - u[..., 3])
        return jnp.stack([dc, dt], axis=-1)

    def bn_rhs(params, x, u):
        net = params
        z = jnp.concatenate([(x - sm) / ss, (u - um) / us], axis=-1)
        a = jnp.tanh(z @ net["w1"] + net["b1"])
        a = jnp.tanh(a @ net["w2"] + net["b2"])
        return dscale * (a @ net["w3"] + net["b3"])

    def rk4_rollout(rhs, params, x0, inputs):
        def row(x, u):
            for _ in range(SUBSTEPS):
                k1 = rhs(params, x, u)
                k2 = rhs(params, x + 0.5 * h * k1, u)
                k3 = rhs(params, x + 0.5 * h * k2, u)
                k4 = rhs(params, x + h * k3, u)
                x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return x, x

        _, states = jax.lax.scan(row, x0, jnp.swapaxes(inputs, 0, 1))
        return jnp.swapaxes(states, 0, 1)

    period = p["sample_period"]
    rows = p["inputs"].shape[1]
    times = period * jnp.arange(1, rows + 1)

    def adaptive_rollout(rhs, params, x0, inputs):
        # the inputs of row i act over [i h, (i + 1) h); the solver is told every row
        # boundary, the windows changing their inputs at different rows

        def one(x0w, uw):
            def field(t, x, args):
                i = jnp.clip(jnp.floor(t / period).astype(int), 0, rows - 1)
                return rhs(params, x, uw[i])

            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(field),
                diffrax.Tsit5(),
                t0=0.0,
                t1=period * rows,
                dt0=1.0,
                y0=x0w,
                saveat=diffrax.SaveAt(ts=times),
                stepsize_controller=diffrax.PIDController(
                    rtol=ADAPTIVE_TOLERANCE,
                    atol=ADAPTIVE_TOLERANCE,
                    jump_ts=period * jnp.arange(1, rows),
                ),
                max_steps=4096,
            )
            return solution.ys

        return jax.vmap(one)(x0, inputs)

    x0, inputs, scored = (jnp.asarray(p[name]) for name in ("x0", "inputs", "scored"))

    def loss(rollout_fn, rhs, params):
        predicted = rollout_fn(rhs, params, x0, inputs)
        return jnp.mean(((predicted - scored) / sigma) ** 2)

    hk_params = (jnp.asarray(p["theta"]), {n: jnp.asarray(v) for n, v in p["hk"].items()})
    bn_params = {n: jnp.asarray(v) for n, v in p["bn"].items()}
    found: dict[str, object] = {
        "framework": "jax",
        "versions": {
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "diffrax": diffrax.__version__,
            "backend": jax.default_backend(),
            "x64": bool(jax.config.jax_enable_x64),
        },
    }

    reference = reference_states(p)
    for label, fn in (("fixed", rk4_rollout), ("adaptive", adaptive_rollout)):
        states = np.asarray(jax.jit(lambda prm, fn=fn: fn(hk_rhs, prm, x0, inputs))(hk_params))
        found[f"hk_identity_{label}_against_reference_in_sigmas"] = float(
            np.max(np.abs(states - reference) / p["sigma"])
        )

    cases = {
        ("hk", "fixed"): (rk4_rollout, hk_rhs, hk_params),
        ("bn", "fixed"): (rk4_rollout, bn_rhs, bn_params),
        ("hk", "adaptive"): (adaptive_rollout, hk_rhs, hk_params),
        ("bn", "adaptive"): (adaptive_rollout, bn_rhs, bn_params),
    }
    for (model, integ), (fn, rhs, params) in cases.items():
        value_and_grad = jax.jit(jax.value_and_grad(lambda prm, fn=fn, rhs=rhs: loss(fn, rhs, prm)))
        began = time.perf_counter()
        value, grad = value_and_grad(params)
        jax.block_until_ready(grad)
        first = time.perf_counter() - began
        seconds = []
        for _ in range(REPEATS):
            began = time.perf_counter()
            jax.block_until_ready(value_and_grad(params))
            seconds.append(time.perf_counter() - began)
        found[f"{model}_{integ}"] = {
            "loss": float(value),
            "first_call_seconds": first,
            "median_seconds": float(np.median(seconds)),
        }

    # gradients against central differences, on the fixed-step loss
    rng = np.random.default_rng(1)
    for model, rhs, params in (("hk", hk_rhs, hk_params), ("bn", bn_rhs, bn_params)):
        flat, unravel = jax.flatten_util.ravel_pytree(params)
        f = jax.jit(lambda v, rhs=rhs, unravel=unravel: loss(rk4_rollout, rhs, unravel(v)))
        gradient = np.asarray(jax.jit(jax.grad(f))(flat))
        # after one step of Adam the output weights of HK are no longer zero, so every weight
        # has a gradient; check at a perturbed point as well as at the start
        points = {
            "start": np.asarray(flat),
            "perturbed": np.asarray(flat) + 0.05 * rng.standard_normal(flat.shape),
        }
        for where, point in points.items():
            gradient = np.asarray(jax.jit(jax.grad(f))(jnp.asarray(point)))
            chosen = sorted(
                set(rng.choice(len(point), size=min(12, len(point)), replace=False))
                | ({0, 1, 2} if model == "hk" else set())
            )
            errors = []
            for i in chosen:
                step = FD_STEP * max(1.0, abs(point[i]))
                up, down = point.copy(), point.copy()
                up[i] += step
                down[i] -= step
                fd = (float(f(jnp.asarray(up))) - float(f(jnp.asarray(down)))) / (2.0 * step)
                errors.append(abs(fd - gradient[i]) / max(abs(fd), abs(gradient[i]), 1e-300))
            found[f"{model}_gradient_against_differences_{where}"] = {
                "parameters_checked": len(chosen),
                "largest_relative_difference": float(max(errors)),
            }

    # Adam, twice
    def adam(model, rhs, params):
        flat, unravel = jax.flatten_util.ravel_pytree(params)
        f = jax.jit(jax.value_and_grad(lambda v: loss(rk4_rollout, rhs, unravel(v))))
        m = jnp.zeros_like(flat)
        v = jnp.zeros_like(flat)
        losses = []
        began = time.perf_counter()
        for step in range(1, ADAM_STEPS + 1):
            value, g = f(flat)
            losses.append(float(value))
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            flat = flat - ADAM_RATE * (m / (1 - 0.9**step)) / (
                jnp.sqrt(v / (1 - 0.999**step)) + 1e-8
            )
        return losses, np.asarray(flat), time.perf_counter() - began

    for model, rhs, params in (("hk", hk_rhs, hk_params), ("bn", bn_rhs, bn_params)):
        first = adam(model, rhs, params)
        second = adam(model, rhs, params)
        found[f"{model}_adam"] = {
            "steps": ADAM_STEPS,
            "seconds": first[2],
            "loss_first_and_last": [first[0][0], first[0][-1]],
            "identical_twice": bool(np.array_equal(first[1], second[1]) and first[0] == second[0]),
            "digest": digest([first[1]]),
        }
    return found


# --------------------------------------------------------------------------- #
# PyTorch
# --------------------------------------------------------------------------- #


def run_torch(p: dict[str, object]) -> dict[str, object]:
    import torch
    import torchdiffeq

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    k = p["known"]
    heat = k.heat_release_per_mole
    thermal_mass = k.thermal_mass
    h = p["sample_period"] / SUBSTEPS
    t64 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
    sm, ss, um, us = (t64(p[n]) for n in ("state_mean", "state_scale", "input_mean", "input_scale"))
    dscale, sigma = t64(p["derivative_scale"]), t64(p["sigma"])

    def hk_rhs(params, x, u):
        theta, net = params
        c, t = x[..., 0], x[..., 1]
        z = (x - sm) / ss
        g = torch.exp((torch.tanh(z @ net["w1"] + net["b1"]) @ net["w2"] + net["b2"])[..., 0])
        k_t = torch.exp(
            theta[0] - REFERENCE_TEMPERATURE * theta[1] * (1.0 / t - 1.0 / REFERENCE_TEMPERATURE)
        )
        rate = k_t * c * g
        d = u[..., 0] / k.volume
        dc = d * (u[..., 1] - c) - rate
        dt = (
            d * (u[..., 2] - t) + heat * rate - torch.exp(theta[2]) / thermal_mass * (t - u[..., 3])
        )
        return torch.stack([dc, dt], dim=-1)

    def bn_rhs(params, x, u):
        net = params
        z = torch.cat([(x - sm) / ss, (u - um) / us], dim=-1)
        a = torch.tanh(z @ net["w1"] + net["b1"])
        a = torch.tanh(a @ net["w2"] + net["b2"])
        return dscale * (a @ net["w3"] + net["b3"])

    def rk4_rollout(rhs, params, x0, inputs):
        x, states = x0, []
        for i in range(inputs.shape[1]):
            u = inputs[:, i]
            for _ in range(SUBSTEPS):
                k1 = rhs(params, x, u)
                k2 = rhs(params, x + 0.5 * h * k1, u)
                k3 = rhs(params, x + 0.5 * h * k2, u)
                k4 = rhs(params, x + h * k3, u)
                x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            states.append(x)
        return torch.stack(states, dim=1)

    period = p["sample_period"]
    rows = p["inputs"].shape[1]
    times = torch.cat([torch.zeros(1), period * torch.arange(1, rows + 1)])

    def adaptive_rollout(rhs, params, x0, inputs):
        # the inputs of row i act over [i h, (i + 1) h); at a jump the solver moves t to
        # the side it integrates on, one representable number away
        def field(t, x):
            i = min(max(int(math.floor(float(t) / period)), 0), rows - 1)
            return rhs(params, x, inputs[:, i])

        states = torchdiffeq.odeint(
            field,
            x0,
            times,
            method="dopri5",
            rtol=ADAPTIVE_TOLERANCE,
            atol=ADAPTIVE_TOLERANCE,
            options={"jump_t": period * torch.arange(1, rows)},
        )
        return states[1:].transpose(0, 1)

    x0, inputs, scored = (t64(p[name]) for name in ("x0", "inputs", "scored"))

    def loss(rollout_fn, rhs, params):
        predicted = rollout_fn(rhs, params, x0, inputs)
        return torch.mean(((predicted - scored) / sigma) ** 2)

    def fresh(model):
        net = {n: t64(v).clone().requires_grad_(True) for n, v in p[model].items()}
        return (t64(p["theta"]).clone().requires_grad_(True), net) if model == "hk" else net

    def leaves(params):
        return (
            [params[0], *params[1].values()] if isinstance(params, tuple) else list(params.values())
        )

    found: dict[str, object] = {
        "framework": "torch",
        "versions": {"torch": torch.__version__, "torchdiffeq": torchdiffeq.__version__},
    }
    reference = reference_states(p)
    with torch.no_grad():
        for label, fn in (("fixed", rk4_rollout), ("adaptive", adaptive_rollout)):
            states = fn(hk_rhs, fresh("hk"), x0, inputs).numpy()
            found[f"hk_identity_{label}_against_reference_in_sigmas"] = float(
                np.max(np.abs(states - reference) / p["sigma"])
            )

    for model, rhs in (("hk", hk_rhs), ("bn", bn_rhs)):
        for integ, fn in (("fixed", rk4_rollout), ("adaptive", adaptive_rollout)):
            params = fresh(model)
            seconds = []
            value = None
            for _ in range(REPEATS + 1):
                began = time.perf_counter()
                for leaf in leaves(params):
                    leaf.grad = None
                value = loss(fn, rhs, params)
                value.backward()
                seconds.append(time.perf_counter() - began)
            found[f"{model}_{integ}"] = {
                "loss": value.item(),
                "first_call_seconds": seconds[0],
                "median_seconds": float(np.median(seconds[1:])),
            }

    rng = np.random.default_rng(1)
    for model, rhs in (("hk", hk_rhs), ("bn", bn_rhs)):
        shapes = [leaf.shape for leaf in leaves(fresh(model))]
        sizes = [int(np.prod(s)) for s in shapes]

        def unravel(vector, model=model, shapes=shapes, sizes=sizes):
            parts, start = [], 0
            for shape, size in zip(shapes, sizes, strict=True):
                parts.append(vector[start : start + size].reshape(shape))
                start += size
            if model == "hk":
                return (parts[0], dict(zip(p["hk"], parts[1:], strict=True)))
            return dict(zip(p["bn"], parts, strict=True))

        start_point = torch.cat(
            [leaf.detach().reshape(-1) for leaf in leaves(fresh(model))]
        ).numpy()
        points = {
            "start": start_point,
            "perturbed": start_point + 0.05 * rng.standard_normal(start_point.shape),
        }
        for where, point in points.items():
            v = t64(point).clone().requires_grad_(True)
            value = loss(rk4_rollout, rhs, unravel(v))
            (gradient,) = torch.autograd.grad(value, v)
            gradient = gradient.numpy()
            chosen = sorted(
                set(rng.choice(len(point), size=min(12, len(point)), replace=False))
                | ({0, 1, 2} if model == "hk" else set())
            )
            errors = []
            with torch.no_grad():
                for i in chosen:
                    step = FD_STEP * max(1.0, abs(point[i]))
                    up, down = point.copy(), point.copy()
                    up[i] += step
                    down[i] -= step
                    fd = (
                        float(loss(rk4_rollout, rhs, unravel(t64(up))))
                        - float(loss(rk4_rollout, rhs, unravel(t64(down))))
                    ) / (2.0 * step)
                    errors.append(abs(fd - gradient[i]) / max(abs(fd), abs(gradient[i]), 1e-300))
            found[f"{model}_gradient_against_differences_{where}"] = {
                "parameters_checked": len(chosen),
                "largest_relative_difference": float(max(errors)),
            }

    def adam(model, rhs):
        params = fresh(model)
        optimiser = torch.optim.Adam(leaves(params), lr=ADAM_RATE, betas=(0.9, 0.999), eps=1e-8)
        losses = []
        began = time.perf_counter()
        for _ in range(ADAM_STEPS):
            optimiser.zero_grad()
            value = loss(rk4_rollout, rhs, params)
            value.backward()
            losses.append(value.item())
            optimiser.step()
        final = torch.cat([leaf.detach().reshape(-1) for leaf in leaves(params)]).numpy()
        return losses, final, time.perf_counter() - began

    for model, rhs in (("hk", hk_rhs), ("bn", bn_rhs)):
        first = adam(model, rhs)
        second = adam(model, rhs)
        found[f"{model}_adam"] = {
            "steps": ADAM_STEPS,
            "seconds": first[2],
            "loss_first_and_last": [first[0][0], first[0][-1]],
            "identical_twice": bool(np.array_equal(first[1], second[1]) and first[0] == second[0]),
            "digest": digest([first[1]]),
        }
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--framework", choices=("jax", "torch"), required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument(
        "--run-directory",
        type=Path,
        help="write into this run directory, that of the other framework, instead of a new one",
    )
    arguments = parser.parse_args()
    root = repository_root().resolve()
    target = data_dir().resolve()
    if target == root or root in target.parents:
        print(f"PT_DATA_DIR resolves to {target}, inside the repository; refused", file=sys.stderr)
        return 1
    state = git_state()
    began = time.perf_counter()
    p = problem(arguments.exports)
    found = run_jax(p) if arguments.framework == "jax" else run_torch(p)
    found["seconds"] = time.perf_counter() - began
    found["provenance"] = {"git": state, "environment": environment(), "data_dir": str(target)}
    found["fixed_before_the_run"] = {
        "run": RUN,
        "seed": SEED,
        "substeps": SUBSTEPS,
        "adaptive_tolerance": ADAPTIVE_TOLERANCE,
        "adam_steps": ADAM_STEPS,
        "adam_rate": ADAM_RATE,
        "repeats": REPEATS,
        "fd_step": FD_STEP,
        "hk_width": HK_WIDTH,
        "bn_width": BN_WIDTH,
        "mr": [p["mr"].k_350, p["mr"].activation_temperature, p["mr"].ua],
    }
    directory = arguments.run_directory or new_run_directory("m1_i3_framework", state)
    summary = directory / f"summary_{arguments.framework}.json"
    summary.write_text(json.dumps(found, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in found.items() if k != "provenance"}, indent=1))
    print(directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
