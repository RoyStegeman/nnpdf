"""
The n3fit <-> backend contract.

**Status.**  This module is the target interface agreed in the design documents
(see ``proposed/n3fit-backend-contract.md``).  It is declared in full and consumed
in part: P0/P1 wired up the registry, the capabilities, the import ban and the
conformance suite; P2 added the role vocabulary and the ``Model`` access path.

======================  =====================================================
Member                  Status
======================  =====================================================
ShapeSpec,              defined; exercised by the conformance suite and by
ParametrizationSpec,    the Keras capabilities/parametrization mapping
ObjectiveSpec,
OptimizerSpec, FitResult,
StepContext
ROLE_*, ROLES,          **active (P2)**: the closed role vocabulary, and the
role_of                 first path component of every weight key
Model                   **active (P2)**: ``weights(role)``, ``bind_input``,
                        ``bound_inputs``, ``override``, ``freeze``, ``summary``
                        (the Keras adapter is ``keras_backend/roles.py``);
                        ``parameters()``/``set_weights()`` still P4/P7
WeightsView, Ensemble   defined; P2b/P4 (per-replica views, whole-model access)
Layer, Tensor, Ops     defined; the Keras backend exposes its existing
                        ``operations`` module as ``ops``
Objective               defined; P3 (the loss layers become one implementation)
Optimizer, Hook, History   defined; P4 (the training loop)
Capabilities,           **active**: consumed by ``n3fit.backends.registry`` and
BackendState            by the conformance suite (P1 moves ``checks``/``hyper_scan``
                        and ``performfit`` over)
Backend                 **active**: the Keras implementation is an adapter over
                        the existing code; ``view(graph)`` is the P2 bridge
get_backend             implemented in ``n3fit/backends/registry.py``
======================  =====================================================

Everything else here is deliberately *declared now, implemented later*: it is cheaper to
review the target interface once than to discover it one method at a time.

The n3fit <-> backend contract.

Everything in this module is backend-free: it may import numpy and the standard library
and nothing else.  A backend implementation (``n3fit.backends.keras_backend``,
``n3fit.backends.jax_backend``, ...) implements the protocols defined here, and ``n3fit``
itself talks *only* to these objects.

Design notes (see ``n3fit-backend-contract.md`` for the full rationale):

* The primitive object exposed to n3fit is a **per-replica** model view.  Whether the
  backend trains all replicas in one stacked graph, with ``vmap``, or in a python loop is
  a private implementation detail, declared through
  :attr:`Capabilities.train_n_replicas_together`.
* Weights are exchanged as ``{path: numpy.ndarray}`` maps, one map per ensemble member.
  The path grammar is ``role[/<index or flavour>][/<layer>]/<parameter>`` with roles
  ``parametrization``, ``preprocessing``, ``objective`` (and structural roles such as
  ``sumrule``/``photon``), e.g. ``parametrization/0/kernel``, ``parametrization/c`` (a
  polynomial coefficient), ``preprocessing/alpha/u``, ``objective/LHC_exp/mask``.  This is
  simultaneously the snapshot format (best weights during training), the mutation format
  (masks, Lagrange multipliers) and the on-disk format.
* The engine's unit is the *optimizer step*, per ensemble member.  Hooks fire at *monitored
  steps*, which happen every ``monitor_every`` steps (>= :meth:`Optimizer.min_monitor_interval`).
  Schedules (positivity multipliers, ...) and stopping patience are expressed in optimizer
  steps so that they do not depend on the monitoring granularity.  Monitoring is strictly
  optional: with ``monitor_every=None`` the optimizer runs its whole budget without ever
  reporting back.
* A **parameter vector** ``theta`` is the concatenation of ``Model.parameters()`` in the
  order of ``FitResult.diagnostics["parameter_layout"]``, each array flattened in C order.
  Parameter-space optimizers (Gauss-Newton/Levenberg-Marquardt, direct solves) and Hessian error
  propagation both work in that space, so the convention is part of the contract.
* Specs (``ShapeSpec``, ``ParametrizationSpec``, ``ObjectiveSpec``, ``OptimizerSpec``) are
  frozen, data-only dataclasses: they can be serialized into the fit output and, later,
  interpreted generically (the "hybrid" objective layer) without changing any call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

Array = np.ndarray
"""A plain numpy array.  No backend tensor ever crosses this interface."""

WeightMap = Mapping[str, Array]
"""Weights of *one* replica as ``{path: array}``.  Paths follow the grammar above."""


# --------------------------------------------------------------------------------------
# Specs: frozen, data-only, serializable
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# The role vocabulary
# --------------------------------------------------------------------------------------
# A *role* names a functional section of a model in a way that survives renaming: n3fit asks
# for ``ROLE_PREPROCESSING``, never for a layer called ``preprocessing_factor``.  Backends map
# roles onto whatever their graph calls them (the Keras one does it in
# ``keras_backend/roles.py``), so a second backend is not obliged to reproduce Keras' layer
# names.
#
# Roles are also the first path component of every weight key (see the path grammar in
# ``n3fit-backend-contract.md`` §1.2): ``weights(role="preprocessing")`` returns exactly the
# keys whose path starts with ``preprocessing/``.
#
# The set is closed on purpose: both n3fit and every backend must agree on the strings, and a
# typo should be an error rather than a silently empty lookup.
ROLE_NN = "nn"
"""The trainable core: the neural network (or whatever parametrization is in use)."""

ROLE_PREPROCESSING = "preprocessing"
"""The preprocessing factors (``alpha``/``beta`` per flavour)."""

ROLE_OBJECTIVE = "objective"
"""The objective terms: masks, covariance data, Lagrange multipliers."""

ROLE_PHOTON = "photon"
"""The photon contribution, bound to an input through :meth:`Model.bind_input`."""

ROLE_REFERENCE = "reference"
"""The reference PDF a diagnostic objective is computed around, set through
:meth:`Model.override`."""

ROLE_SUMRULE = "sumrule"
"""The sum-rule normalisation section."""

ROLES = (ROLE_NN, ROLE_PREPROCESSING, ROLE_OBJECTIVE, ROLE_PHOTON, ROLE_REFERENCE, ROLE_SUMRULE)
"""Every role a backend may declare."""


# --------------------------------------------------------------------------------------
# The objective-group vocabulary (P3)
# --------------------------------------------------------------------------------------
# Membership is n3fit's decision (which terms make up the training objective is physics, and
# n3fit already computes it); evaluation and minimization are the backend's.  The three
# optimization groups are the ones ``Optimizer.run``/``evaluate`` speak about; the report groups
# exist so that positivity and integrability losses can be monitored and printed on their own
# instead of being recovered from the ``"<name>_loss"`` key convention (which D11 deletes).
GROUP_TRAINING = "training"
"""The terms minimized by the optimizer."""

GROUP_VALIDATION = "validation"
"""The terms used to decide when to stop (D13: irrelevant to optimizers with their own
convergence criterion)."""

GROUP_EXPERIMENTAL = "experimental"
"""The terms of the full dataset -- no fold mask, the "true" chi2."""

GROUP_POSITIVITY = "positivity"
GROUP_INTEGRABILITY = "integrability"
"""Reporting groups (P3).  Their terms are *also* members of ``training``: these groups exist so
the logs carry each penalty separately rather than only inside a total."""

OPTIMIZATION_GROUPS = (GROUP_TRAINING, GROUP_VALIDATION, GROUP_EXPERIMENTAL)
"""The groups :meth:`Optimizer.run`/``evaluate`` operate on."""

REPORT_GROUPS = (GROUP_POSITIVITY, GROUP_INTEGRABILITY)
"""Groups n3fit asks for by name when it wants a penalty's own value."""


def role_of(path):
    """The role of a weight path, e.g. ``"preprocessing/alpha/up"`` -> ``"preprocessing"``."""
    return path.split("/")[0]


@dataclass(frozen=True)
class ShapeSpec:
    """Every static shape the backend needs in order to build its graphs.

    Passed at build time; this is what allows a ``jit``-compiled backend to work at all
    (and what ``tensor_splitter``/``Mask`` encode implicitly today).
    """

    n_replicas: int
    xgrid_size: int
    flavours_in: int
    flavours_out: int = 14
    ndata: Mapping[str, int] = field(default_factory=dict)
    integration_grid_size: int = 0


@dataclass(frozen=True)
class ParametrizationSpec:
    """The trainable core of the PDF: anything that maps a set of parameters to a function
    of x (D10).

    ``kind`` is a key of :attr:`Capabilities.parametrizations`, which also declares the
    schema of ``options`` -- so this dataclass is *not* a closed vocabulary: a backend can
    offer ``dense``/``dense_per_flavour``/``lstm`` with options
    ``{"nodes": ..., "activations": ..., "initializer": ...}`` *or* e.g. ``polynomial``
    with options ``{"degree": 5, "basis": "chebyshev", "trainable_normalization": True}``
    without any change to n3fit.  n3fit never interprets ``options``; it validates it
    against the declared schema, serializes it and reports it.

    ``seeds`` holds one seed per ensemble member; it is only meaningful for stochastic
    parametrizations, and reproducing the current per-layer seeding rules is the
    backend's responsibility (pinned down by the conformance suite).

    Everything *around* the core (input transform, preprocessing factor, basis rotation,
    sum rules, photon) is n3fit-owned and identical for every ``kind``: swapping the
    parametrization does not re-open the PDF-level chain.
    """

    kind: str
    options: Mapping[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class ObjectiveSpec:
    """The optimization metric (D1: implemented by the backend, described by this spec).

    ``kind`` is one of :attr:`Capabilities.objectives`, which declares a *schema* per kind
    (which ``data`` arrays, which ``set_scalar`` scalars, which build-time ``options``), so n3fit
    can validate a requested metric against a backend instead of against a hard-coded list.
    ``data`` carries everything the term needs at build time (``invcovmat``, ``covmat``,
    ``target``, ...) and ``mask`` the per-replica training/validation/fold mask, if any.

    ``options`` are build-time knobs (e.g. positivity's ``alpha``); values that *change during a
    fit* are not options — they are declared in the schema under ``scalars`` and set through
    :meth:`Objective.set_scalar` (e.g. the Lagrange multiplier pushed every N steps).  One
    documented exception, because it is exactly what the Lagrange multiplier needs: ``options`` may
    carry the *initial value* of a scalar the schema declares (an initial value is a build-time
    decision; the schedule then moves the scalar).  A name that is neither a declared option nor a
    declared scalar is an error, not a silently ignored key.
    """

    kind: str
    name: str
    data: Mapping[str, Array] = field(default_factory=dict)
    mask: Array | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    prediction: str | None = None
    """Which graph output the term is applied to (P4 amendment).

    Since P4 the models output *predictions* and the optimizer applies the terms, so a term has to
    say what it measures.  ``None`` means "the output of the same name", which is the common case
    (every chi2 term is applied to its own dataset's prediction, and n3fit names the loss after the
    dataset).  Two terms may name the same output when they genuinely measure the same prediction --
    positivity and integrability are both applied to the masked PDF, and before P4 they were two
    loss layers fed by one tensor.
    """


@dataclass(frozen=True)
class ObjectiveGroup:
    """Which terms compose a group (P3).

    Built by n3fit -- it knows which losses belong to the training objective, the validation
    objective and the "true" chi2 -- and consumed by the optimizer, which never decides
    membership.  A term may appear in more than one group (the positivity terms are in
    ``training`` and in their own reporting group), which is why this is a mapping from group
    name to term names and not a partition.

    P3 materializes it and uses it for construction and reporting; P4 hands it to
    ``Optimizer.run``/``evaluate``.
    """

    terms: Mapping[str, Sequence[str]]
    """``{"training": ("LHC", "POS_PV", ...), "validation": (...), ...}``"""

    def names(self, group: str) -> tuple[str, ...]:
        """The term names of ``group`` (empty if the group has none -- a fit without positivity)."""
        return tuple(self.terms.get(group, ()))

    def groups_of(self, term: str) -> tuple[str, ...]:
        """Every group ``term`` belongs to, in the order the groups were declared."""
        return tuple(name for name, members in self.terms.items() if term in members)

    def __contains__(self, term: str) -> bool:
        return any(term in members for members in self.terms.values())

    @property
    def all_terms(self) -> tuple[str, ...]:
        """Every term name, each once, in first-seen order."""
        seen = {}
        for members in self.terms.values():
            for term in members:
                seen.setdefault(term, None)
        return tuple(seen)


@dataclass(frozen=True)
class OptimizerSpec:
    """How the objective is minimized (D10).

    ``name`` is a key of :attr:`Capabilities.optimizers`, which declares the accepted
    ``options``, whether the optimizer is iterative, and what it requires (a gradient, a
    Jacobian/Hessian, a linear solve, a parametrization linear in its parameters, ...).

    This replaces the previous ``OptimizerSpec``: first-order optimizers are simply one
    family of optimizers, so ``OptimizerSpec("adam", {"learning_rate": 0.01, "clipnorm": 1.0})``
    and ``OptimizerSpec("levenberg_marquardt", {"max_iter": 200})`` and
    ``OptimizerSpec("linear_least_squares", {})`` are all expressible.  Differentiation
    itself is *not* part of the contract: it is backend-internal, and the contract only
    carries the declared requirements so that an impossible combination (e.g. a direct
    linear solve for a parametrization that is not linear in its parameters) is rejected
    by :meth:`Backend.check_feasible` before anything is built.
    """

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class FitResult:
    """What an optimizer returns.

    ``parameters`` keeps the per-member convention (one :data:`WeightMap` per ensemble
    member).  ``covariance`` is optional and is the second thing a Hessian method is
    after: it is the parameter covariance in the *flattened* parameter space, whose
    ordering is given by ``diagnostics["parameter_layout"]`` (the ordered list of weight
    paths).  MC replica training leaves it ``None``; a linear/Hessian fit returns it and
    it is what generates eigenvector sets downstream.
    """

    parameters: list[WeightMap]
    covariance: Array | None = None
    history: "History | None" = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Graph objects
# --------------------------------------------------------------------------------------
@runtime_checkable
class Tensor(Protocol):
    """Opaque backend tensor.

    Tensors are *never* seen by n3fit code outside the layer definitions, and never
    appear in the signature of any method defined in this module.
    """


@runtime_checkable
class Layer(Protocol):
    """A callable taking tensors to tensors.

    n3fit's physics layers (FK convolutions, rotations, preprocessing, sum rules, ...) are
    written exclusively against :class:`Ops`, so that they can run on any backend.
    """

    def __call__(self, *inputs: Tensor) -> Tensor: ...


@runtime_checkable
class Ops(Protocol):
    """The primitive operation set the backend must provide.

    Every n3fit custom layer (FK convolution, rotations, preprocessing, sum rules, masks,
    x-operations) is written against these and nothing else, so this list *is* the
    physics/backend boundary.  Names and numpy semantics must match, which is what
    ``tests/backend_conformance/test_ops.py`` pins down (today: ``test_backend.py``).
    """

    # tensor algebra
    def einsum(self, subscripts: str, *tensors: Tensor) -> Tensor: ...
    def tensordot(self, a: Tensor, b: Tensor, axes) -> Tensor: ...
    def matmul(self, a: Tensor, b: Tensor) -> Tensor: ...
    def sum(self, x: Tensor, axis=None, keepdims: bool = False) -> Tensor: ...
    def nansum(self, x: Tensor, axis=None) -> Tensor:
        """Like ``numpy.nansum``: NaN treated as 0, inf as a very large number."""

    def nan_to_num(self, x: Tensor) -> Tensor: ...

    # elementwise / activations
    def pow(self, x: Tensor, y) -> Tensor: ...
    def log(self, x: Tensor) -> Tensor: ...
    def multiply(self, x: Tensor, y: Tensor) -> Tensor:
        """Elementwise product.  (Added in round 9: the layers need it, and spelling it as
        ``einsum`` at every call site would be unreadable.)"""

    def subtract(self, x: Tensor, y: Tensor) -> Tensor:
        """Elementwise difference (see ``multiply``)."""

    def clip(self, x: Tensor, lo, hi) -> Tensor: ...
    def absolute(self, x: Tensor) -> Tensor: ...
    def tanh(self, x: Tensor) -> Tensor: ...
    def elu(self, x: Tensor, alpha: float = 1.0) -> Tensor: ...

    # shape manipulation
    def reshape(self, x: Tensor, shape) -> Tensor: ...
    def transpose(self, x: Tensor, axes) -> Tensor: ...
    def expand_dims(self, x: Tensor, axis: int) -> Tensor: ...
    def concatenate(self, xs: Sequence[Tensor], axis: int) -> Tensor: ...
    def stack(self, xs: Sequence[Tensor], axis: int = 0) -> Tensor: ...
    def split(self, x: Tensor, indices: Sequence[int], axis: int) -> Sequence[Tensor]: ...
    def gather(self, x: Tensor, indices, axis: int = 0) -> Tensor: ...
    def scatter_update(self, x: Tensor, indices, values) -> Tensor: ...
    def repeat(self, x: Tensor, n: int, axis: int = 0) -> Tensor: ...

    # creation / conversion
    def constant(self, value: Array, dtype: str | None = None) -> Tensor: ...
    def zeros(self, shape) -> Tensor: ...
    def ones(self, shape) -> Tensor: ...
    def cast(self, x: Tensor, dtype: str) -> Tensor: ...
    def to_numpy(self, x: Tensor) -> Array: ...

    # layer generation helpers
    def as_layer(self, fn: Callable[..., Tensor], **kwargs) -> Layer:
        """Wrap a python function on tensors into a :class:`Layer`."""

    def as_input(self, value: Array, name: str | None = None) -> Tensor:
        """Create an input slot carrying a constant value (may be re-bound later)."""

    def splitter(self, shape, sizes: Sequence[int], axis: int, name: str) -> Layer:
        """Layer splitting a tensor of ``shape`` into chunks of ``sizes`` along ``axis``."""


@runtime_checkable
class WeightsView(Protocol):
    """A mutable handle on the weights of one model (one replica)."""

    def get(self, role: str | None = None) -> WeightMap:
        """Return a copy of the weights, optionally restricted to one role."""

    def assign(self, path: str, value: Array) -> None:
        """Set a single weight in place (semantics as ``variable.assign``)."""

    def update(self, values: WeightMap) -> None:
        """Set several weights in place."""


@runtime_checkable
class Model(Protocol):
    """A differentiable function built by the backend and driven by n3fit.

    **A model maps inputs to a prediction** -- a PDF, an observable, or whatever the model is
    for.  It does *not* return a loss: objectives are separate objects applied to predictions
    (:class:`Objective.__call__`).  The current Keras training models violate this (their output
    is the loss, by construction -- see the note in :class:`Objective`); that is legacy of the
    loop that reuses Keras' ``fit``, it is P4's job to remove, and a new backend must not copy it.

    A model is always single-replica; ensembles are requested through
    :meth:`Backend.ensemble`.
    """

    shapes: ShapeSpec

    def __call__(self, inputs: Mapping[str, Array] | None = None) -> Array:
        """Evaluate on numpy arrays and return numpy arrays (inference only).

        ``inputs`` only needs to contain the slots which are not bound constants; see
        :meth:`bind_input`.  ``None`` evaluates the graph as it was built — every input is
        bound, which is the case for the diagnostics that read a model back out of a fit.

        The output keeps the shape the graph produces (a leading batch or replica axis is
        **not** squeezed away): this replaces ``MetaModel.predict`` and must be a drop-in for
        it, indexing included.
        """

    def weights(self, role: str | None = None) -> WeightMap: ...

    def parameters(self, role: str | None = None) -> WeightMap:
        """The *trainable* weights only, in the order of ``parameter_layout``.

        This is the space in which parameter-space optimizers (Gauss-Newton/Levenberg-Marquardt,
        direct solves) and Hessian error propagation work.  The convention is: ``theta`` is the
        concatenation of ``parameters()`` in layout order, each array flattened in C order, and
        ``FitResult.diagnostics["parameter_layout"]`` names those paths.  n3fit owns the
        (pure-numpy) flatten/unflatten helpers; the backend guarantees that the layout it
        reports matches this mapping.
        """

    def set_weights(self, values: WeightMap) -> None: ...

    def weights_view(self) -> WeightsView:
        """Live handle used by hooks to mutate masks/multipliers during training."""

    def bind_input(self, name: str, value: Array) -> None:
        """Bind a constant numpy array to an input slot, replacing any previous binding.

        This is the mechanism behind the photon (``AddPhoton.register_photon`` +
        re-``compile``) and, more generally, behind every fixed input grid.
        """

    def bound_inputs(self) -> Mapping[str, Array]:
        """The constant inputs currently bound to the graph, as numpy arrays.

        The read side of :meth:`bind_input`: a model may carry inputs whose value is part of
        the graph rather than of the data (the x grid, an integration grid, the photon
        contribution).  n3fit needs to read them back for diagnostics -- the hyperopt "future
        tests" metric evaluates a PDF set on the fit's own x grid.
        """

    def override(self, role: str, fn: Callable[[Mapping[str, Array]], Array]) -> None:
        """Replace a section of the graph by a fixed function of the inputs.

        Needed by the "future tests" hyperopt metric, which evaluates the experimental chi2 of
        a replica *around the central value* of a PDF set.  The section is the PDF model itself,
        which the generators nest under the name ``PDFs`` (it was ``PDF_0`` until 2023, when the
        rename missed this call site); the Keras adapter overrides its ``call``.
        """

    def freeze(self) -> None:
        """Make the model permanently non-trainable."""

    def summary(self) -> None:
        """Human-readable description (replaces the ``get_layer(...).summary()`` chain)."""


@runtime_checkable
class Objective(Protocol):
    """A differentiable term of the loss, built from an :class:`ObjectiveSpec`.

    **A term is a function of the model's prediction**, not a part of the model::

        term(prediction) -> Array        # the term's value, shaped as in ``set_data``'s arrays
                                         # (one entry per replica where the term has a replica axis)

    This is the piece of the contract a backend must not improvise: today the Keras
    implementation *is* the last layer of the training graph, which is why the model's output is
    itself a loss.  A backend may fuse term and graph internally -- that is an implementation
    choice -- but the interface it presents is this one, and ``Model`` presents predictions
    (see :class:`Model`).  It is what makes ``Optimizer.jacobian``/``hessian`` meaningful: they
    differentiate the *terms* with respect to ``theta``, so the residual structure must survive.
    """

    spec: ObjectiveSpec

    def __call__(self, prediction: Array) -> Array:
        """The term's value for a model prediction (numpy in, numpy out)."""

    def apply(self, prediction: Tensor) -> Tensor:
        """The same term, applied to a prediction *inside a graph* (tensor in, tensor out).

        The two spellings are the same operation in the two modes the interface distinguishes
        everywhere else: ``__call__`` evaluates, ``apply`` builds.  n3fit needs the second one
        while it still assembles the training graph itself (P3); the backend needs it internally
        when it builds a step (P4).  A backend that fuses the term into its own machinery
        implements both by the route it prefers -- the contract only fixes what comes out.
        """

    def set_data(self, **arrays: Array) -> None:
        """Replace part of the term's data (fold masks, the covariance, ...).

        Semantics for the covariance (the one case n3fit would otherwise guess wrong):
        ``set_data(covmat=C)`` means *"this is the covariance the term should use"* -- the term
        recomputes whatever it needs from it.  A caller adding a piece to the covariance it
        already holds passes the sum (:meth:`Objective.spec`'s ``data["covmat"]`` is readable for
        that purpose); the term does not add on its own.
        """

    def set_scalar(self, name: str, value: float) -> None:
        """Set a single scalar, e.g. the positivity/integrability Lagrange multiplier.

        Declared per kind in :attr:`Capabilities.objectives`.  This replaces the legacy trick of
        mutating a non-trainable weight (``LagrangeCallback`` multiplying ``lagMult`` in place).
        """

    def scalar(self, name: str) -> float:
        """The current value of a declared scalar -- the read side of :meth:`set_scalar`.

        The Lagrange schedule needs it: the legacy callback *scaled* the weight in place (so it
        never had to read it), a hook computes the next value and must start from the current one.
        """


@runtime_checkable
class Ensemble(Protocol):
    """The models of a fit: the replicas of the trained graph, plus the role graphs.

    P4 amendment: ``model(role)``.  A fit is not one graph -- n3fit trains a *training* graph and
    evaluates two others (*validation*, *experimental*), each with its own masks and data, and the
    engine has to be able to evaluate them (``StepContext.evaluate("validation")``).  Before P4 the
    three graphs only met inside ``model_trainer``; the engine needs them, so they travel with the
    ensemble.  ``__iter__``/``weights`` refer to the *training* graph, which is the one being
    trained.

    Roles are the n3fit-side names (``"training"``, ``"validation"``, ``"experimental"``), the same
    strings :data:`GROUP_TRAINING` and friends carry.
    """

    shapes: ShapeSpec

    def model(self, role: str = GROUP_TRAINING) -> Model:
        """The role's graph, as a contract :class:`Model` (inference API)."""

    def __iter__(self) -> Iterator[Model]:
        """The replicas of the training graph, in order (replica ``i`` of n3fit is ``ens[i - 1]``).

        This may be expensive (the Keras adapter builds one single-replica graph per replica);
        the engine must not need it, only callers that want a per-replica model.
        """

    def __getitem__(self, replica: int) -> Model: ...

    def __len__(self) -> int: ...

    def weights(self, role: str | None = None) -> list[WeightMap]: ...

    def set_weights(self, values: Sequence[WeightMap]) -> None: ...


# --------------------------------------------------------------------------------------
# The training engine
# --------------------------------------------------------------------------------------
@dataclass
class StepContext:
    """Everything a hook may see at a monitored step."""

    step: int
    """Global optimizer-step index (not a monitored-step index)."""

    logs: Mapping[str, Array]
    """Per-term, per-replica losses of the monitored group, computed *before* the update
    of this step (the convention the current Keras path implements through
    ``CallbackStep.correct_logs``)."""

    weights: WeightsView
    """Live weights of the model being trained."""

    evaluate: Callable[[str], Mapping[str, Array]]
    """Request the evaluation of a named group ("validation", "experimental", ...)."""

    stop: Callable[[], None]
    """Ask the engine to stop training after this monitored step."""


@runtime_checkable
class Hook(Protocol):
    """User-side logic injected into the training loop.

    Replaces ``StoppingCallback``/``LagrangeCallback``/``TimerCallback``.

    A hook that needs the *graph* rather than the weights (tensorboard does: it writes a model
    summary and needs a handle on the model it is logging) may also implement ``on_train_start``;
    the engine calls it once, before the first step, for any hook that has it.  It is deliberately
    not a mandatory member: the stopping rule, the Lagrange schedule and the timing hook are all
    functions of the step context alone, and most hooks should stay that way.
    """

    def on_monitored_step(self, ctx: StepContext) -> None: ...

    def on_train_start(self, ensemble) -> None: ...  # optional (duck-typed)

    def on_train_end(self) -> None: ...


class History(Protocol):
    """The record of a training run."""

    monitored_steps: list[int]
    losses: Mapping[str, list[Array]]


@runtime_checkable
class Optimizer(Protocol):
    """How the objective is minimized: SGD-family, second-order, or an exact solve.

    Replaces the ``Engine`` concept: ``Optimizer.run`` subsumes what ``Engine.train`` did for
    iterative first-order methods, and extends it to optimizers with different loop semantics
    (D10).  For a non-iterative optimizer (``is_iterative()`` false, e.g. a direct linear
    least-squares solve) there is no loop: ``steps`` and ``monitor_every`` must be ``None``,
    hooks fire at most once through :meth:`Hook.on_train_end`, stopping does not apply, and
    the work happens inside :meth:`run`.
    """

    def is_iterative(self) -> bool: ...

    def min_monitor_interval(self) -> int:
        """Smallest ``monitor_every`` the backend can honour (1 for the current
        backends).  n3fit uses ``max(this, settings.monitor_every)``."""

    def run(
        self,
        ensemble: Ensemble,
        terms: Mapping[str, Objective],
        groups: Mapping[str, Sequence[str]],
        *,
        steps: int | None,
        monitor_every: int | None,
        hooks: Sequence[Hook] = (),
    ) -> FitResult:
        """Minimize the objective.

        ``steps`` = number of optimizer updates per ensemble member (``None`` for
        non-iterative optimizers).  ``monitor_every=None`` disables monitoring entirely: no
        hooks, no evaluation, no history during the run (this is the "run the whole budget
        then evaluate once" mode that lets a jit-compiled backend fuse the whole loop, and
        it is the natural mode for fixed-budget fits and for benchmark runs).

        ``groups`` maps a group name (``training``/``validation``/``experimental``) to the
        terms that compose it; membership is n3fit's decision, exactly as the current code
        decides which losses enter the training/validation/experimental models.
        """

    def evaluate(
        self,
        ensemble: Ensemble,
        terms: Mapping[str, Objective],
        group: str,
    ) -> Mapping[str, Array]:
        """Evaluate a group without training, returning per-term, per-member values."""

    def jacobian(
        self,
        model: Model,
        terms: Mapping[str, Objective],
        group: str = "training",
    ) -> Mapping[str, Array] | None:
        """``d(predictions)/d theta`` for each term of ``group``, at the current weights.

        Returns one ``(n_points, n_params)`` array per term name, ordered according to
        :meth:`Model.parameters` / ``parameter_layout``.  Combined with the terms' weights
        (their ``invcovmat``) this gives the Gauss-Newton Hessian ``J^T W J``, hence the
        parameter covariance that Hessian *error propagation* needs -- the alternative to MC
        replicas, and the same object a Gauss-Newton/Levenberg-Marquardt optimizer builds
        internally.

        Optional: gated by ``Capabilities.derivatives``.  Backends that cannot differentiate
        return ``None`` (and n3fit reports "this backend cannot do Hessian error propagation").
        """

    def hessian(
        self,
        model: Model,
        terms: Mapping[str, Objective],
        group: str = "training",
        *,
        approximation: str = "gauss_newton",
    ) -> Array | None:
        """``d^2 chi2 / d theta^2``, optionally under a named approximation.

        ``"gauss_newton"`` (the default, and what LM uses) must be derivable from
        :meth:`jacobian`; ``"exact"`` requires true second derivatives and may be indefinite.
        Provided so that a backend can avoid materialising a large ``J`` when it has a
        cheaper route to the same matrix.
        """


# --------------------------------------------------------------------------------------
# Capabilities, state, backend
# --------------------------------------------------------------------------------------
@runtime_checkable
class Capabilities(Protocol):
    """What this backend can do.  This replaces the hard-coded ``accepted_optimizers`` /
    ``initializers`` / ``layers`` dicts that ``checks.py`` and ``hyper_scan.py`` read today.

    ``parametrizations`` and ``optimizers`` are *registries with schemas*: the backend
    declares which kinds exist, the accepted options (names, types, defaults) and, for
    optimizers, whether they are iterative and what they require.  n3fit validates user input
    against these maps, which is how a polynomial parametrization or a second-order optimizer
    becomes usable without touching n3fit.
    """

    parametrizations: Mapping[str, Mapping[str, Any]]
    """``{"dense": {"options": {...}, ...}, "polynomial": {"options": {...},
    "linear_in_parameters": True}, ...}``"""

    optimizers: Mapping[str, Mapping[str, Any]]
    """``{"adam": {"options": {...}, "is_iterative": True, "requires": {"gradient"},
    "uses_validation_stopping": True},
    "levenberg_marquardt": {"is_iterative": True, "requires": {"jacobian"},
    "uses_validation_stopping": False},
    "linear_least_squares": {"is_iterative": False,
    "requires": {"linear_in_parameters"}}, ...}``

    ``uses_validation_stopping`` is what tells n3fit whether to install the stopping hook
    at all: an optimizer with its own convergence criterion (Gauss-Newton/Levenberg-Marquardt,
    direct solves) has a monotonically decreasing objective, so cross-validation stopping is
    meaningless.  n3fit then uses ``steps`` as a plain iteration budget and installs no
    stopping hook -- but may still monitor, since ``monitor_every`` is independent of it.
    """

    objectives: Mapping[str, Mapping[str, Any]]
    """``{"chi2": {"data": ("invcovmat", "covmat", "target"), "scalars": (),
    "options": ()},
    "positivity": {"data": (), "scalars": ("multiplier",), "options": ("alpha",)},
    "integrability": {"data": (), "scalars": ("multiplier",), "options": ()}, ...}``

    Like ``parametrizations``/``optimizers``, a registry with schemas rather than a set of names:
    ``data`` are the arrays an :class:`ObjectiveSpec` must carry, ``scalars`` the names
    :meth:`Objective.set_scalar` accepts, ``options`` the build-time knobs.  A backend that wants
    a new metric declares its kind here and implements :class:`Objective`; n3fit then accepts the
    kind without changes.
    """

    derivatives: frozenset[str]
    """Which derivative objects the backend can produce, any of ``{"gradient", "jacobian",
    "hessian"}`` (all with respect to ``theta``, see :meth:`Optimizer.jacobian`).  Solvers
    declare their own needs separately in :attr:`optimizers`; this is what tells n3fit whether
    Hessian error propagation is available at all."""

    initializers: Mapping[str, Mapping[str, Any]]
    activations: frozenset[str]
    regularizers: frozenset[str]
    dtypes: frozenset[str]
    train_n_replicas_together: bool
    supports_weight_mutation: bool
    supports_tensorboard: bool
    """Whether the backend can drive tensorboard (Keras: only on its tensorflow backend)."""

    fast_single_replica_convolution: bool

    requires_eager_workaround: bool
    """Whether the backend/version needs eager execution switched on around model building
    that happens *outside* a training loop -- the historical ``tensorflow`` < 2.4 workaround
    in the future-test diagnostic (``hyper_optimization.rewards``, whose ``fit_future_tests``
    was deleted in P4 -- this flag stays because the *question* is the backend's).  A backend
    that does not need it
    declares ``False`` and the workaround simply does not run, instead of n3fit sniffing the
    framework version itself."""


@runtime_checkable
class BackendState(Protocol):
    """Global state of the backend process (replaces ``n3fit.backends.internal_state``)."""

    def configure(
        self,
        *,
        dtype: str = "float32",
        threads: int | None = None,
        seed: int | None = None,
        deterministic: bool = False,
        eager: bool | None = None,
        max_cores: int | None = None,
    ) -> None: ...

    def clear(self) -> None:
        """Release state between fits (hyperopt trials, k-folds)."""

    def set_eager(self, enabled: bool) -> None:
        """Turn eager (step-by-step) execution on or off.

        A no-op for backends that are eager by definition.  ``configure(eager=...)`` is the
        same switch expressed as part of the initial setup; this method exists because the
        workaround that needs it (see ``Capabilities.requires_eager_workaround``) toggles it
        in the middle of a run, and re-running the whole configuration to do so would be
        both surprising and destructive.
        """

    def devices(self) -> list[str]:
        """Available compute devices (used by the parallel hyperopt scheduler)."""


@runtime_checkable
class Backend(Protocol):
    """The single object n3fit talks to."""

    name: str
    version: str
    ops: Ops
    """The primitive operation set (see ``Ops`` in the implementation); n3fit's custom
    layers are written against it."""

    capabilities: Capabilities
    state: BackendState

    def parametrization(self, spec: ParametrizationSpec, shapes: ShapeSpec) -> Model:
        """Build the trainable core (one ensemble member) from its spec.

        This is where ``_generate_nn``'s ``if architecture == ...`` and the layer
        vocabulary (``dense``/``dense_per_flavour``/``lstm``/activations/initializers)
        move to.  A ``polynomial`` kind has no ``nodes``: it returns a model whose
        parameters are the polynomial coefficients.
        """

    def model(
        self,
        shapes: ShapeSpec,
        inputs: Mapping[str, Tensor],
        outputs: Tensor,
        *,
        name: str,
        roles: Mapping[str, Sequence[str]] | None = None,
    ) -> Model: ...

    def view(self, graph) -> Model:
        """Expose a graph this backend has already produced through the :class:`Model` API.

        A migration bridge, and the reason P2 can land before P3: today n3fit builds its graphs
        through the legacy generators and then needs role-based access to them
        (``weights(role=...)``, ``bind_input``, ``override``, ``freeze``, ``summary``).  Once P3
        makes :meth:`model` the primary constructor, ``view`` remains what it is here -- a way to
        talk about an existing graph without knowing what produced it.

        The object returned is a *view*: it must mutate the graph it wraps in place, not copy it
        (``override`` and ``bind_input`` are only useful if the caller's model changes).
        """

    def objective(self, spec: ObjectiveSpec) -> Objective: ...

    def ensemble(
        self, models: "Sequence[Model] | Model", strategy: str | None = None
    ) -> Ensemble:
        """The replicas of ``models``, as individual models (D2).

        ``models`` is either a sequence of models or **a single model whose graph carries the
        replicas stacked** — the legacy shape, which the backend splits the same way
        ``MetaModel.split_replicas`` did.  Iteration order is the replica order; n3fit counts
        replicas from 1 and indexes the ensemble from 0.

        ``strategy`` names how the replicas share work (independent, vmap, a shared trunk); it
        is the backend's business and unused until P4 gives it meaning.
        """

    def optimizer(self, spec: OptimizerSpec) -> Optimizer: ...

    def check_feasible(
        self,
        parametrization: ParametrizationSpec,
        optimizer: OptimizerSpec,
        objective: ObjectiveSpec,
    ) -> None:
        """Raise if this combination cannot be built (e.g. a direct linear solve with a
        parametrization that is not linear in its parameters, or a second-order optimizer with
        an objective that is not twice differentiable).  Called by n3fit at validation
        time, so that unsupported combinations fail with a clear message instead of deep
        inside the backend."""

    def save(self, model, path) -> None:
        """Write the weights of one replica plus a manifest, in the backend-neutral schema.

        The schema is one replica per file: ``model`` may be an ensemble holding exactly one
        replica, a single :class:`Model`, or a raw graph; holding more than one replica is an
        error (save the replicas one file each).
        """

    def load(self, ensemble: Ensemble, path, replica: int | None = None) -> None:
        """Read a weight file back into an ensemble built from the same specs.

        ``replica=None`` broadcasts the file's replica into every replica of the ensemble (the
        legacy ``load:`` warm-start semantics); ``replica=i`` fills only that replica (the
        ``load_weights_from_fit`` semantics).  A file whose layout does not match the ensemble
        is an error, never a partial load.
        """

    def version_info(self) -> Mapping[str, str]:
        """Library versions, for the fit output (replaces ``writer.version()``)."""


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def get_backend(name: str | None = None) -> Backend:
    """Return the backend selected by ``name``, ``$N3FIT_BACKEND`` or ``$KERAS_BACKEND``.

    Importing this module must never import a framework: the backend is imported lazily
    only when a backend is actually requested (so that ``n3fit.checks``, the parameter
    validation and the documentation can run without any of them installed).
    """
    raise NotImplementedError(
        "Implemented in n3fit/backends/__init__.py once the backends are registered"
    )
