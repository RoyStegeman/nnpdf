"""Give a graph output the name of the objective term that consumes it (P4).

Before P4 a model's outputs *were* the losses (one loss layer per term, ``_default_loss = nansum``
turned the list of outputs into a single number).  From P4 on the outputs are **predictions** and
the terms are applied by the engine, which has to know which output each term reads.  The engine
asks ``Optimizer`` for ``spec.prediction`` (or the term's own name when that is not given) and
resolves it against the graph's output names, so "the term called ``LHC`` consumes the output called
``LHC``" has to be true of the graph that gets built.

A graph is named by its layers, not by the tensors in it, so the prediction of each term goes
through this layer: numerically the identity (no weights, no arithmetic), and named after the term.

Two things it is *not*:

* not a re-labeling of the mask/rotation layer that happens to produce the prediction -- those names
  are the backend's business (``trmask_LHC``), and a term's prediction is not always the output of a
  mask layer (the experimental chi2 has no mask; two terms may share one prediction, in which case
  the spec says so with ``prediction`` and only one of them needs to own the name);
* not a place to put behaviour.  If it ever does arithmetic, the numbers move: the purpose of the
  layer is that the graph *without* it and the graph with it are the same function.
"""

from n3fit.backends import MetaLayer


class NamedOutput(MetaLayer):
    """Identity layer whose name is the name of the objective term that reads it.

    Parameters
    ----------
        **kwargs:
            passed to ``MetaLayer``; ``name`` is the term's name and is the whole point.
    """

    def call(self, x):
        """Return the input unchanged: this layer exists to be *called*, and to have a name."""
        return x
