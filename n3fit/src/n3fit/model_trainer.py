"""
The ModelTrainer class is the true driver around the n3fit code

This class is initialized with all information about the NN, inputs and outputs.
The construction of the NN and the fitting is performed at the same time when the
hyperparametrizable method of the function is called.

This allows to use hyperscanning libraries, that need to change the parameters of the network
between iterations while at the same time keeping the amount of redundant calls to a minimum
"""

from collections import namedtuple
from itertools import zip_longest
import logging

import numpy as np

from n3fit import model_gen
from n3fit.backends import (
    GROUP_EXPERIMENTAL,
    GROUP_INTEGRABILITY,
    GROUP_POSITIVITY,
    GROUP_TRAINING,
    GROUP_VALIDATION,
    MetaModel,
    ObjectiveGroup,
    OptimizerSpec,
    clear_backend_state,
    get_backend,
)
from n3fit.backends import operations as op
from n3fit.hyper_optimization.hyper_scan import HYPEROPT_STATUSES
import n3fit.hyper_optimization.penalties
from n3fit.hyper_optimization.rewards import HyperLoss
from n3fit.layers import losses
from n3fit.scaler import generate_scaler
from n3fit.stopping import FitRecord, LagrangeHook, LogHook, StoppingHook, _parse_chi2
from n3fit.vpinterface import N3PDF, compute_hyperopt_metrics
from validphys.convolution import central_predictions, predictions
from validphys.core import DataGroupSpec
from validphys.loader import Loader
from validphys.photon.compute import Photon

log = logging.getLogger(__name__)
l = Loader()
# Threshold defaults
# Any partition with a chi2 over the threshold will discard its hyperparameters
HYPER_THRESHOLD = 50.0
CHI2_THRESHOLD = 10.0
# Each how many epochs do we increase the positivitiy Lagrange Multiplier
PUSH_POSITIVITY_EACH = 100

# Each how many epochs do we increase the integrability Lagrange Multiplier
PUSH_INTEGRABILITY_EACH = 100

# Final number of flavours
FLAVOURS = 14

# See ModelTrainer::_xgrid_generation for the definition of each field and how they are generated
InputInfo = namedtuple("InputInfo", ["input", "split", "idx"])


def _pdf_injection(pdf_layers, observables, masks):
    """
    Takes as input a list of PDF layers each corresponding to one observable (also given as a list)
    And (where neded) a mask to select the output.
    Returns a list of ``(prediction, term)`` pairs.

    The pairing is the point (P4): one call of an observable wrapper produces *both* the tensor the
    role graph will output and the objective term over it, so the terms of the three roles cannot
    drift from the graphs that carry their predictions.
    Note that the list of masks don't need to be the same size as the list of layers/observables
    """
    return [f(x, mask=m) for f, x, m in zip_longest(observables, pdf_layers, masks)]


def _parametrization_spec(replicas_settings):
    """The :class:`ParametrizationSpec` of a fit, from the per-replica settings (P4).

    The kinds and options here are the ones n3fit passes to ``generate_pdf_model``; the name is the
    architecture (``dense``, ``dense_per_flavour``) and the seeds are the per-replica ones.  It is
    only used to ask the backend whether the combination is buildable (``check_feasible``), so it
    carries no state: the graph is still built by ``generate_pdf_model`` until P7 makes the
    parametrization a backend member.
    """
    from n3fit.backends.base import ParametrizationSpec

    first = replicas_settings[0]
    return ParametrizationSpec(
        kind=first.architecture,
        options={
            "nodes": tuple(first.nodes),
            "activations": tuple(first.activations),
            "initializer": first.initializer,
            "dropout_rate": first.dropout_rate,
        },
        seeds=tuple(settings.seed for settings in replicas_settings),
    )


def _ndata_for_terms(term_names, entries, key):
    """``{term name: ndata array}`` for the chi2 terms of one role, in the order they were built.

    ``entries`` is the reporting list ``_prepare_reporting`` produces; a dataset whose number of
    points is zero in every replica is left out, exactly as the legacy ``parse_ndata`` did (it
    dropped those entries, so their loss was not part of the chi2 either).  This is what a k-fold
    partition relies on.
    """
    ndata = {}
    for name, entry in zip(term_names, entries):
        points = np.asarray(entry[key])
        if points.sum() != 0:
            ndata[name] = points
    return ndata


def _LM_initial_and_multiplier(input_initial, input_multiplier, max_lambda, steps):
    """
    If any of input_initial or input_multiplier is None this function computes
    the missing values taking as input the maximum lambda multiplier and the number of steps needed
    to reach the maximum number of epochs
    """
    initial = input_initial
    multiplier = input_multiplier
    # If the multiplier is None, compute it from known values
    if multiplier is None:
        # If the initial value is also None, set it to one
        if initial is None:
            initial = 1.0
        multiplier = pow(max_lambda / initial, 1 / max(steps, 1))
    elif initial is None:
        # Select the necessary initial value to get to max_lambda after all steps
        initial = max_lambda / pow(multiplier, steps)
    return initial, multiplier


class ModelTrainer:
    """
    ModelTrainer Class:

    Wrapper around the fitting code and the generation of the Neural Network

    The ``hyperparametrizable`` method accepts a dictionary of hyper-parameters
    which defines the Neural Network.
    When it is called with a dictionary of parameters,
    it generates a NN and subsequentially performs a fit.

    The motivation behind this class is minimising the amount
    of redundant calls of each hyperopt run, in particular this allows to completely reset
    the NN at the beginning of each iteration reusing some of the previous work.
    """

    def __init__(
        self,
        experiments_data,
        exp_info,
        pos_info,
        integ_info,
        flavinfo,
        fitbasis,
        nnseeds,
        boundary_condition,
        debug=False,
        kfold_parameters=None,
        max_cores=None,
        model_file=None,
        sum_rules=None,
        theoryid=None,
        lux_params=None,
        replicas=None,
        trials=None,
        load_weights_dict=None,
    ):
        """
        Parameters
        ----------
            experiments_data: list
                list of `validphys.core.DataGroupSpec` containing experiments
            exp_info: list(list(dict))
                A list of dictionaries (once per experiment) for each replica.
                This dictionary contains the experimental data and inverse covmats seprated by
                training/validation as well as possible data transformations (e.g. diagonalization)
            pos_info: list
                list of dictionaries containing positivity sets, similar to ``exp_info``
                but all replicas equal
            integ_info: list
                list of dictionaries containing integrability sets, similar to ``pos_info``
            flavinfo: list(dict)
                the fitting::basis object from the runcard
            fitbasis: str
                the name of the basis in which the fit is being done, fitting::fitbasis
            nnseeds: list(int)
                seed used to initialise the NN for each model to be passed to model_gen
                generated by ``validphys.n3fit_data.replica_nnseed``, one per replica
            debug: bool
                flag to activate debug options
            kfold_parameters: dict
                parameters defining the kfolding method
            max_cores: int
                maximum number of cores available to the fitting routine
            model_file: str
                name of the (h5) file in which the final NN model will be saved in each replica
                if not given, the model is not saved.
            sum_rules: str
                which sum rules should be enabled (All, MSR, VSR, False), defaults to ALL
            theoryid: validphys.core.TheoryIDSpec
                object contining the theoryid that should be used to generate the photon
            lux_params: dict
                dictionary containing the params needed from LuxQED
                if not give, the photon is not generated
            replicas: list
                list with the replicas ids to be fitted
            trials: dict
                dictionary containing the trials defining the methodology
        """
        # Save all input information
        self.exp_info = list(exp_info)
        self.pos_info = [] if pos_info is None else pos_info
        self.integ_info = [] if integ_info is None else integ_info
        self.boundary_condition = boundary_condition
        self.flavinfo = flavinfo
        self.fitbasis = fitbasis
        self._nn_seeds = nnseeds
        self.debug = debug
        self.all_datasets = []
        self._scaler = None
        self.theoryid = theoryid
        self.lux_params = lux_params
        self.replicas = replicas
        self.experiments_data = experiments_data
        self.trials = trials

        # Initialise internal variables which define behaviour
        if debug:
            self.max_cores = 1
        else:
            self.max_cores = max_cores
        self.model_file = model_file
        self.load_weights_dict = load_weights_dict
        self.print_summary = True
        self.mode_hyperopt = False
        self.impose_sumrule = sum_rules
        self._hyperkeys = None
        if kfold_parameters is None:
            self.kpartitions = [None]
            self.hyper_threshold = None
        else:
            self.kpartitions = kfold_parameters["partitions"]
            self.hyper_threshold = kfold_parameters.get("threshold", HYPER_THRESHOLD)
            # if there are penalties enabled, set them up
            penalties = kfold_parameters.get("penalties", [])
            self.hyper_penalties = []
            for penalty in penalties:
                pen_fun = getattr(n3fit.hyper_optimization.penalties, penalty)
                self.hyper_penalties.append(pen_fun)
                log.info("Adding penalty: %s", penalty)
            # Check what is the hyperoptimization target function
            replica_statistic = kfold_parameters.get("replica_statistic", None)
            fold_statistic = kfold_parameters.get("fold_statistic", None)
            loss_type = kfold_parameters.get("loss_type", None)
            self._hyper_loss = HyperLoss(
                loss_type=loss_type,
                replica_statistic=replica_statistic,
                fold_statistic=fold_statistic,
                reduce_proportion=kfold_parameters.get("reduce_proportion", 0.85),
                penalties_in_loss=kfold_parameters.get("penalties_in_loss", False),
            )

        # Initialize the dictionaries which contain all fitting information
        self.input_list = []
        self.training = {
            "output": [],
            "expdata": [],
            "ndata": 0,
            "model": None,
            "posdatasets": [],
            "posmultipliers": [],
            "integdatasets": [],
            "integmultipliers": [],
            "folds": [],
            # P3: the names of the terms each model carries, in the order they were built --
            # reported by ``model_gen.observable_generator``, never re-derived here.
            "chi2_names": [],
            "penalty_names": [],
            # term name -> the multiplier the term was built with (what a k-fold reset puts back)
            "penalty_initials": {},
        }
        self.validation = {
            "output": [],
            "expdata": [],
            "ndata": 0,
            "model": None,
            "folds": [],
            "posdatasets": [],
            "chi2_names": [],
            "penalty_names": [],
        }
        self.experimental = {
            "output": [],
            "expdata": [],
            "ndata": 0,
            "model": None,
            "folds": [],
            "chi2_names": [],
        }
        #: Which terms compose each objective group (P3).  Filled by ``_generate_observables``
        #: from the names the generators returned; consumed by the optimizer (P4), the k-fold
        #: reset and the reporting.
        self.objective_groups = None
        self.tr_masks = []

        self._fill_the_dictionaries()

        if self.validation["ndata"] == 0:
            # If there is no validation, the validation chi2 = training chi2
            self.no_validation = True
            self.validation["expdata"] = self.training["expdata"]
        else:
            # Consider the validation only if there is validation (of course)
            self.no_validation = False

        # Hooks that only watch: timing (``debug``) and, if the runcard asked for it, tensorboard.
        # They are added to the fit's hook list and can neither change the fit nor stop it.
        self._diagnostic_hooks = []
        if debug:
            self._diagnostic_hooks.append(LogHook())

        # The fitted model, as the engine sees it: the flat term mapping, the role ensemble and
        # the optimizer that drove it (P4).  ``evaluate`` and the hyperopt metrics read these
        # after the fit, one forward pass per group.
        self.terms = {}
        self.ensemble = None
        self.optimizer = None
        self._chi2_reporting = []

    def set_hyperopt(self, hyperopt_on, keys=None):
        """Set hyperopt options on and off (mostly suppresses some printing)"""
        if keys is None:
            keys = []
        self._hyperkeys = keys
        if hyperopt_on:
            self.print_summary = False
            self.mode_hyperopt = True
        else:
            self.print_summary = True
            self.mode_hyperopt = False

    ###########################################################################
    # # Internal functions                                                    #
    # Never to be called from the dark and cold outside world                 #
    ###########################################################################
    def _fill_the_dictionaries(self):
        """
        This function fills the following dictionaries
            -``training``: data for the fit
            -``validation``: data which for the stopping
            -``experimental``: 'true' data, only used for reporting purposes
        with fixed information.

        Fixed information: information which will not change between different runs of the code.
        This information does not depend on the parameters of the fit at any stage
        and so it will remain unchanged between different runs of the hyperoptimizer.

        The aforementioned information corresponds to:
            - ``expdata``: experimental data
            - ``name``: names of the experiment
            - ``ndata``: number of experimental points
        """
        for exp_dict in self.exp_info[0]:
            self.training["expdata"].append(exp_dict["expdata"])
            self.validation["expdata"].append(exp_dict["expdata_vl"])
            self.experimental["expdata"].append(exp_dict["expdata_true"])

            self.training["folds"].append(exp_dict["folds"]["training"])
            self.validation["folds"].append(exp_dict["folds"]["validation"])
            self.experimental["folds"].append(exp_dict["folds"]["experimental"])

            nd_tr = exp_dict["ndata"]
            nd_vl = exp_dict["ndata_vl"]

            self.training["ndata"] += nd_tr
            self.validation["ndata"] += nd_vl
            self.experimental["ndata"] += nd_tr + nd_vl

            for dataset in exp_dict["datasets"]:
                self.all_datasets.append(dataset.name)
        self.all_datasets = set(self.all_datasets)

        for pos_dict in self.pos_info:
            self.training["expdata"].append(pos_dict["expdata"])
            self.training["posdatasets"].append(pos_dict["name"])
            self.validation["expdata"].append(pos_dict["expdata"])
            self.validation["posdatasets"].append(pos_dict["name"])

        for integ_dict in self.integ_info:
            self.training["expdata"].append(integ_dict["expdata"])
            self.training["integdatasets"].append(integ_dict["name"])

    def _xgrid_generation(self):
        """
        Generates the full x-grid pertaining to the complete set of observables to be fitted.

        To first approximation, the full x-grid is a concatenation of all x-grid requested by
        all fk-tables.

        In the case of pineappl models all fktables ask for the same grid in x
        and so the input can be simplified to be a single grid for all (or most) datasets.
        However, this is not a _strict_ requirement for pineappl and was not a requirement before
        so the solution below must be kept general enough.

        Detailed implementation of the union of xgrids:
            let's assume an input [x1, x1, x1, x2, x2, x3]
            where each xi is a different grid, this will be broken into two lists:
            [x1, x2, x3] (unique grids) and [0,0,0,1,1,2] (index of the grid per dataset)
            The pdf will then be evaluated to concatenate([x1,x2,x3]) and then split (x1, x2, x3)
            Then each of the experiment, looking at the indexes, will receive one of the 3 PDFs
            The decision whether two grids (x1 and x1) are really the same is decided below

        The necessary information to redistribute the x-grid is held by a ``InputInfo`` tuple
        which is returned by this function.

        Returns
        ------
            Instance of ``InputInfo`` containing the input information necessary for the PDF model:
            - input:
                backend input layer with an array attached which is a concatenation of the unique
                inputs of the Model
                two inputs are the same if and only if they have the same shape, values and order
            - split:
                backend layer which splits the aforementioned concatenation back into the separate
                unique inputs, to be applied after the PDF is called
            - idx:
                indices of the observables to which the split PDF must be distributed
        """
        log.info("Generating the input grid")

        inputs_unique = []
        inputs_idx = []
        for igrid in self.input_list:
            for idx, arr in enumerate(inputs_unique):
                if igrid.size == arr.size and np.allclose(igrid, arr):
                    inputs_idx.append(idx)
                    break
            else:
                inputs_idx.append(len(inputs_unique))
                inputs_unique.append(igrid)

        # Concatenate the unique inputs
        input_arr = np.concatenate(inputs_unique, axis=1).T
        if self._scaler:
            # Apply feature scaling if given
            input_arr = self._scaler(input_arr)
        input_layer = op.numpy_to_input(input_arr, name="pdf_input")

        # The PDF model is called with a concatenation of all inputs
        # however, each output layer might require a different subset, this is achieved by
        # splitting back the output
        # Input shape: (batch size, replicas, input array, flavours)
        ishape = (1, len(self.replicas), input_arr.shape[0], FLAVOURS)
        xsizes = [i.shape[1] for i in inputs_unique]
        sp_layer = op.tensor_splitter(ishape, xsizes, axis=2, name="splitter")

        return InputInfo(input_layer, sp_layer, inputs_idx)

    def _model_generation(self, xinput, pdf_model, partition, partition_idx):
        """
        Fills the three dictionaries (``training``, ``validation``, ``experimental``)
        with the ``model`` entry

        Compiles the validation and experimental models with fakes optimizers and learning rate
        as they are never trained, but this is needed by some backends
        in order to run evaluate on them.

        Compiles nothing and carries no loss: the models are *predictions* and the engine applies
        the terms (P4).

        Before entering this function we have the input of the model
        and a list of outputs, but they are not connected.
        This function connects inputs with outputs by injecting the PDF.
        At this point we have a PDF model that takes an input (1, None, 1)
        and outputs in return (1, none, 14).

        The injection of the PDF is done by concatenating all inputs and calling
        pdf_model on it.
        This in turn generates an output_layer that needs to be splitted for every experiment
        as we have a set of observable "functions" that each take (1, exp_xgrid_size, 14)
        and output (1, masked_ndata) where masked_ndata can be the training/validation
        or the experimental mask (in which cased masked_ndata == ndata).
        Several models can be fitted at once by passing a list of models with a shared input
        so that every mode receives the same input and the output will be concatenated at the end
        the final output of the model is then (1, None, 14, n) (with n=number of parallel models).

        Parameters
        ----------
            xinput: InputInfo
                a tuple containing the input layer (with all values of x), and the information
                (in the form of a splitting layer and a list of indices) to distribute
                the results of the PDF (PDF(xgrid)) among the different observables
            pdf_model: n3fit.backend.MetaModel
                a model that produces PDF values
            partition: dict
                Only active during k-folding, information about the partition to be fitted
            partition_idx: int
                Index of the partition

        Returns
        -------
            models: dict
                dict of MetaModels (prediction graphs, P4) for training, validation and
                experimental.  The objective terms are collected on ``self.terms``.
        """
        log.info("Generating the Model")

        # For multireplica fits:
        #   The trainable part of the n3fit framework is a concatenation of all PDF models
        # We apply the Model as Layers and save for later the model (full_pdf)
        full_model_input_dict, full_pdf = pdf_model.apply_as_layer({"pdf_input": xinput.input})

        split_pdf_unique = xinput.split(full_pdf)

        # Now reorganize the uniques PDF so that each experiment receives its corresponding PDF
        split_pdf = [split_pdf_unique[i] for i in xinput.idx]
        # If we are in a kfolding partition, select which datasets are out
        training_mask = validation_mask = experimental_mask = [None]
        if partition and partition["datasets"]:
            # If we want to overfit the fold, leave the training and validation masks as [None]
            # otherwise, use the mask generated for the fold.
            # The experimental model instead is always limited to the fold
            if not partition.get("overfit", False):
                training_mask = [i[partition_idx] for i in self.training["folds"]]
                validation_mask = [i[partition_idx] for i in self.validation["folds"]]
            experimental_mask = [i[partition_idx] for i in self.experimental["folds"]]

        # Training and validation leave out the kofld dataset
        # experiment leaves out the negation.
        #
        # Each call returns the ``(prediction, term)`` pair of one dataset in one role (P4): the
        # prediction goes into the graph, the term into the flat mapping the engine is handed.
        pairs_tr = _pdf_injection(split_pdf, self.training["output"], training_mask)
        training = MetaModel(full_model_input_dict, [prediction for prediction, _ in pairs_tr])

        # Validation skips integrability and the "true" chi2 skips also positivity,
        # so we must only use the corresponding subset of PDF functions
        val_pdfs = []
        exp_pdfs = []
        for partial_pdf, obs in zip(split_pdf, self.training["output"]):
            if not obs.positivity and not obs.integrability:
                val_pdfs.append(partial_pdf)
                exp_pdfs.append(partial_pdf)
            elif not obs.integrability and obs.positivity:
                val_pdfs.append(partial_pdf)

        # We don't want to included the integrablity in the validation
        pairs_vl = _pdf_injection(val_pdfs, self.validation["output"], validation_mask)
        validation = MetaModel(full_model_input_dict, [prediction for prediction, _ in pairs_vl])

        # Or the positivity in the total chi2
        pairs_ex = _pdf_injection(exp_pdfs, self.experimental["output"], experimental_mask)
        experimental = MetaModel(
            full_model_input_dict, [prediction for prediction, _ in pairs_ex]
        )

        # One flat ``{term name: Objective}`` mapping over the three roles -- the mapping the
        # engine is given.  Building it from the pairs (rather than from the graphs) is what makes
        # A6 enforceable: two roles cannot end up sharing a term object, and a name that is used
        # twice is caught here instead of silently routing a term to the wrong prediction.
        self.terms = {}
        for prediction, term in pairs_tr + pairs_vl + pairs_ex:
            if term.name in self.terms:
                raise RuntimeError(
                    f"the objective term {term.name!r} is used by more than one role: a term is "
                    f"identified by its name in the engine's flat mapping, so every role needs its "
                    f"own (e.g. 'POS' and 'POS_val')"
                )
            self.terms[term.name] = term

        # P3: each group must name exactly the terms its model carries (and in the same order).
        # The groups were built from the names the generators returned, so this only fires if the
        # two ever drift apart -- which is the failure mode that would otherwise surface as a
        # silently wrong loss in P4, when the names are what pairs an output with its term.
        for group, graph, pairs in (
            (GROUP_TRAINING, training, pairs_tr),
            (GROUP_VALIDATION, validation, pairs_vl),
            (GROUP_EXPERIMENTAL, experimental, pairs_ex),
        ):
            names = list(self.objective_groups.names(group))
            built = [term.name for _, term in pairs]
            if names != built:
                raise RuntimeError(
                    f"the {group!r} objective group members {names} are not the terms the "
                    f"{group!r} graph was built from ({built}); the groups and the models are out "
                    f"of sync"
                )
            # ... and a term must be *findable* in its graph, because that is how the engine routes
            # it to a prediction: by name (``{name: tensor}`` outputs, contract §1.15).  The names
            # here are the output names -- for a Keras graph, the names of the ``NamedOutput``
            # layers, which is what ``Model.output_names`` reports.
            outputs = list(getattr(graph, "output_names", ()))
            for term in (term for _, term in pairs):
                wanted = term.spec.prediction or term.name
                if outputs and wanted not in outputs:
                    raise RuntimeError(
                        f"the term {term.name!r} reads the prediction {wanted!r}, which is not an "
                        f"output of the {group!r} graph ({outputs})"
                    )

        if self.print_summary:
            # One role-aware summary instead of a chain of get_layer(...).summary() calls: which
            # sections exist (photon, sum rule, preprocessing) is the backend's business.
            get_backend().view(training).summary()

        models = {"training": training, "validation": validation, "experimental": experimental}

        return models

    def _reset_observables(self):
        """
        Resets the 'output' and 'losses' entries of all 3 dictionaries:
                            (``training``, ``validation``, ``experimental``)
        as well as the input_list
        this is necessary as these can either depend on the parametrization of the NN
        or be obliterated when/if the backend state is reset
        """
        self.input_list = []
        for key in ["output", "posmultipliers", "integmultipliers", "chi2_names", "penalty_names"]:
            self.training[key] = []
            self.validation[key] = []
            self.experimental[key] = []
        self.training["penalty_initials"] = {}
        # The terms (P4): a flat ``{name: Objective}`` mapping over the three roles, filled by
        # ``_model_generation`` from the very calls that build the graphs.  Reset here for the same
        # reason as the outputs: they belong to the models of one fold.
        self.terms = {}

    ############################################################################
    # # Parameterizable functions                                                #
    #                                                                          #
    # The functions defined in this block accept a 'params' dictionary which   #
    # defines the fit and the behaviours of the Neural Networks                #
    #                                                                          #
    # These are all called by the function hyperparamizable below              #
    # i.e., the most important function is hyperparametrizable, which is a     #
    # wrapper around all of these                                              #
    ############################################################################
    def _generate_observables(
        self,
        all_pos_multiplier,
        all_pos_initial,
        all_integ_multiplier,
        all_integ_initial,
        epochs,
        interpolation_points,
    ):
        """
        This functions fills the 3 dictionaries (training, validation, experimental)
        with the output layers and the loss functions
        It also fill the list of input tensors (input_list)

        The arguments of this function are used to define the initial positivity of the
        positivity observables and the multiplier to be applied at each step.

        Parameters
        ----------
            all_pos_multiplier: float, None
                multiplier to be applied to the positivity each ``PUSH_POSITIVITY_EACH`` epochs
            all_pos_initial: float, None
                initial value for the positivity lambda
            epochs: int
                total number of epochs for the run
        """

        # First reset the dictionaries
        self._reset_observables()
        log.info("Generating layers")

        # validphys has generated the self.exp_info information replica-by-replica
        # Here we transpose all information for convenience so that the loop over observables
        # and the vectorization over replicas is made explicit
        experiment_data = {
            "trmask": [],
            "vlmask": [],
            "expdata": [],
            "expdata_vl": [],
            "invcovmat": [],
            "invcovmat_vl": [],
        }

        # Loop over datasets
        for i in range(len(self.exp_info[0])):
            # Loop over data fields
            for key, value in experiment_data.items():
                replica_data = []
                # Loop over replicas
                for replica in self.exp_info:
                    if key in ["expdata", "expdata_vl"]:
                        # Save the data with shape (ndata) instead of (1, ndata)
                        replica_data.append(replica[i][key][0])
                    else:
                        replica_data.append(replica[i][key])
                # Stack
                value.append(np.stack(replica_data))

        # Now we need to loop over all dictionaries (First exp_info, then pos_info and integ_info)
        for i, exp_dict in enumerate(self.exp_info[0]):
            if not self.mode_hyperopt:
                log.info("Generating layers for experiment %s", exp_dict["name"])

            # Stacked tr-vl mask array for all replicas for this dataset
            exp_layer = model_gen.observable_generator(
                exp_dict,
                self.boundary_condition,
                training_mask_array=experiment_data["trmask"][i],
                validation_mask_array=experiment_data["vlmask"][i],
                training_data=experiment_data["expdata"][i],
                validation_data=experiment_data["expdata_vl"][i],
                invcovmat_tr=experiment_data["invcovmat"][i],
                invcovmat_vl=experiment_data["invcovmat_vl"][i],
                n_replicas=len(self.replicas),
            )

            # Save the input(s) corresponding to this experiment
            self.input_list.append(exp_layer["inputs"])

            # Now save the observable layer, the losses and the experimental data
            self.training["output"].append(exp_layer["output_tr"])
            self.validation["output"].append(exp_layer["output_vl"])
            self.experimental["output"].append(exp_layer["output"])

            # ... and the names of those terms, so that the objective groups below name the same
            # terms the models carry, in the same order
            self.training["chi2_names"].append(exp_layer["objective_tr"])
            self.validation["chi2_names"].append(exp_layer["objective_vl"])
            self.experimental["chi2_names"].append(exp_layer["objective_exp"])

        # Generate the positivity penalty
        for pos_dict in self.pos_info:
            if not self.mode_hyperopt:
                log.info("Generating positivity penalty for %s", pos_dict["name"])

            positivity_steps = int(epochs / PUSH_POSITIVITY_EACH)
            max_lambda = pos_dict["lambda"]

            pos_initial, pos_multiplier = _LM_initial_and_multiplier(
                all_pos_initial, all_pos_multiplier, max_lambda, positivity_steps
            )
            num_experiments = len(self.exp_info)
            replica_masks = np.stack([pos_dict["trmask"]] * num_experiments)
            training_data = np.stack([pos_dict["expdata"].flatten()] * num_experiments)

            pos_layer = model_gen.observable_generator(
                pos_dict,
                self.boundary_condition,
                positivity_initial=pos_initial,
                training_mask_array=replica_masks,
                training_data=training_data,
                validation_data=training_data,
                n_replicas=len(self.replicas),
            )
            # The input list is still common
            self.input_list.append(pos_layer["inputs"])

            # The positivity penalty is part of both the training and the validation objective,
            # as *two* terms with their own names (P4): the contract identifies a term by its name
            # and one name cannot live in two roles.  ``POS`` is the one the Lagrange schedule
            # scales, ``POS_val`` is the one the stopping rule's positivity check reads (the
            # legacy read the validation model's, which was built with the initial multiplier and
            # never scaled -- the callback looked the layer up in the training model).
            self.training["output"].append(pos_layer["output_tr"])
            self.validation["output"].append(pos_layer["output_vl"])

            self.training["posmultipliers"].append(pos_multiplier)
            # the term names, as the generator named them, and the multiplier they were built with
            pos_name = pos_layer["objective_tr"]
            self.training["penalty_names"].append(pos_name)
            self.validation["penalty_names"].append(pos_layer["objective_vl"])
            self.training["penalty_initials"][pos_name] = pos_initial

        # Finally generate the integrability penalty
        for integ_dict in self.integ_info:
            if not self.mode_hyperopt:
                log.info("Generating integrability penalty for %s", integ_dict["name"])

            integrability_steps = int(epochs / PUSH_INTEGRABILITY_EACH)
            max_lambda = integ_dict["lambda"]

            integ_initial, integ_multiplier = _LM_initial_and_multiplier(
                all_integ_initial, all_integ_multiplier, max_lambda, integrability_steps
            )

            integ_layer = model_gen.observable_generator(
                integ_dict,
                self.boundary_condition,
                positivity_initial=integ_initial,
                integrability=True,
                n_replicas=len(self.replicas),
            )
            # The input list is still common
            self.input_list.append(integ_layer["inputs"])

            # The integrability all falls to the training
            self.training["output"].append(integ_layer["output_tr"])
            self.training["integmultipliers"].append(integ_multiplier)
            integ_name = integ_layer["objective_tr"]
            self.training["penalty_names"].append(integ_name)
            self.training["penalty_initials"][integ_name] = integ_initial

        # The objective groups (P3).  Membership is decided *here*, by the code that built the
        # terms, and in the order the models carry them (chi2 first, then the penalties) -- P4
        # hands these names to the optimizer together with the terms, and the optimizer never
        # decides which loss is part of which objective.
        self.objective_groups = ObjectiveGroup(
            {
                GROUP_TRAINING: tuple(self.training["chi2_names"])
                + tuple(self.training["penalty_names"]),
                GROUP_VALIDATION: tuple(self.validation["chi2_names"])
                + tuple(self.validation["penalty_names"]),
                GROUP_EXPERIMENTAL: tuple(self.experimental["chi2_names"]),
                GROUP_POSITIVITY: tuple(self.training["posdatasets"]),
                GROUP_INTEGRABILITY: tuple(self.training["integdatasets"]),
            }
        )

        # Store a reference to the interpolator as self._scaler
        if interpolation_points:
            self._scaler = generate_scaler(self.input_list, interpolation_points)

    def _prepare_reporting(self, partition):
        """Parses the information received by the :py:class:`n3fit.ModelTrainer.ModelTrainer`
        to select the bits necessary for reporting the chi2.
        Receives the chi2 partition data to see whether any dataset is to be left out
        """
        reported_keys = ["name", "count_chi2", "positivity", "integrability"]
        reporting_list = []

        # Most of the information is shared among replicas, only ndata/ndata_vl
        # might change replica to replica and they need to be filled with care
        for idx, exp_dict in enumerate(self.exp_info[0]):
            # Fill in the keys that are equal across replicas
            reporting_dict = {k: exp_dict.get(k) for k in reported_keys}

            # Now loop over replicas to fill in all data points as a list
            list_ndata = []
            list_ndata_vl = []
            for replica in self.exp_info:
                replica_exp_dict = replica[idx]

                ndata = replica_exp_dict.get("ndata")
                ndata_vl = replica_exp_dict.get("ndata_vl")

                if partition:
                    # If we are in a k-fold partition, we need to remove the folded data
                    # from both the training and validation to avoid calculating the chi2 wrong
                    for dataset in replica_exp_dict["datasets"]:
                        if dataset in partition["datasets"]:
                            dataset_ndata = dataset["ndata"]
                            frac = dataset["frac"]
                            ndata -= int(dataset_ndata * frac)
                            ndata_vl -= int(dataset_ndata * (1 - frac))

                list_ndata.append(ndata)
                list_ndata_vl.append(ndata_vl)

            reporting_dict["ndata"] = list_ndata
            reporting_dict["ndata_vl"] = list_ndata_vl
            reporting_list.append(reporting_dict)

        for exp_dict in self.pos_info + self.integ_info:
            reporting_dict = {k: exp_dict.get(k) for k in reported_keys}
            reporting_dict["ndata"] = [exp_dict.get("ndata")]
            reporting_dict["ndata_vl"] = [exp_dict.get("ndata_vl")]
            reporting_list.append(reporting_dict)

        return reporting_list

    def _hyperopt_override(self, params):
        """Unrolls complicated hyperopt structures into very simple dictionaries"""
        # If the input contains all parameters, then that's your dictionary of hyperparameters
        hyperparameters = params.get("parameters")
        if hyperparameters is not None:
            return hyperparameters
        # Else, loop over all different keys and unroll the dictionaries within hyperparameters
        for hyperkey in self._hyperkeys:
            item = params[hyperkey]
            if isinstance(item, dict):
                params.update(item)
        return params

    def enable_tensorboard(self, logdir, weight_freq=0, profiling=False):
        """Ask the backend for a tensorboard hook and add it to the fit's diagnostics.

        Tensorboard is a *capability* (contract §1.10/A3), not a callback n3fit keeps a list of:
        the Keras backend can only offer it on its tensorflow backend, and a backend that cannot
        raises here instead of failing later inside the loop.

        Parameters
        ----------
            logdir: Path
                path where to save the tensorboard logs
            weight_freq: int
                frequency (in epochs) at which to save weight histograms
            profiling: bool
                flag to enable the tensorboard profiler
        """
        self._diagnostic_hooks.append(
            get_backend().tensorboard_hook(logdir, histogram_freq=weight_freq, profiling=profiling)
        )

    def evaluate(self):
        """The training, validation and experimental chi2 of the fitted model.

        One forward pass per group, terms applied by the engine (P4) -- n3fit no longer asks a
        model for its loss, because a model no longer *is* a loss.  The three numbers are the ones
        the fit output has always carried: the training chi2 restricted to the chi2 terms, the
        validation chi2 the stopping rule ended on, and the experimental chi2 per point of the
        (fold-aware) experimental set.

        Returns
        -------
            train_chi2: chi2 of the trainining set, per replica
            val_chi2 : chi2 of the validation set, per replica
            exp_chi2: chi2 of the experimental data, per replica
        """
        if not self.terms:
            raise RuntimeError("ModelTrainer.evaluate was called before any training")
        groups = dict(self.objective_groups.terms)
        if self.no_validation:
            groups[GROUP_VALIDATION] = groups[GROUP_TRAINING]
        train = self.optimizer.evaluate(self.ensemble, self.terms, GROUP_TRAINING)
        validation = self.optimizer.evaluate(self.ensemble, self.terms, GROUP_VALIDATION)
        experimental = self.optimizer.evaluate(self.ensemble, self.terms, GROUP_EXPERIMENTAL)
        # ``_parse_chi2`` is the engine's own arithmetic (sum over the named terms, divided by
        # their points), so the reported chi2 cannot drift from the one the stopping saw.
        train_chi2, _ = _parse_chi2(train, _ndata_for_terms(
            self.training["chi2_names"], self._chi2_reporting, "ndata"
        ))
        val_chi2, _ = _parse_chi2(validation, _ndata_for_terms(
            self.validation["chi2_names"], self._chi2_reporting, "ndata_vl"
        ))
        exp_chi2, _ = _parse_chi2(experimental, _ndata_for_terms(
            self.experimental["chi2_names"], self._chi2_reporting, "ndata"
        ))
        return train_chi2, val_chi2, exp_chi2

    def _filter_datagroupspec(self, datasets_partition, filter_in=True):
        """Takes a list of strings with dataset names to either filter in or out
        and returns instances of :class:`validphys.core.DataGroupSpec` which contain
        either only the "in" datasets or all datasets minus the "out".
        To control whether the dataset_partition should be selected or deselected
        the ``filter_in`` variable must be set to either True (select) or False (deselect)

        The use case of this function is to return a modified experiment group object
        following the same criteria that is used during the training, but with only
        a subset of datasets being considered.

        Parameters
        ----------
            datasets_partition: List[str]
                List with names of the datasets you want to select or deselect.
            filter_in: bool
                Whether the datasets should be selected in (True, default) or out (False)

        Parameters
        ----------
            datasets_partition: List[str]
                List with names of the datasets you want to select.

        Returns
        -------
            filtered_datagroupspec: List[validphys.core.DataGroupSpec]
                List of filtered exp datasets whose names are in datasets_partition.
        """
        filtered_datagroupspec = []

        # self.experiments_data is composed of a list of `DataGroupSpec` objects
        # These represent a group of related exp data sets
        # Loop over this list
        for datagroup in self.experiments_data:
            filtered_datasetspec = []

            # Each `DataGroupSpec` is composed by several `DataSetSpec` objects
            # `DataSetSpec` represents each exp dataset
            # Now, loop over them
            for dataset in datagroup.datasets:
                # Include `DataSetSpec`s whose names are in datasets_partition
                if (dataset.name in datasets_partition) == filter_in:
                    filtered_datasetspec.append(dataset)

            # List of filtered experiments as `DataGroupSpec`
            filtered_datagroupspec.append(
                DataGroupSpec(name=f"{datagroup.name}_exp", datasets=filtered_datasetspec)
            )

        return filtered_datagroupspec

    def hyperparametrizable(self, params):
        """
        Wrapper around all the functions defining the fit.

        After the ModelTrainer class has been instantiated,
        a call to this function (with a ``params`` dictionary) is necessary
        in order to generate the whole PDF model and perform a fit.

        This is a necessary step for hyperopt to work

        Parameters used only here:
            - ``epochs``: maximum number of iterations for the fit to run
            - ``stopping_patience``: patience of the stopper after finding a new minimum
            - ``stopping_delta``: minimum improvement to consider it a new minimum

        All other parameters are passed to the corresponding functions
        """
        # Reset the internal state of the backend every time this function is called
        print("")
        clear_backend_state()
        # Clean also validphys' internal caches which keep references to n3fit models
        central_predictions.cache_clear()
        predictions.cache_clear()

        # When doing hyperopt some entries in the params dictionary
        # can bring with them overriding arguments
        if self.mode_hyperopt:
            log.info("Performing hyperparameter scan")
            for key in self._hyperkeys:
                log.info(" > > Testing %s = %s", key, params[key])
            params = self._hyperopt_override(params)
        # Preprocess some hyperparameters
        if self.mode_hyperopt or (not self.trials):
            epochs = int(params["epochs"])
            stopping_patience = params["stopping_patience"]
        else:
            idx_hyperparamters = self.replicas[0] % self.trials["number_of_trials"]
            epochs = int(self.trials["epochs"][idx_hyperparamters])
            stopping_patience = self.trials["stopping_patience"][idx_hyperparamters]
        stopping_delta = params.get("stopping_delta", 0.0)
        stopping_epochs = int(epochs * stopping_patience)

        # Fill the 3 dictionaries (training, validation, experimental) with the layers and losses
        # when k-folding, these are the same for all folds
        positivity_dict = params.get("positivity", {})
        if not self.mode_hyperopt and self.trials:
            positivity_dict['initial'] = self.trials["initial"][idx_hyperparamters]
        integrability_dict = params.get("integrability", {})
        self._generate_observables(
            positivity_dict.get("multiplier"),
            positivity_dict.get("initial"),
            integrability_dict.get("multiplier"),
            integrability_dict.get("initial"),
            epochs,
            params.get("feature_scaling_points"),
        )
        threshold_pos = positivity_dict.get("threshold", 1e-6)
        threshold_chi2 = params.get("threshold_chi2", CHI2_THRESHOLD)

        # Initialize the chi2 dictionaries
        l_valid = []
        l_exper = []
        l_hyper = []
        # Hyperopt metrics evaluated over training/validation exp data
        trvl_chi2_per_fold = []
        trvl_phi2_per_fold = []
        trvl_logp_per_fold = []
        trvl_chi2exp_per_fold = []

        # Generate the grid in x, note this is the same for all partitions
        xinput = self._xgrid_generation()

        # Initialize all photon classes for the different replicas:
        if self.lux_params:
            photons = Photon(
                theoryid=self.theoryid, lux_params=self.lux_params, replicas=self.replicas
            )
        else:
            photons = None

        # Prepare the settings for all replica
        replicas_settings = []
        if self.mode_hyperopt or (not self.trials):
            for seed in self._nn_seeds:
                tmp = model_gen.ReplicaSettings(
                    seed=seed,
                    nodes=params["nodes_per_layer"],
                    activations=params["activation_per_layer"],
                    initializer=params["initializer"],
                    architecture=params["layer_type"],
                    dropout_rate=params["dropout"],
                    regularizer=params.get("regularizer"),
                    regularizer_args=params.get("regularizer_args"),
                )
                replicas_settings.append(tmp)
        else:
            # read hyperparameter values from hyperopt results
            for rep, seed in zip(self.replicas, self._nn_seeds):
                idx_hyperparamters = rep % self.trials["number_of_trials"]
                activations = [self.trials["activation_per_layer"][idx_hyperparamters]] * (
                    len(self.trials["nodes_per_layer"][idx_hyperparamters]) - 1
                )
                # last layer activation is always linear
                activations.append('linear')

                tmp = model_gen.ReplicaSettings(
                    seed=seed,
                    nodes=self.trials["nodes_per_layer"][idx_hyperparamters],
                    activations=activations,
                    initializer=self.trials["initializer"][idx_hyperparamters],
                    architecture=self.trials["layer_type"][idx_hyperparamters],
                    dropout_rate=self.trials["dropout"][idx_hyperparamters],
                    regularizer=params.get("regularizer"),
                    regularizer_args=params.get("regularizer_args"),
                )
                replicas_settings.append(tmp)

        ### Training loop
        for k, partition in enumerate(self.kpartitions):

            if k > 0:
                # When hyperoptimizing every patition takes the exact same model,
                # only the seed needs to be updated,.
                # Generate random integers for each k-fold from the input `nnseeds`
                # this helps avoid the integer overflow that may occur when doing k*nnseeds
                for seed, settings in zip(self._nn_seeds, replicas_settings):
                    rng = np.random.default_rng(seed=seed)
                    settings.seed = rng.integers(1, pow(2, 30)) * k

            # Generate the pdf model
            pdf_model = model_gen.generate_pdf_model(
                replicas_settings=replicas_settings,
                flav_info=self.flavinfo,
                fitbasis=self.fitbasis,
                impose_sumrule=self.impose_sumrule,
                scaler=self._scaler,
                photons=photons,
            )

            if photons:
                # The grid lives in the graph as a bound input; the view rebinds it (and
                # rebuilds the graph, and does nothing at all if the fit has no photon).
                backend = get_backend()
                photon_grid = backend.ops.to_numpy(xinput.input.tensor_content)
                if self._scaler:  # select only the non-scaled input
                    photon_grid = photon_grid[:, :, 1:]
                backend.view(pdf_model).bind_input("photon", photon_grid)

            # Model generation joins all the different observable layers
            # together with pdf model generated above
            models = self._model_generation(xinput, pdf_model, partition, k)

            # After model generation, apply possible weights files (P5's ``n3fit-weights/2``
            # files, one replica each).  The possibilities are a single model file (``load:``),
            # broadcast so that every replica starts with the same weights, or one file per
            # replica from ``load_weights_from_fit`` -- which the legacy code loaded into slot 0
            # every time (it never passed the replica index); the store loads each file into its
            # own replica.
            if self.model_file or self.load_weights_dict:
                backend = get_backend()
                weight_ensemble = backend.ensemble(
                    {GROUP_TRAINING: pdf_model}, weights_graph=pdf_model
                )
                if self.model_file:
                    log.info("Applying model file %s", self.model_file)
                    backend.load(weight_ensemble, self.model_file)
                if self.load_weights_dict:
                    for slot, replica in enumerate(self.replicas):
                        weights_path = self.load_weights_dict[replica]
                        log.info("Loading weights from path: %s", weights_path)
                        backend.load(weight_ensemble, weights_path, replica=slot)

            if k > 0:
                # Reset the positivity and integrability multipliers to the values their terms
                # were built with.  ``set_scalar`` *sets* (the legacy LagrangeCallback *scales*),
                # which is exactly what a reset is, and it goes through the contract instead of
                # poking the weights of a graph by layer name.
                #
                # This used to pass ``posinitials + posinitials`` for
                # ``posdatasets + integdatasets``, i.e. the *positivity* initial was also used for
                # integrability and, when there were more terms than initials, the last ones were
                # never reset at all (a runcard with ``positivity: initial`` !=
                # ``integrability: initial`` is all it takes).  The initials now travel with the
                # term names they belong to (``training["penalty_initials"]``).
                for name in self.objective_groups.names(
                    GROUP_POSITIVITY
                ) + self.objective_groups.names(GROUP_INTEGRABILITY):
                    self.terms[name].set_scalar(
                        "multiplier", self.training["penalty_initials"][name]
                    )

            # Generate the list containing reporting info necessary for chi2.  Kept on the
            # instance: ``evaluate`` normalizes by the same numbers the stopping used.
            reporting = self._prepare_reporting(partition)
            self._chi2_reporting = [entry for entry in reporting if entry.get("count_chi2")]

            if self.no_validation:
                # Substitute the validation model with the training model: with nothing held out,
                # the validation objective *is* the training one (the legacy read the validation
                # numbers out of the training model's logs), so the group membership follows.
                models["validation"] = models["training"]

            # The three role graphs, as one ensemble (P4).  ``weights_graph`` is the PDF model:
            # the roles are built by re-applying it, so that is where the trainable weights live.
            ensemble = get_backend().ensemble(
                {
                    GROUP_TRAINING: models["training"],
                    GROUP_VALIDATION: models["validation"],
                    GROUP_EXPERIMENTAL: models["experimental"],
                },
                weights_graph=pdf_model,
            )
            groups = dict(self.objective_groups.terms)
            if self.no_validation:
                groups[GROUP_VALIDATION] = groups[GROUP_TRAINING]

            # What the stopping decides on: the *term names* of each role, with the number of
            # points each carries.  ``_prepare_reporting`` gives the points per experiment (and
            # per fold); the names come from the generators, in the same order.
            chi2_entries = [entry for entry in reporting if entry.get("count_chi2")]
            tr_ndata = _ndata_for_terms(self.training["chi2_names"], chi2_entries, "ndata")
            vl_ndata = _ndata_for_terms(self.validation["chi2_names"], chi2_entries, "ndata_vl")
            if self.no_validation:
                vl_ndata = None  # the hook watches the training terms then (legacy behaviour)

            # The record of the fit (P4): filled by the stopping hook as the fit runs, read
            # afterwards by the reporting -- under the names the legacy ``Stopping`` object used.
            record = FitRecord()
            stopping_object = StoppingHook(
                record,
                ensemble,
                ndata=tr_ndata,
                vl_ndata=vl_ndata,
                positivity_terms=self.validation["penalty_names"],
                total_steps=epochs,
                stopping_patience=stopping_epochs,
                stopping_delta=stopping_delta,
                threshold_chi2=threshold_chi2,
                threshold_positivity=threshold_pos,
            )
            hooks = [
                stopping_object,
                # The schedule scales the *training* penalty of each group; the validation
                # positivity term is deliberately not in here (see ``_generate_observables``).
                LagrangeHook(
                    {
                        name: self.terms[name]
                        for name in self.objective_groups.names(GROUP_POSITIVITY)
                    },
                    dict(
                        zip(
                            self.objective_groups.names(GROUP_POSITIVITY),
                            self.training["posmultipliers"],
                        )
                    ),
                    period=PUSH_POSITIVITY_EACH,
                ),
                LagrangeHook(
                    {
                        name: self.terms[name]
                        for name in self.objective_groups.names(GROUP_INTEGRABILITY)
                    },
                    dict(
                        zip(
                            self.objective_groups.names(GROUP_INTEGRABILITY),
                            self.training["integmultipliers"],
                        )
                    ),
                    period=PUSH_INTEGRABILITY_EACH,
                ),
                LogHook(),
                *self._diagnostic_hooks,
            ]

            if self.mode_hyperopt or (not self.trials):
                optimizer_params = dict(params["optimizer"])
            else:
                idx_hyperparamters = self.replicas[0] % self.trials["number_of_trials"]
                optimizer_params = {
                    "clipnorm": self.trials['clipnorm'][idx_hyperparamters],
                    "learning_rate": self.trials['learning_rate'][idx_hyperparamters],
                    "optimizer_name": self.trials['optimizer'][idx_hyperparamters],
                }
            optimizer_spec = OptimizerSpec(
                optimizer_params.pop("optimizer_name"), optimizer_params
            )

            # Ask the backend whether this combination can be built at all, before building it
            # (contract ``Backend.check_feasible``; it is what used to surface as a
            # ``NotImplementedError`` from inside ``MetaModel.compile``).
            backend = get_backend()
            backend.check_feasible(
                _parametrization_spec(replicas_settings),
                optimizer_spec,
                self.terms[self.objective_groups.names(GROUP_TRAINING)[0]].spec,
            )

            # n3fit monitors at the engine's default interval unless the runcard asked for a
            # coarser one (D6); the backend may impose a minimum.
            optimizer = backend.optimizer(optimizer_spec)
            monitor_every = max(
                params.get("monitor_every", 1), optimizer.min_monitor_interval()
            )
            self.optimizer = optimizer
            self.ensemble = ensemble
            optimizer.run(
                ensemble,
                self.terms,
                groups,
                steps=epochs,
                monitor_every=monitor_every,
                hooks=hooks,
            )
            # The training chi2 of the *fitted* model -- what the legacy ``Stopping`` computed on
            # demand (``evaluate_training``).  A closure, so a consumer reads it exactly once.
            record._training_evaluation = (
                lambda ensemble=ensemble: self.optimizer.evaluate(
                    ensemble, self.terms, GROUP_TRAINING
                )
            )

            if self.mode_hyperopt:
                validation_loss = record.vl_chi2

                # number of active points in this fold
                # it would be nice to have a ndata_per_fold variable coming in the vp object...
                ndata = np.sum([np.count_nonzero(i[k]) for i in self.experimental["folds"]])
                # If ndata == 0 then it's the opposite, all data is in!
                if ndata == 0:
                    ndata = self.experimental["ndata"]

                # Compute experimental loss over the excluded datasets: the sum of the
                # experimental terms (per replica), per point.
                exp_terms = self.optimizer.evaluate(
                    ensemble, self.terms, GROUP_EXPERIMENTAL
                )
                exp_loss_raw = sum(
                    np.asarray(value) for value in exp_terms.values()
                )
                experimental_loss = exp_loss_raw / ndata

                # Penalties consume the FitRecord's read-side results (best epochs, validation
                # chi2, and patience), not the StoppingHook that makes the stopping decisions.
                penalties = {
                    penalty.__name__: penalty(pdf_model=pdf_model, stopping_object=record)
                    for penalty in self.hyper_penalties
                }

                # Extracting the necessary data to compute phi
                # First, create a list of `validphys.core.DataGroupSpec`
                # containing only exp datasets within the held out fold
                folded_datasets = partition["datasets"]
                experimental_data = self._filter_datagroupspec(folded_datasets)

                vplike_pdf = N3PDF(get_backend().ensemble(pdf_model))
                if self.boundary_condition is not None:
                    vplike_pdf.register_boundary(self.boundary_condition["unpolarized_bc"])

                # Compute per replica hyper losses
                hyper_loss = self._hyper_loss.compute_loss(
                    penalties=penalties,
                    experimental_loss=experimental_loss,
                    validation_loss=validation_loss,
                    pdf_object=vplike_pdf,
                    experimental_data=experimental_data,
                    fold_idx=k,
                )

                # Create another list of `validphys.core.DataGroupSpec`
                # containing now exp datasets that are included in the training/validation dataset
                trvl_data = self._filter_datagroupspec(folded_datasets, filter_in=False)
                # Evaluate the hyperopt metrics on the training/validation experimental sets
                hyper_metrics = compute_hyperopt_metrics(vplike_pdf, trvl_data)

                # Now save all information from this fold
                l_hyper.append(hyper_loss)
                l_valid.append(validation_loss)
                l_exper.append(experimental_loss)
                trvl_chi2_per_fold.append(hyper_metrics.chi2)
                trvl_chi2exp_per_fold.append(hyper_metrics.chi2exp)
                trvl_phi2_per_fold.append(hyper_metrics.phi2)
                trvl_logp_per_fold.append(hyper_metrics.logp)

                if hyper_loss > self.hyper_threshold:
                    log.info(
                        "Loss above threshold (%.1f > %.1f), breaking",
                        hyper_loss,
                        self.hyper_threshold,
                    )
                    # Apply a penalty proportional to the number of folds not computed
                    pen_mul = len(self.kpartitions) - k
                    l_hyper = [i * pen_mul for i in l_hyper]
                    passed = False
                    break
                else:
                    passed = True
                    log.info("Fold %d finished, loss=%.1f, pass=%s", k + 1, hyper_loss, passed)

            # endfor

        if self.mode_hyperopt:
            # turn losses into arrays
            l_hyper = np.array(l_hyper)
            l_valid = np.array(l_valid)
            l_exper = np.array(l_exper)

            # Compute the loss over all folds for hyperopt
            final_hyper_loss = self._hyper_loss.reduce_over_folds(l_hyper)

            # Add penalty term to ensure convergence
            exp_chi2_fitted_data = np.average(trvl_chi2exp_per_fold)
            expchi2_penalty = losses.LossHyperopt()
            final_hyper_loss += expchi2_penalty(exp_chi2_fitted_data)

            # Hyperopt needs a dictionary with information about the losses
            # it is possible to store arbitrary information in the trial file
            # by adding it to this dictionary
            dict_out = {
                "status": HYPEROPT_STATUSES[passed],
                "loss": final_hyper_loss,
                "validation_loss": np.average(l_valid),
                "experimental_loss": np.average(l_exper),
                "kfold_meta": {
                    "validation_losses": l_valid,
                    "trvl_losses_chi2": np.array(trvl_chi2_per_fold),
                    "trvl_losses_chi2exp": np.array(trvl_chi2exp_per_fold),
                    "trvl_losses_phi2": np.array(trvl_phi2_per_fold),
                    "trvl_losses_logp": np.array(trvl_logp_per_fold),
                    "experimental_losses": l_exper,
                    "hyper_losses": np.array(self._hyper_loss.exp_chi2_matrix),
                    "hyper_losses_chi2": np.array(self._hyper_loss.hyper_chi2_vector),
                    "hyper_losses_phi2": np.array(self._hyper_loss.hyper_phi2_vector),
                    "hyper_losses_logp": np.array(self._hyper_loss.hyper_logp_vector),
                    "penalties": {
                        name: np.array(values)
                        for name, values in self._hyper_loss.penalties.items()
                    },
                },
            }
            return dict_out

        # Keep a reference to the models after training for future reporting
        self.training["model"] = models["training"]
        self.experimental["model"] = models["experimental"]
        self.validation["model"] = models["validation"]

        # In a normal run, the only information we need to output is the record of the fit
        # (metadata about the stopping, under the names the reporting uses)
        # and the pdf model (which are used to generate the PDF grids and compute arclengths)
        if not self.mode_hyperopt:
            passed = any(bool(i) for i in record.e_best_chi2)
        dict_out = {"status": passed, "stopping_object": record, "pdf_model": pdf_model}
        return dict_out
