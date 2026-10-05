"""The Keras callbacks that only *watch* the fit (P4).

The three callbacks that made up the training loop -- ``StoppingCallback``, ``LagrangeCallback``
and ``TimerCallback``, all of them built on ``CallbackStep`` -- were deleted with
``MetaModel.perform_fit``: the loop is ``KerasOptimizer.run`` now and the stopping, the Lagrange
schedule and the timing are *hooks* it calls (``n3fit.stopping``), not Keras callbacks.  What is
left here is the one thing tensorboard needs: it is a Keras callback by nature, it is
tensorflow-only, and it is *the backend's* to provide -- n3fit asks for a hook and the backend
decides whether it can offer one (``Capabilities.supports_tensorboard``, contract §1.10/A3).
"""

from keras import callbacks


def gen_tensorboard_callback(log_dir, profiling=False, histogram_freq=0):
    """
    Generate a tensorboard callback writing to ``log_dir``.

    Note the usage of this callback can hurt performance
    At the moment can only be used with TensorFlow: https://github.com/keras-team/keras/issues/19121

    Parameters
    ----------
        log_dir: str
            Directory in which to save tensorboard details
        profiling: bool
            Whether or not to save profiling information (default False)
        histogram_freq: int
            Frequency (in epochs) at which to save weight histograms
    """
    profile_batch = 1 if profiling else 0
    return callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_freq,
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        profile_batch=profile_batch,
    )
