import tensorflow as tf


class TensorFlowTheanoFunction(object):
    def __init__(self, inputs, outputs, updates=()):
        self._inputs = inputs
        self._outputs = outputs
        self._updates = updates

    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg
        try:
            outputs_identity = [tf.identity(output) for output in self._outputs]
            output_is_list = True
        except TypeError:
            outputs_identity = [tf.identity(self._outputs)]
            output_is_list = False
        with tf.control_dependencies(outputs_identity):
            assign_ops = [tf.assign(variable, replacement)
                          for variable, replacement in self._updates]
        outputs_list = tf.get_default_session().run(
            outputs_identity + assign_ops, feeds)[:len(outputs_identity)]
        if output_is_list:
            return outputs_list
        else:
            assert len(outputs_list) == 1
            return outputs_list[0]
