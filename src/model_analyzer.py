# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tools for analyzing the operations and variables in a TensorFlow graph.

To analyze the operations in a graph:

  images, labels = LoadData(...)
  predictions = MyModel(images)

  slim.model_analyzer.analyze_ops(tf.compat.v1.get_default_graph(),
  print_info=True)

To analyze the model variables in a graph:

  variables = tf.compat.v1.model_variables()
  slim.model_analyzer.analyze_vars(variables, print_info=False)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import reduce_util as ds_reduce_util


def tensor_description(var):
    """Returns a compact and informative string about a tensor.

  Args:
    var: A tensor variable.

  Returns:
    a string with type and size, e.g.: (float32 1x8x8x1024).
  """
    description = '(' + str(var.dtype.name) + ' '
    sizes = var.get_shape()
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += 'x'
    description += ')'
    return description


def analyze_ops(graph, print_info=False):
    """Compute the estimated size of the ops.outputs in the graph.

  Args:
    graph: the graph containing the operations.
    print_info: Optional, if true print ops and their outputs.

  Returns:
    total size of the ops.outputs
  """
    if print_info:
        print('---------')
        print('Operations: name -> (type shapes) [size]')
        print('---------')
    total_size = 0
    for op in graph.get_operations():
        op_size = 0
        shapes = []
        for output in op.outputs:
            # if output.num_elements() is None or [] assume size 0.
            output_size = output.get_shape().num_elements() or 0
            if output.get_shape():
                shapes.append(tensor_description(output))
            op_size += output_size
        if print_info:
            print(op.name, '\t->', ', '.join(shapes), '[' + str(op_size) + ']')
        total_size += op_size
    return total_size


def analyze_vars(variables, print_info=False):
    """Prints the names and shapes of the variables.

  Args:
    variables: list of variables, for example tf.compat.v1.global_variables().
    print_info: Optional, if true print variables and their shape.

  Returns:
    (total size of the variables, total bytes of the variables)
  """
    if print_info:
        print('---------')
        print('Variables: name (type shape) [size]')
        print('---------')
    total_size = 0
    total_bytes = 0
    for var in variables:
        # if var.num_elements() is None or [] assume size 0.
        var_size = var.get_shape().num_elements() or 0
        var_bytes = var_size * var.dtype.size
        total_size += var_size
        total_bytes += var_bytes
        if print_info:
            print(var.name, tensor_description(var),
                  '[%d, bytes: %d]' % (var_size, var_bytes))
    if print_info:
        print('Total size of variables: %d' % total_size)
        print('Total bytes of variables: %d' % total_bytes)
    return total_size, total_bytes


def _zero_debias(strategy, unbiased_var, value, decay):
    """Compute the delta required for a debiased Variable.

  All exponential moving averages initialized with Tensors are initialized to 0,
  and therefore are biased to 0. Variables initialized to 0 and used as EMAs are
  similarly biased. This function creates the debias updated amount according to
  a scale factor, as in https://arxiv.org/abs/1412.6980.

  To demonstrate the bias the results from 0-initialization, take an EMA that
  was initialized to `0` with decay `b`. After `t` timesteps of seeing the
  constant `c`, the variable have the following value:

  ```
    EMA = 0*b^(t) + c*(1 - b)*b^(t-1) + c*(1 - b)*b^(t-2) + ...
        = c*(1 - b^t)
  ```

  To have the true value `c`, we would divide by the scale factor `1 - b^t`.

  In order to perform debiasing, we use two shadow variables. One keeps track of
  the biased estimate, and the other keeps track of the number of updates that
  have occurred.

  Args:
    strategy: `Strategy` used to create and update variables.
    unbiased_var: A Variable representing the current value of the unbiased EMA.
    value: A Tensor representing the most recent value.
    decay: A Tensor representing `1-decay` for the EMA.

  Returns:
    Operation which updates unbiased_var to the debiased moving average value.
  """
    with variable_scope.variable_scope(
            unbiased_var.name[:-len(":0")], values=[unbiased_var, value, decay]):
        with ops.init_scope():
            biased_initializer = init_ops.zeros_initializer()
            local_step_initializer = init_ops.zeros_initializer()

        def _maybe_get_unique(name):
            """Get name for a unique variable, if not `reuse=True`."""
            if variable_scope.get_variable_scope().reuse:
                return name
            vs_vars = [
                x.op.name
                for x in variable_scope.get_variable_scope().global_variables()
            ]
            full_name = variable_scope.get_variable_scope().name + "/" + name
            if full_name not in vs_vars:
                return name
            idx = 1
            while full_name + ("_%d" % idx) in vs_vars:
                idx += 1
            return name + ("_%d" % idx)

        with strategy.extended.colocate_vars_with(unbiased_var):
            biased_var = variable_scope.get_variable(
                _maybe_get_unique("biased"),
                initializer=biased_initializer,
                shape=unbiased_var.get_shape(),
                dtype=unbiased_var.dtype,
                trainable=False)
            local_step = variable_scope.get_variable(
                _maybe_get_unique("local_step"),
                shape=[],
                dtype=unbiased_var.dtype,
                initializer=local_step_initializer,
                trainable=False)

    def update_fn(v, value, biased_var, local_step):
        update_biased = state_ops.assign_sub(biased_var,
                                             (biased_var - value) * decay)
        update_local_step = local_step.assign_add(1)

        # This function gets `1 - decay`, so use `1.0 - decay` in the exponent.
        bias_factor = 1 - math_ops.pow(1.0 - decay, update_local_step)
        return state_ops.assign(
            v, update_biased / bias_factor, name=ops.get_name_scope() + "/")

    return strategy.extended.update(
        unbiased_var, update_fn, args=(value, biased_var, local_step))


# TODO(touts): switch to variables.Variable.
def assign_moving_average(variable, value, decay, zero_debias=True, name=None):
    """Compute the moving average of a variable.

  The moving average of 'variable' updated with 'value' is:
    variable * decay + value * (1 - decay)

  The returned Operation sets 'variable' to the newly computed moving average,
  by performing this subtraction:
     variable -= (1 - decay) * (variable - value)

  Since variables that are initialized to a `0` value will be `0` biased,
  `zero_debias` optionally enables scaling by the mathematically correct
  debiasing factor of
    1 - decay ** num_updates
  See `ADAM: A Method for Stochastic Optimization` Section 3 for more details
  (https://arxiv.org/abs/1412.6980).

  The names of the debias shadow variables, by default, include both the scope
  they were created in and the scope of the variables they debias. They are also
  given a uniquifying-suffix.

  E.g.:

  ```
    with tf.compat.v1.variable_scope('scope1'):
      with tf.compat.v1.variable_scope('scope2'):
        var = tf.compat.v1.get_variable('foo')
        update_1 = tf.assign_moving_average(var, 0.0, 1.0)
        update_2 = tf.assign_moving_average(var, 0.0, 0.9)

    # var.name: 'scope1/scope2/foo'
    # shadow var names: 'scope1/scope2/scope1/scope2/foo/biased'
    #                   'scope1/scope2/scope1/scope2/foo/biased_1'
  ```

  Args:
    variable: A Variable.
    value: A tensor with the same shape as 'variable'.
    decay: A float Tensor or float value.  The moving average decay.
    zero_debias: A python bool. If true, assume the variable is 0-initialized
      and unbias it, as in https://arxiv.org/abs/1412.6980. See docstring in
        `_zero_debias` for more details.
    name: Optional name of the returned operation.

  Returns:
    A tensor which if evaluated will compute and return the new moving average.
  """

    with ops.name_scope(name, "AssignMovingAvg",
                        [variable, value, decay]) as scope:
        decay = ops.convert_to_tensor(1.0 - decay, name="decay")
        if decay.dtype != variable.dtype.base_dtype:
            decay = math_ops.cast(decay, variable.dtype.base_dtype)

        def update_fn(v, value):
            return state_ops.assign_sub(v, (v - value) * decay, name=scope)

        def update(strategy, v, value):
            if zero_debias:
                return _zero_debias(strategy, v, value, decay)
            else:
                return strategy.extended.update(v, update_fn, args=(value,))

        replica_context = distribution_strategy_context.get_replica_context()
        if replica_context:
            # In a replica context, we update variable using the mean of value across
            # replicas.
            def merge_fn(strategy, v, value):
                value = strategy.extended.reduce_to(ds_reduce_util.ReduceOp.MEAN, value,
                                                    v)
                return update(strategy, v, value)

            return replica_context.merge_call(merge_fn, args=(variable, value))
        else:
            strategy = distribution_strategy_context.get_cross_replica_context()
            return update(strategy, variable, value)
