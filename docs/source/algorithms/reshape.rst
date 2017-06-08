Reshape
==============================================================================


Key Concepts
------------------------------


Mathematical Equations
------------------------------

Process Steps
------------------------------

reshape data

::

    1. initialization

        input_start_axis = this->layer_param_.reshape_param().axis()
        start_axis = (input_start_axis >= 0) ? input_start_axis : bottom[0]->num_axes() + input_start_axis + 1

        num_axes = this->layer_param_.reshape_param().num_axes()
        end_axis = (num_axes == -1) ? bottom[0]->num_axes() : (start_axis + num_axes)

        num_axes_replaced = end_axis - start_axis
        num_axes_retained = bottom[0]->num_axes() - num_axes_replaced
        


Source Codes
------------------------------


Test Examples
------------------------------
