ReLU
==============================================================================


Key Concepts
------------------------------


Mathematical Equations
------------------------------

Process Steps
------------------------------

Forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # relu(x) = max(x, 0) + negative_slope * min(x, 0)
    # - negative_slope: [default value: 0]
    top_data[i] = std::max(bottom_data[i], Dtype(0)) + negative_slope * std::min(bottom_data[i], Dtype(0));

Backward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Source Codes
------------------------------


Test Examples
------------------------------
