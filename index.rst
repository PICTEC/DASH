.. DASH documentation master file, created by
   sphinx-quickstart on Mon Feb 25 17:51:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DASH's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

This is a listing of DASH classes that may be relevant for further development and
understanding of code. This is by no means exhaustive or complete, but should give
a highlight of how the demonstrator works.

Speech enhancement modules
--------------------------

.. automodule:: mono_model

.. autoclass:: MonoModel
	:members:

.. automodule:: mvdr_model
.. autoclass:: Model
	:members:


.. automodule:: post_filter
.. autoclass:: DAEPostFilter
	:members:

Preprocessing modules
---------------------

.. automodule:: audio
.. autoclass:: Audio
	:members:

.. autoclass:: PlayThread
	:members:

.. autoclass:: ReadThread
	:members:

.. automodule:: runtime
.. autoclass:: Runtime
	:members:

.. automodule:: utils
.. autofunction:: BufferMixin

.. autoclass:: Remix
	:members:

.. autoclass:: AdaptiveGain
	:members:
