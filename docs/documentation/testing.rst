.. _cosmic_conchometer-test:

=================
Running the tests
=================

The tests are written assuming they will be run with `pytest <http://doc.pytest.org/>`_ using the Astropy `custom test runner <http://docs.astropy.org/en/stable/development/testguide.html>`_. To set up a Conda environment to run the full set of tests, install ``cosmic_conchometer`` or see the setup.cfg file for dependencies. Once the dependencies are installed, you can run the tests two ways:

1. By importing ``cosmic_conchometer``::

    import cosmic_conchometer
    cosmic_conchometer.test()

2. By cloning the ``cosmic_conchometer`` repository and running::

    python setup.py test


Reference/API
=============

The test functions.

.. currentmodule:: cosmic_conchometer.tests
