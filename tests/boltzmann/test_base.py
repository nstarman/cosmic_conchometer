
"""Testing :mod:`cosmic_conchometer.boltzmann.base`."""

__all__ = ["TestTransferFunctionBase"]


##############################################################################
# IMPORTS


import pytest

# PROJECT-SPECIFIC
#
from cosmic_conchometer.boltzmann.base import TransferFunctionBase

##############################################################################
# TESTS
##############################################################################


class TestTransferFunctionBase:
    """Test :class:`cosmic_conchometer.boltzmann.base.TransferFunctionBase`."""

    @pytest.fixture
    def tf(self):

        # need to subclass, b/c abstract
        class TransferFunctionSubClass(TransferFunctionBase):
            def __call__(self):
                return "test"

        return TransferFunctionBase()

    #################################################################
    # Method Tests

    @pytest.mark.skip("TODO")
    def test_method(self):
        """Test :class:`PACKAGE.CLASS.METHOD`."""
        assert False


##############################################################################
# END
