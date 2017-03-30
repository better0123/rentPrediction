"""This module is for testing the trained result reached our claimed R2 value
"""

from predict_rent import score_rent

def test_rent():
    r"""Test the trained model with the claimed R2 value

    """
    assert score_rent() > 0.59
