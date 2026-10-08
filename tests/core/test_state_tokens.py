from unittest_jax_support import JaxTestCase

import jax  # noqa: F401

from simsopt_jax.core.state_tokens import make_state_token_factory


class TestStateTokens(JaxTestCase):
    def test_state_token_factories_are_independent_monotonic_sequences(self):
        """State-token factories advance independently through monotonic integer
        sequences."""
        first = make_state_token_factory()
        second = make_state_token_factory()

        self.assertTrue(
            [first(), first(), first()] == [0, 1, 2],
            "[first(), first(), first()] == [0, 1, 2]",
        )
        self.assertTrue(
            [second(), second()] == [0, 1], "[second(), second()] == [0, 1]"
        )
        self.assertTrue(first() == 3, "first() == 3")
