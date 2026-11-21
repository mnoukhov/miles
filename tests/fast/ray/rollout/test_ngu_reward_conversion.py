"""Chain-baseline regression coverage without a Ray cluster fixture."""

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.train_data_conversion import _post_process_rewards
from miles.rollout.resampling import REWARD_BASELINE_KEY


class TestNeverGiveUpRewards:
    @staticmethod
    def _never_give_up(rewards, *, grpo_std_normalization, chain):
        """Advantages of one merged NGU group whose chain has (reward sum, sum of squares, count)."""
        args = make_args(
            advantage_estimator="grpo",
            rewards_normalization=True,
            grpo_std_normalization=grpo_std_normalization,
        )
        samples = make_samples_grouped(1, len(rewards), rewards=rewards)
        for sample in samples:
            sample.metadata[REWARD_BASELINE_KEY] = {"sum": chain[0], "sq_sum": chain[1], "count": chain[2]}
        return _post_process_rewards(args, samples, custom_reward_post_process_func=None)[1]

    @pytest.mark.parametrize(
        "grpo_std_normalization, expected",
        [(True, [2.475, -0.825, -0.825, -0.825]), (False, [0.875, -0.2917, -0.2917, -0.2917])],
    )
    def test_never_give_up_binary_group_trains_against_its_stale_attempt(self, grpo_std_normalization, expected):
        """Stale [0, 0, 0, 0] plus trained [1, 0, 0, 0]: the chain has sum 1, sum of squares 1, 8 samples."""
        processed = self._never_give_up(
            [1.0, 0.0, 0.0, 0.0], grpo_std_normalization=grpo_std_normalization, chain=(1.0, 1.0, 8)
        )
        assert processed == pytest.approx(expected, abs=1e-3)

    def test_never_give_up_partial_rewards_train_against_their_stale_attempt(self):
        """Stale [0.2] * 4 plus trained [1, 0.6, 0.2, 0]: the chain has sum 2.6, sum of squares 1.56, 8 samples."""
        processed = self._never_give_up([1.0, 0.6, 0.2, 0.0], grpo_std_normalization=True, chain=(2.6, 1.56, 8))
        assert processed == pytest.approx([2.112, 0.339, -0.913, -1.538], abs=1e-3)

    @pytest.mark.parametrize("grpo_std_normalization", [True, False])
    def test_never_give_up_group_without_stale_attempts_matches_plain_grpo(self, grpo_std_normalization):
        rewards = [1.0, 0.0, 0.0, 0.0]
        processed = self._never_give_up(rewards, grpo_std_normalization=grpo_std_normalization, chain=(1.0, 1.0, 4))
        args = make_args(
            advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=grpo_std_normalization
        )
        plain = _post_process_rewards(
            args, make_samples_grouped(1, 4, rewards=rewards), custom_reward_post_process_func=None
        )[1]
        assert processed == pytest.approx(plain, abs=1e-5)
