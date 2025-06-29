"""Tests for the FrequencyResampler utility."""

from unittest.mock import patch

import pytest

from robodm.utils.resampler import FrequencyResampler


class TestFrequencyResampler:
    """Test FrequencyResampler class."""

    def test_init_basic(self):
        """Test basic initialization."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)

        assert resampler.period_ms == 100
        assert resampler.sl_start == 0
        assert resampler.sl_stop is None
        assert resampler.sl_step == 1
        assert resampler._seek_offset_frames == 0
        assert resampler.last_pts == {}
        assert resampler.kept_idx == {}

    def test_init_with_seek_offset(self):
        """Test initialization with seek offset."""
        resampler = FrequencyResampler(period_ms=50,
                                       sl_start=10,
                                       sl_stop=100,
                                       sl_step=2,
                                       seek_offset_frames=5)

        assert resampler.period_ms == 50
        assert resampler.sl_start == 10
        assert resampler.sl_stop == 100
        assert resampler.sl_step == 2
        assert resampler._seek_offset_frames == 5

    def test_register_feature_new(self):
        """Test registering a new feature."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)

        with patch("robodm.utils.resampler.logger") as mock_logger:
            resampler.register_feature("test_feature")

        assert "test_feature" in resampler.kept_idx
        assert "test_feature" in resampler.last_pts
        assert resampler.kept_idx[
            "test_feature"] == -1  # seek_offset_frames - 1
        assert resampler.last_pts["test_feature"] is None
        mock_logger.debug.assert_called_once()

    def test_register_feature_with_seek_offset(self):
        """Test registering feature with seek offset."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1,
                                       seek_offset_frames=10)

        resampler.register_feature("test_feature")

        assert resampler.kept_idx["test_feature"] == 9  # seek_offset_frames - 1

    def test_register_feature_already_exists(self):
        """Test registering an already existing feature."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)

        # Register first time
        resampler.register_feature("test_feature")
        original_idx = resampler.kept_idx["test_feature"]

        # Register again - should not change
        resampler.register_feature("test_feature")

        assert resampler.kept_idx["test_feature"] == original_idx

    def test_process_packet_no_pts(self):
        """Test processing packet with no timestamp."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        with patch("robodm.utils.resampler.logger") as mock_logger:
            keep_current, num_duplicates = resampler.process_packet(
                "test_feature", None, False)

        assert keep_current is True
        assert num_duplicates == 0
        mock_logger.debug.assert_called_once()

    def test_process_packet_no_resampling(self):
        """Test processing packet when resampling is disabled."""
        resampler = FrequencyResampler(
            period_ms=None,
            sl_start=0,
            sl_stop=None,
            sl_step=1  # Disabled
        )
        resampler.register_feature("test_feature")

        keep_current, num_duplicates = resampler.process_packet(
            "test_feature", 1000, True)

        assert keep_current is True
        assert num_duplicates == 0

    def test_process_packet_first_packet(self):
        """Test processing the first packet."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        keep_current, num_duplicates = resampler.process_packet(
            "test_feature", 1000, False)

        assert keep_current is True
        assert num_duplicates == 0

    def test_process_packet_downsampling(self):
        """Test downsampling - gap smaller than period."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # Process first packet
        resampler.process_packet("test_feature", 1000, False)
        resampler.update_last_pts("test_feature", 1000)

        # Process second packet with small gap (50ms < 100ms period)
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature", 1050, True)

        assert keep_current is False  # Should be skipped
        assert num_duplicates == 0

    def test_process_packet_normal_gap(self):
        """Test normal gap equal to period."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # Process first packet
        resampler.process_packet("test_feature", 1000, False)
        resampler.update_last_pts("test_feature", 1000)

        # Process second packet with exact period gap
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature", 1100, True)

        assert keep_current is True
        assert num_duplicates == 0

    def test_process_packet_upsampling(self):
        """Test upsampling - gap larger than period."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # Process first packet
        resampler.process_packet("test_feature", 1000, False)
        resampler.update_last_pts("test_feature", 1000)

        # Process second packet with large gap (350ms > 100ms period)
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature", 1350, True)

        assert keep_current is True
        assert num_duplicates == 2  # (350 // 100) - 1 = 2 duplicates

    def test_process_packet_upsampling_no_prior_frame(self):
        """Test upsampling when no prior frame exists."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # Process first packet
        resampler.process_packet("test_feature", 1000, False)
        resampler.update_last_pts("test_feature", 1000)

        # Process second packet with large gap but no prior frame
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature",
            1350,
            False  # has_prior_frame=False
        )

        assert keep_current is True
        assert num_duplicates == 0  # No duplicates when no prior frame

    def test_next_index(self):
        """Test next_index method."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # Initial index should be -1
        assert resampler.kept_idx["test_feature"] == -1

        # First call should return 0
        next_idx = resampler.next_index("test_feature")
        assert next_idx == 0
        assert resampler.kept_idx["test_feature"] == 0

        # Second call should return 1
        next_idx = resampler.next_index("test_feature")
        assert next_idx == 1
        assert resampler.kept_idx["test_feature"] == 1

    def test_want_basic_slice(self):
        """Test want method with basic slice parameters."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=5,
                                       sl_stop=15,
                                       sl_step=2)

        # Test indices before start
        assert resampler.want(0) is False
        assert resampler.want(4) is False

        # Test indices within range with correct step
        assert resampler.want(5) is True  # start
        assert resampler.want(7) is True  # start + step
        assert resampler.want(9) is True  # start + 2*step
        assert resampler.want(11) is True  # start + 3*step
        assert resampler.want(13) is True  # start + 4*step

        # Test indices within range but wrong step
        assert resampler.want(6) is False
        assert resampler.want(8) is False
        assert resampler.want(10) is False

        # Test indices at/after stop
        assert resampler.want(15) is False
        assert resampler.want(16) is False

    def test_want_no_stop(self):
        """Test want method with no stop limit."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=10,
                                       sl_stop=None,
                                       sl_step=3)

        # Test indices before start
        assert resampler.want(9) is False

        # Test indices with correct step
        assert resampler.want(10) is True  # start
        assert resampler.want(13) is True  # start + step
        assert resampler.want(16) is True  # start + 2*step
        assert resampler.want(100) is True  # large index with correct step

        # Test indices with wrong step
        assert resampler.want(11) is False
        assert resampler.want(12) is False
        assert resampler.want(14) is False

    def test_want_step_one(self):
        """Test want method with step=1 (every frame)."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=5,
                                       sl_stop=10,
                                       sl_step=1)

        # All indices in range should be wanted
        assert resampler.want(4) is False
        assert resampler.want(5) is True
        assert resampler.want(6) is True
        assert resampler.want(7) is True
        assert resampler.want(8) is True
        assert resampler.want(9) is True
        assert resampler.want(10) is False

    def test_update_last_pts(self):
        """Test update_last_pts method."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # Initial value should be None
        assert resampler.last_pts["test_feature"] is None

        # Update with timestamp
        resampler.update_last_pts("test_feature", 1500)
        assert resampler.last_pts["test_feature"] == 1500

        # Update with None
        resampler.update_last_pts("test_feature", None)
        assert resampler.last_pts["test_feature"] is None

    def test_multiple_features(self):
        """Test resampler with multiple features."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)

        # Register multiple features
        resampler.register_feature("feature1")
        resampler.register_feature("feature2")

        # Each feature should have independent bookkeeping
        assert len(resampler.kept_idx) == 2
        assert len(resampler.last_pts) == 2

        # Process packets for different features
        resampler.process_packet("feature1", 1000, False)
        resampler.update_last_pts("feature1", 1000)

        resampler.process_packet("feature2", 2000, False)
        resampler.update_last_pts("feature2", 2000)

        # Each feature should maintain separate state
        assert resampler.last_pts["feature1"] == 1000
        assert resampler.last_pts["feature2"] == 2000

        # Increment indices independently
        idx1 = resampler.next_index("feature1")
        idx2 = resampler.next_index("feature2")

        assert idx1 == 0
        assert idx2 == 0
        assert resampler.kept_idx["feature1"] == 0
        assert resampler.kept_idx["feature2"] == 0


class TestFrequencyResamplerEdgeCases:
    """Test edge cases for FrequencyResampler."""

    def test_zero_period(self):
        """Test with zero period."""
        resampler = FrequencyResampler(period_ms=0,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # First packet
        resampler.process_packet("test_feature", 1000, False)
        resampler.update_last_pts("test_feature", 1000)

        # Second packet with same timestamp
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature", 1000, True)

        # With period=0, gap (0) is not < period (0), so should keep
        assert keep_current is True
        assert num_duplicates == 0

    def test_very_large_gap(self):
        """Test with very large timestamp gap."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # Process first packet
        resampler.process_packet("test_feature", 1000, False)
        resampler.update_last_pts("test_feature", 1000)

        # Process packet with very large gap
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature",
            10000,
            True  # 9000ms gap
        )

        assert keep_current is True
        assert num_duplicates == 89  # (9000 // 100) - 1

    def test_negative_timestamps(self):
        """Test with negative timestamps."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # Process packet with negative timestamp
        resampler.process_packet("test_feature", -1000, False)
        resampler.update_last_pts("test_feature", -1000)

        # Process second packet
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature",
            -900,
            True  # 100ms gap
        )

        assert keep_current is True
        assert num_duplicates == 0

    def test_slice_edge_cases(self):
        """Test slice filtering edge cases."""
        resampler = FrequencyResampler(
            period_ms=100,
            sl_start=0,
            sl_stop=1,
            sl_step=1  # Very small range
        )

        # Only index 0 should be wanted
        assert resampler.want(0) is True
        assert resampler.want(1) is False

    def test_large_step_size(self):
        """Test with large step size."""
        resampler = FrequencyResampler(
            period_ms=100,
            sl_start=0,
            sl_stop=100,
            sl_step=50  # Large step
        )

        # Only every 50th index should be wanted
        assert resampler.want(0) is True
        assert resampler.want(50) is True
        assert resampler.want(25) is False
        assert resampler.want(75) is False

    def test_exact_period_boundaries(self):
        """Test exact period boundary conditions."""
        resampler = FrequencyResampler(period_ms=100,
                                       sl_start=0,
                                       sl_stop=None,
                                       sl_step=1)
        resampler.register_feature("test_feature")

        # First packet
        resampler.process_packet("test_feature", 1000, False)
        resampler.update_last_pts("test_feature", 1000)

        # Test gap exactly equal to period - 1
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature",
            1099,
            True  # 99ms gap
        )
        assert keep_current is False  # Should be dropped (gap < period)

        # Test gap exactly equal to period
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature",
            1100,
            True  # 100ms gap
        )
        assert keep_current is True  # Should be kept
        assert num_duplicates == 0

        # Update for next test
        resampler.update_last_pts("test_feature", 1100)

        # Test gap exactly equal to period + 1
        keep_current, num_duplicates = resampler.process_packet(
            "test_feature",
            1201,
            True  # 101ms gap
        )
        assert keep_current is True  # Should be kept
        assert num_duplicates == 0  # No duplicates (gap // period == 1)

    def test_complex_resampling_scenario(self):
        """Test complex scenario with multiple operations."""
        resampler = FrequencyResampler(period_ms=50,
                                       sl_start=2,
                                       sl_stop=10,
                                       sl_step=2,
                                       seek_offset_frames=5)

        # Register feature
        resampler.register_feature("complex_feature")

        # Check initial state
        assert resampler.kept_idx["complex_feature"] == 4  # seek_offset - 1

        # Process multiple packets with varying gaps
        timestamps = [1000, 1025, 1075, 1200, 1300]
        results = []

        for i, ts in enumerate(timestamps):
            has_prior = i > 0
            keep, duplicates = resampler.process_packet(
                "complex_feature", ts, has_prior)
            results.append((keep, duplicates))
            if keep:
                resampler.update_last_pts("complex_feature", ts)

        # Verify results
        # ts=1000: first packet, always keep
        assert results[0] == (True, 0)

        # ts=1025: gap=25ms < period=50ms, should drop
        assert results[1] == (False, 0)

        # ts=1075: gap=75ms > period=50ms, keep with 0 duplicates
        assert results[2] == (True, 0)

        # ts=1200: gap=125ms, keep with 1 duplicate (125//50 - 1 = 1)
        assert results[3] == (True, 1)

        # ts=1300: gap=100ms, keep with 1 duplicate (100//50 - 1 = 1)
        assert results[4] == (True, 1)
