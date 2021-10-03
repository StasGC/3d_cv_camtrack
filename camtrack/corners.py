#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    _MAX_CORNERS = 1000
    _RADIUS = 100
    st_params = dict(qualityLevel=0.03,
                     minDistance=15,
                     blockSize=8)

    lk_params = dict(winSize=(10, 10),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))

    image_0 = frame_sequence[0]
    corners0 = cv2.goodFeaturesToTrack(image_0,
                                       mask=None,
                                       maxCorners=_MAX_CORNERS,
                                       **st_params)

    corners0 = corners0.reshape(-1, 2)

    num_corners = corners0.shape[0]
    block_sizes = np.zeros(num_corners)
    block_sizes.fill(st_params['blockSize'])

    corners = FrameCorners(
        np.arange(num_corners),
        corners0,
        block_sizes
    )
    builder.set_corners_at_frame(0, corners)

    for frame, image in enumerate(frame_sequence[1:], 1):
        corners1, status, _ = cv2.calcOpticalFlowPyrLK(np.uint8(image_0 * 255),
                                                       np.uint8(image * 255),
                                                       corners.points,
                                                       None,
                                                       **lk_params)

        status = status.flatten()
        corners1 = corners1.reshape(-1, 2)

        corners = FrameCorners(
            corners.ids[status == 1],
            corners1[status == 1],
            corners.sizes[status == 1]
        )

        if _MAX_CORNERS - sum(status) > 300:
            mask = np.full_like(image, 255, dtype='uint8')
            curr_corners = corners.points.round().astype(np.uint8)

            for corner, size in zip(curr_corners, corners.sizes):
                cv2.circle(mask, corner, _RADIUS, 0, -1)

            new_corners = cv2.goodFeaturesToTrack(image,
                                                  mask=mask,
                                                  maxCorners=_MAX_CORNERS - sum(status),
                                                  **st_params)

            if new_corners is not None:
                new_corners = new_corners.reshape(-1, 2)
                max_ids = corners.ids.max()
                corners = FrameCorners(
                    np.vstack((corners.ids,
                               np.arange(max_ids,
                                         max_ids + new_corners.shape[0]).reshape((-1, 1)) + 1)),
                    np.vstack((corners.points, new_corners)),
                    np.vstack((corners.sizes, np.full(new_corners.shape[0], st_params['blockSize']).reshape((-1, 1))))
                )

        builder.set_corners_at_frame(frame, corners)
        image_0 = image


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
