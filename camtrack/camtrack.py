#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    TriangulationParameters,
    build_correspondences,
    triangulate_correspondences,
    PnpParameters,
    check_inliers_mask,
    rodrigues_and_translation_to_view_mat3x4
)


def _add_new_points(point_cloud: PointCloudBuilder,
                    corners_1: FrameCorners,
                    corners_2: FrameCorners,
                    view_matrix_1: np.ndarray,
                    view_matrix_2: np.ndarray,
                    intrinsic_mat: np.ndarray,
                    params: TriangulationParameters) -> None:
    correspondences = build_correspondences(corners_1, corners_2)
    points, ids, err = triangulate_correspondences(correspondences,
                                                   view_matrix_1,
                                                   view_matrix_2,
                                                   intrinsic_mat,
                                                   params)
    point_cloud.add_points(ids, points)
    print(f"{points.shape[0]} points was triangulated")


def _view_matrix_from_pnp(point_cloud: PointCloudBuilder,
                          corners: FrameCorners,
                          intrinsic_mat: np.ndarray,
                          params: PnpParameters) -> Tuple[np.array, float]:
    _, (ind_1, ind_2) = snp.intersect(point_cloud.ids.flatten(), corners.ids.flatten(), indices=True)

    iterationsCount = int(np.ceil(np.log(1.0 - params.inliers_probability) /
                                  np.log(1.0 - (1.0 - params.supposed_outliers_ratio) ** params.correspondences_num)))
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        point_cloud.points[ind_1],
        corners.points[ind_2],
        intrinsic_mat,
        None,
        iterationsCount=iterationsCount,
        reprojectionError=params.max_reprojection_error,
        flags=cv2.SOLVEPNP_EPNP
    )

    # TODO: реализовать уточнение решения на инлаерах (пока не работает)
    # if not retval:
    #     return None, 0
    # inliers = inliers.flatten()
    # inliers_mask = np.full_like(ids, False)
    # inliers_mask[inliers] = True
    # if not check_inliers_mask(inliers_mask, params.min_inlier_count, params.min_inlier_ratio):
    #     return None, 0
    #
    # retval_2, rvec_2, tvec_2 = cv2.solvePnP(
    #     point_cloud.points[ind_1[inliers]],
    #     corners.points[ind_2[inliers]],
    #     intrinsic_mat,
    #     None,
    #     rvec,
    #     tvec,
    #     True,
    #     cv2.SOLVEPNP_ITERATIVE
    # )
    # if retval_2:
    #     return rodrigues_and_translation_to_view_mat3x4(rvec_2, tvec_2), len(inliers)
    # else:
    #     return rodrigues_and_translation_to_view_mat3x4(rvec, tvec), len(inliers)

    print(f"On this iterations was found {len(inliers)} inliers")
    return rodrigues_and_translation_to_view_mat3x4(rvec, tvec), len(inliers)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    # TODO: implement
    frame_count = len(corner_storage)
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count

    _max_reprojection_error = 1.0
    triangulate_params = TriangulationParameters(
        max_reprojection_error=_max_reprojection_error,
        min_triangulation_angle_deg=1.0,
        min_depth=0.1
    )
    pnp_params = PnpParameters(
        max_reprojection_error=_max_reprojection_error,
        min_inlier_count=5,
        min_inlier_ratio=0.1,
        supposed_outliers_ratio=0.5,
        inliers_probability=0.999,
        correspondences_num=5
    )

    # TODO: implement
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    correspondences = build_correspondences(
        corner_storage[known_view_1[0]],
        corner_storage[known_view_2[0]]
    )
    _points, _ids, err = triangulate_correspondences(correspondences,
                                                     view_mats[known_view_1[0]],
                                                     view_mats[known_view_2[0]],
                                                     intrinsic_mat,
                                                     triangulate_params)
    point_cloud_builder = PointCloudBuilder(_ids, _points)
    print(f"{_points.shape[0]} triangulated points from first frames")
    print(f"Point cloud size: {point_cloud_builder.points.shape[0]}")

    known_view_mats = [known_view_1[0], known_view_2[0]]
    step_size = 10

    for i in range(step_size):
        curr_frame = i
        prev_frame = i
        print(f"Processing the {curr_frame} frame")
        if curr_frame not in known_view_mats:
            view_mats[curr_frame], num_inliers = _view_matrix_from_pnp(
                point_cloud_builder,
                corner_storage[curr_frame],
                intrinsic_mat,
                pnp_params
            )
            known_view_mats.append(curr_frame)

        for curr_frame in range(step_size + i, frame_count, step_size):
            print(f"Processing the {curr_frame} frame")
            if curr_frame not in known_view_mats:
                view_mats[curr_frame], num_inliers = _view_matrix_from_pnp(
                    point_cloud_builder,
                    corner_storage[curr_frame],
                    intrinsic_mat,
                    pnp_params
                )
                known_view_mats.append(curr_frame)

            _add_new_points(
                point_cloud_builder,
                corner_storage[curr_frame],
                corner_storage[prev_frame],
                view_mats[curr_frame],
                view_mats[prev_frame],
                intrinsic_mat,
                triangulate_params
            )
            print(f"Point cloud size: {point_cloud_builder.points.shape[0]}")
            prev_frame = curr_frame

    # prev_frame = -1
    # print(view_mats)
    # while True:
    #     if None in view_mats:
    #         curr_frame = view_mats.index(None)
    #         view_mats[curr_frame], num_inliers = _view_matrix_from_pnp(
    #             point_cloud_builder,
    #             corner_storage[curr_frame],
    #             intrinsic_mat,
    #             pnp_params
    #         )
    #         known_view_mats.append(curr_frame)
    #
    #         if prev_frame != -1:
    #             _add_new_points(
    #                 point_cloud_builder,
    #                 corner_storage[curr_frame],
    #                 corner_storage[prev_frame],
    #                 view_mats[curr_frame],
    #                 view_mats[prev_frame],
    #                 intrinsic_mat,
    #                 triangulate_params
    #             )
    #         prev_frame = curr_frame
    #     else:
    #         break

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
