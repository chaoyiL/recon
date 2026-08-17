import tempfile
import unittest
from pathlib import Path

import numpy as np

from get_surface import (point_set_from_surface_grids,
                         save_generated_uv_xyz_map,save_uv_xyz_map)
from utils.process import build_reconstruction_point_set


class UvXyzMapSaveTest(unittest.TestCase):
    def test_jax_surface_grids_convert_to_aligned_point_set(self) -> None:
        surface_count=2; surface_rows=3; columns=4
        rows=surface_count*surface_rows
        row,column=np.meshgrid(
            np.arange(rows,dtype=np.float32),
            np.arange(columns,dtype=np.float32),indexing="ij")
        xyz=np.stack([column,row,100+row],axis=-1)
        uv=np.stack([20+2*column,30+3*row],axis=-1)
        one_s,one_t=np.meshgrid(
            np.linspace(0,1,surface_rows,dtype=np.float32),
            np.linspace(0,1,columns,dtype=np.float32),indexing="ij")
        st=np.tile(np.stack([one_s,one_t],axis=-1),(surface_count,1,1))
        depth=xyz[...,2]
        K=np.asarray([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]])
        point_set=point_set_from_surface_grids(
            xyz,uv,st,depth,K,np.zeros(5),surface_count=surface_count,
            surface_rows=surface_rows)
        self.assertEqual(point_set.xyz.shape,(rows*columns,3))
        np.testing.assert_array_equal(
            point_set.source_index,
            np.repeat(np.arange(surface_count),surface_rows*columns))
        np.testing.assert_array_equal(
            point_set.is_edge.reshape(rows,columns)[:,[0,-1]],True)
        self.assertFalse(np.any(point_set.is_edge.reshape(rows,columns)[:,1:-1]))

    def test_npz_keeps_all_point_rows_aligned(self) -> None:
        K = np.asarray(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
        )
        left = np.asarray([[-1.0, 0.0, 100.0], [-1.0, 1.0, 100.0]])
        right = np.asarray([[1.0, 0.0, 100.0], [1.0, 1.0, 100.0]])
        point_set, _, _ = build_reconstruction_point_set(
            left,
            right,
            K,
            np.zeros(5),
            np.zeros(3),
            0.0,
            n_fill=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = save_uv_xyz_map(Path(directory) / "surface.png", point_set)
            with np.load(output) as saved:
                np.testing.assert_allclose(saved["xyz"], point_set.xyz)
                np.testing.assert_allclose(saved["uv"], point_set.uv)
                np.testing.assert_allclose(saved["st"], point_set.st)
                np.testing.assert_allclose(
                    saved["undistorted_uv"], point_set.undistorted_uv
                )
                np.testing.assert_allclose(saved["camera_depth"], point_set.camera_depth)
                np.testing.assert_array_equal(saved["is_edge"], point_set.is_edge)
                self.assertEqual(saved["xyz"].shape[0], saved["uv"].shape[0])

    def test_stream_map_is_loadable(self) -> None:
        point_set, _, _ = build_reconstruction_point_set(
            np.asarray([[-1.,0.,100.],[-1.,1.,100.]]),
            np.asarray([[1.,0.,100.],[1.,1.,100.]]),
            np.asarray([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]]),
            np.zeros(5), np.zeros(3), 0., n_fill=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_generated_uv_xyz_map(Path(directory) / "current.npz", point_set)
            with np.load(path) as saved:
                np.testing.assert_allclose(saved["xyz"], point_set.xyz)
                np.testing.assert_allclose(saved["st"], point_set.st)


if __name__ == "__main__":
    unittest.main()
