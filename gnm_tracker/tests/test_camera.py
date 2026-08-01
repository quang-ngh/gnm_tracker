import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gnm_tracker.model.camera import PerspectiveCamera, axis_angle_to_matrix  # noqa: E402


def test_axis_angle_identity():
    r = axis_angle_to_matrix(torch.zeros(3))
    assert torch.allclose(r, torch.eye(3), atol=1e-6)


def test_axis_angle_180_about_x():
    r = axis_angle_to_matrix(torch.tensor([np.pi, 0.0, 0.0]))
    assert torch.allclose(r, torch.diag(torch.tensor([1.0, -1.0, -1.0])), atol=1e-5)


def test_projection_to_principal_point(cfg):
    cam = PerspectiveCamera.from_config(cfg, (480, 640))
    cx, cy = cam.principal_point.tolist()
    # A world point that maps to camera [0,0,Z] must project to the principal point.
    target_cam = torch.tensor([0.0, 0.0, 0.5])
    world = cam.r_wc.T @ target_cam
    uv, z = cam.project(world)
    assert torch.allclose(uv, torch.tensor([cx, cy]), atol=1e-3)
    assert z.item() > 0


def test_up_maps_to_smaller_v(cfg):
    """+Y in GNM (up) must land higher in the image (smaller v).

    A point in front of the camera has negative GNM-world z (r_wc = diag(1,-1,-1)
    sends -z_world -> +z_cam); +Y_world (up) -> -y_cam -> smaller v (image up).
    """
    cam = PerspectiveCamera.from_config(cfg, (480, 640))
    cy = cam.principal_point[1].item()
    world = torch.tensor([0.0, 0.05, -0.5])  # GNM-world: 5cm up, in front
    uv, z = cam.project(world)
    assert z.item() > 0        # in front of the camera
    assert uv[1].item() < cy   # up in GNM -> higher in image
