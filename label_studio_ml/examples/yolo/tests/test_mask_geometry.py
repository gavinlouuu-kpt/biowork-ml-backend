import numpy as np

from utils import mask_geometry


def test_mask_bbox_returns_pixel_bounds():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    assert mask_geometry.mask_bbox(mask) == {
        "x": 3,
        "y": 2,
        "width": 4,
        "height": 3,
    }


def test_brush_region_bbox_decodes_rle_mask(monkeypatch):
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:5] = 1

    monkeypatch.setattr(mask_geometry, "decode_brush_rle", lambda rle, width, height: mask)

    region = {
        "type": "brushlabels",
        "original_width": 5,
        "original_height": 4,
        "value": {
            "format": "rle",
            "rle": [1, 2, 3],
            "brushlabels": ["Cell"],
        },
    }

    assert mask_geometry.brush_region_bbox(region) == {
        "x": 2,
        "y": 1,
        "width": 3,
        "height": 2,
    }
