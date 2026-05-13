from typing import Dict, Optional

import numpy as np


class _InputStream:
    def __init__(self, bits: str):
        self.bits = bits
        self.index = 0

    def read(self, size: int) -> int:
        value = self.bits[self.index:self.index + size]
        self.index += size
        return int(value, 2) if value else 0


def _bytes_to_bits(data) -> str:
    return "".join(f"{int(byte):08b}" for byte in data)


def _decode_rle_fallback(rle):
    stream = _InputStream(_bytes_to_bits(rle))
    size = stream.read(32)
    word_size = stream.read(5) + 1
    run_sizes = [stream.read(4) + 1 for _ in range(4)]
    decoded = np.zeros(size, dtype=np.uint8)

    index = 0
    while index < size:
        is_run = stream.read(1)
        run_length = index + 1 + stream.read(run_sizes[stream.read(2)])
        if is_run:
            decoded[index:run_length] = stream.read(word_size)
            index = run_length
        else:
            while index < run_length:
                decoded[index] = stream.read(word_size)
                index += 1

    return decoded


def decode_brush_rle(rle, width: int, height: int) -> np.ndarray:
    """Decode a Label Studio BrushLabels RLE payload into a 2D binary mask."""
    decoder = None
    try:
        from label_studio_sdk.converter import brush
        decoder = brush.decode_rle
    except Exception:
        try:
            from label_studio_converter import brush
            decoder = brush.decode_rle
        except Exception:
            decoder = _decode_rle_fallback

    decoded = np.asarray(decoder(rle), dtype=np.uint8)
    expected = int(width) * int(height) * 4
    if decoded.size < expected:
        raise ValueError(
            f"Brush RLE decoded to {decoded.size} values, expected at least {expected}"
        )

    rgba = decoded[:expected].reshape((int(height), int(width), 4))
    return (rgba[:, :, 3] > 0).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> Optional[Dict[str, int]]:
    mask_bool = np.asarray(mask).astype(np.uint8) > 0
    if not mask_bool.any():
        return None

    ys, xs = np.nonzero(mask_bool)
    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1

    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def brush_region_bbox(region: Dict) -> Optional[Dict[str, int]]:
    value = region.get("value") or {}
    rle = value.get("rle")
    width = region.get("original_width") or value.get("original_width")
    height = region.get("original_height") or value.get("original_height")
    if rle is None or not width or not height:
        return None

    return mask_bbox(decode_brush_rle(rle, int(width), int(height)))
