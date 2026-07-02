import attr
import cv2
import numpy as np
from ml_tools.imageprocessing import resize_cv, rotate, normalize, resize_and_pad
import enum
import logging


class TrackChannels(enum.Enum):
    """Indexes to channels in track."""

    thermal = 0
    filtered = 1
    mask = 4
    raw = 6
    thermal_norm = 7


@attr.s(slots=True, eq=False)
class Frame:
    thermal = attr.ib()
    filtered = attr.ib()
    frame_number = attr.ib()
    mask = attr.ib(default=None)
    thermal_norm = attr.ib(default=None)
    scaled_thermal = attr.ib(default=None)
    ffc_affected = attr.ib(default=False)
    region = attr.ib(default=None)
    frame_temp_median = attr.ib(default=None)
    preprocessed = attr.ib(default=False)

    def get_channel(self, channel):
        # just leave this top one for old style
        # will remove soon obselete 08/06/2026
        if channel == TrackChannels.thermal:
            return self.thermal
        elif channel == TrackChannels.raw:
            return self.thermal
        elif channel == TrackChannels.filtered:
            return self.filtered
        elif channel == TrackChannels.mask:
            return self.mask
        elif channel == TrackChannels.thermal_norm:
            return self.thermal_norm
        return None

    @classmethod
    def from_channels(
        cls,
        frame,
        channels,
        frame_number,
        ffc_affected=False,
        region=None,
    ):
        f = cls(
            None,
            None,
            frame_number,
            ffc_affected=ffc_affected,
            region=region,
        )
        for channel, data in zip(channels, frame):
            if TrackChannels.thermal == channel:
                f.thermal = data
            if TrackChannels.filtered == channel:
                f.filtered = data
            if TrackChannels.mask == channel:
                f.mask = data
        return f

    @classmethod
    def from_array(
        cls,
        frame_arr,
        frame_number,
        ffc_affected=False,
        region=None,
    ):
        t = frame_arr[TrackChannels.thermal]
        f = frame_arr[TrackChannels.filtered]
        if len(frame_arr) >= 3:
            m = frame_arr[2]
        else:
            m = None
        return cls(
            t,
            f,
            frame_number,
            mask=m,
            region=region,
            ffc_affected=ffc_affected,
        )

    def as_array(self):
        data = [self.thermal]
        if self.filtered is not None:
            data.append(self.filtered)
        else:
            return np.array(data)
        if self.mask is not None:
            data.append(self.mask)
        return np.asarray(data)

    def normalize(self):
        if self.thermal is not None:
            self.thermal, _ = normalize(self.thermal, new_max=255)
        if self.filtered is not None:
            self.filtered, _ = normalize(self.filtered, new_max=255)

    def brightness_adjust(self, adjust):
        if self.thermal is not None:
            self.thermal += adjust

    def contrast_adjust(self, adjust):
        if self.thermal is not None:
            self.thermal *= adjust
        if self.filtered is not None:
            self.filtered *= adjust

    def crop_by_region_with_padding(self, region, crop_rectangle, resize_dim):
        top = 0
        left = 0
        # if our region has becomes out of bounds i.e. negative x or y we need to calculated the top offset
        if region.width < crop_rectangle.width and crop_rectangle.left > region.left:
            left = crop_rectangle.left - region.left

        #
        if region.height < crop_rectangle.height and crop_rectangle.top > region.top:
            top = crop_rectangle.top - region.top

        new_width = min(crop_rectangle.width, region.width)
        new_height = min(crop_rectangle.height, region.height)
        new_width = max(resize_dim, new_width)
        new_height = max(resize_dim, new_height)
        cropped_region = region.copy()
        cropped_region.crop(crop_rectangle)

        thermal = None
        filtered = None
        mask = None
        if self.thermal is not None:
            sub_thermal = cropped_region.subimage(self.thermal)
            thermal = np.full(
                (new_height, new_width), np.amin(sub_thermal), dtype=np.float32
            )
            thermal[
                top : cropped_region.height + top, left : cropped_region.width + left
            ] = sub_thermal

        if self.filtered is not None:
            sub_filtered = cropped_region.subimage(self.filtered)
            filtered = np.full(
                (new_height, new_width), np.amin(sub_filtered), dtype=np.float32
            )
            filtered[
                top : cropped_region.height + top, left : cropped_region.width + left
            ] = sub_filtered

        if self.mask is not None:
            sub_mask = cropped_region.subimage(self.mask)
            mask = np.zeros((new_height, new_width), dtype=sub_mask.dtype)
            mask[
                top : cropped_region.height + top, left : cropped_region.width + left
            ] = sub_mask

        return Frame(
            thermal,
            filtered,
            self.frame_number,
            mask=mask,
            ffc_affected=self.ffc_affected,
            region=cropped_region,
        )

    def crop_by_region(self, region, only_thermal=False, out=None):
        thermal = None
        filtered = None
        mask = None
        if self.thermal is not None:
            thermal = region.subimage(self.thermal)
        if not only_thermal:
            if self.filtered is not None:
                filtered = region.subimage(self.filtered)
            if self.mask is not None:
                mask = region.subimage(self.mask)
        if out:
            out.thermal = thermal
            out.filtered = filtered
            out.mask = mask
            out.region = region
            frame = out
        else:
            frame = Frame(
                thermal,
                filtered,
                self.frame_number,
                mask=mask,
                ffc_affected=self.ffc_affected,
                region=region,
            )
        return frame

    def resize_with_aspect(
        self,
        dim,
        crop_rectangle,
        keep_edge=False,
        edge_offset=(0, 0, 0, 0),
        original_region=None,
        interpolation=cv2.INTER_NEAREST,
        no_padding=False,
    ):
        if self.thermal is not None:
            self.thermal = resize_and_pad(
                self.thermal,
                dim,
                self.region,
                crop_rectangle,
                keep_edge=keep_edge,
                edge_offset=edge_offset,
                original_region=original_region,
                interpolation=interpolation,
            )
        if self.thermal_norm is not None:
            self.thermal_norm = resize_and_pad(
                self.thermal_norm,
                dim,
                self.region,
                crop_rectangle,
                keep_edge=keep_edge,
                edge_offset=edge_offset,
                original_region=original_region,
                interpolation=interpolation,
            )
        if self.mask is not None:
            self.mask = resize_and_pad(
                self.mask,
                dim,
                self.region,
                crop_rectangle,
                keep_edge=keep_edge,
                original_region=original_region,
                interpolation=interpolation,
                edge_offset=edge_offset,
            )
        if self.filtered is not None:
            self.filtered = resize_and_pad(
                self.filtered,
                dim,
                self.region,
                crop_rectangle,
                keep_edge=keep_edge,
                edge_offset=edge_offset,
                original_region=original_region,
                interpolation=interpolation,
            )

    def resize(self, dim, interpolation=cv2.INTER_NEAREST):
        self.thermal = resize_cv(self.thermal, dim, interpolation=interpolation)
        self.filtered = resize_cv(self.filtered, dim, interpolation=interpolation)
        self.thermal_norm = resize_cv(
            self.thermal_norm, dim, interpolation=interpolation
        )
        if self.mask is not None:
            self.mask = resize_cv(self.mask, dim, interpolation=interpolation)

    def rotate(self, degrees):
        if self.thermal is not None:
            self.thermal = rotate(self.thermal, degrees)
        if self.mask is not None:
            self.mask = rotate(self.mask, degrees)
        if self.filtered is not None:
            self.filtered = rotate(self.filtered, degrees)

    def float_arrays(self):
        if self.thermal is not None:
            self.thermal = np.float32(self.thermal)
        if self.mask is not None:
            self.mask = np.float32(self.mask)
        if self.filtered is not None:
            self.filtered = np.float32(self.filtered)

    def copy(self):
        return Frame(
            None if self.thermal is None else self.thermal.copy(),
            None if self.filtered is None else self.filtered.copy(),
            self.frame_number,
            mask=None if self.mask is None else self.mask.copy(),
            ffc_affected=self.ffc_affected,
            region=None if self.region is None else self.region.copy(),
        )

    def flip(self):
        if self.thermal is not None:
            self.thermal = np.flip(self.thermal, axis=1)
        if self.mask is not None:
            self.mask = np.flip(self.mask, axis=1)
        if self.filtered is not None:
            self.filtered = np.flip(self.filtered, axis=1)

    @property
    def shape(self):
        return self.thermal.shape



def repeat_with_thresh(border, filtered_thresh):
    """For each position in border, find the index of the pixel that should be
    repeated into it, skipping over runs where the value is over filtered_thresh
    (possible animal on border). -1 means no valid pixel was found to use."""
    indices = np.full(len(border), -1, dtype=int)
    prev_idx = None
    fill_from = None
    # logging.info("Repeat with thresh %s border %s",filtered_thresh, border)
    for i, val in enumerate(border):
        if val > filtered_thresh or val == -1:
            if prev_idx is not None:
                indices[i] = prev_idx
            elif fill_from is None:
                fill_from = i
        else:
            if fill_from is not None:
                indices[fill_from:i] = i
                # logging.info("Filling border %s with values %s",fill_from,val)

                fill_from = None
            indices[i] = i
            prev_idx = i
            # logging.info("Filling border %s with %s",i,val)
    return indices


def gather_border(values, indices, default_val):
    """Builds a border using indices (as returned by repeat_with_thresh),
    falling back to default_val wherever indices is -1."""
    border_fill = np.full_like(values, default_val)
    valid = indices >= 0
    border_fill[valid] = values[indices[valid]]
    return border_fill


def repeat_border(frame, new_height, new_width, top, left, filtered_thresh, pad_values):
    """Pad channels to (new_height, new_width), placing existing content at (top, left)."""
    h, w = frame.thermal.shape[:2]

    padded_thermal = np.full((new_height, new_width), -1, dtype=frame.thermal.dtype)
    padded_thermal[top : top + h, left : left + w] = frame.thermal

    padded_thermal_norm = np.full(
        (new_height, new_width), -1, dtype=frame.thermal_norm.dtype
    )
    padded_thermal_norm[top : top + h, left : left + w] = frame.thermal_norm

    padded_filtered = np.full((new_height, new_width), -1, dtype=frame.filtered.dtype)
    padded_filtered[top : top + h, left : left + w] = frame.filtered

    channels = (
        (padded_thermal, pad_values.thermal),
        (padded_thermal_norm, pad_values.thermal_norm),
        (padded_filtered, pad_values.filtered),
    )
    # pad each side with its repeated border instead of the flat pad value, unless
    # the border itself contains an animal there. Which pixel index to repeat is
    # always decided from the filtered channel; thermal and thermal_norm reuse
    # those same indices against their own values. Processed left, top, right,
    # bottom in that order so each side's full-length fill picks up the previous
    # sides' fills already sitting in its corners.
    pad_left = left > 0
    if pad_left:
        indices = repeat_with_thresh(padded_filtered[:, left], filtered_thresh)
        for padded, default_val in channels:
            border_fill = gather_border(padded[:, left], indices, default_val)
            padded[:, :left] = border_fill[:, np.newaxis]

    pad_top = top > 0
    if pad_top:
        indices = repeat_with_thresh(padded_filtered[top, :], filtered_thresh)
        for padded, default_val in channels:
            border_fill = gather_border(padded[top, :], indices, default_val)
            padded[:top, :] = border_fill[np.newaxis, :]

    pad_right = left + w < new_width
    if pad_right:
        indices = repeat_with_thresh(padded_filtered[:, left + w - 1], filtered_thresh)
        for padded, default_val in channels:
            border_fill = gather_border(padded[:, left + w - 1], indices, default_val)
            padded[:, left + w :] = border_fill[:, np.newaxis]

    pad_bottom = top + h < new_height
    if pad_bottom:
        indices = repeat_with_thresh(padded_filtered[top + h - 1, :], filtered_thresh)
        for padded, default_val in channels:
            border_fill = gather_border(padded[top + h - 1, :], indices, default_val)
            padded[top + h :, :] = border_fill[np.newaxis, :]

    frame.thermal = padded_thermal
    frame.thermal_norm = padded_thermal_norm
    frame.filtered = padded_filtered

    # if np.any(frame.filtered==0):
    #     import cv2
    #     image = np.uint8(frame.filtered *255)
    #     cv2.imshow("f",image)
    #     cv2.waitKey()
    assert np.all(frame.thermal > -1)
    assert np.all(frame.filtered > -1)
    assert np.all(frame.thermal_norm > -1)
    assert np.amax(frame.filtered) < 1.1, "filtered is normalized 0-1"
    assert filtered_thresh < 1, " thresh should be less than 1 too"

