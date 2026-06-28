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
