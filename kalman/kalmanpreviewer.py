from ml_tools.previewer import Previewer


class KalmanPreviewer(Previewer):
    def add_tracks(
        self, draw, tracks, frame_number, track_predictions=None, screen_bounds=None
    ):
        # look for any tracks that occur on this frame
        for index, track in enumerate(tracks):
            frame_offset = frame_number - track.start_frame
            if frame_offset >= 0 and frame_offset < len(track.bounds_history) - 1:
                # draw frame
                rect = track.bounds_history[frame_offset]
                draw.rectangle(
                    self.rect_points(rect),
                    outline=self.TRACK_COLOURS[index % len(self.TRACK_COLOURS)],
                )

                # draw centre
                xx = rect.mid_x * 4.0
                yy = rect.mid_y * 4.0
                center = track.bounds_history[frame_offset].mid_x
                draw.arc((xx - 4, yy - 4, xx + 4, yy + 4), 0, 360)
