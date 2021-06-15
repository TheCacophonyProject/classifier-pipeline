import numpy as np
import cv2

min_left = 1
min_top = 1
max_right = 159
max_bottom = 119
edge_bounds = [min_left, min_top, max_right, max_bottom]


def preprocess_frame(frame, bound, out_dim):
    frame = frame.filtered
    if np.sum(frame) == 0:
        print(f"discarding zero frame")
        return None
    else:
        maxdim = max(frame.shape)
        if maxdim > out_dim:
            mindim = min(frame.shape)
            diff = maxdim - mindim
            if diff:
                f = np.zeros((maxdim, maxdim), dtype=np.float32)
                c0, c1 = center_position(
                    bound.left, bound.right, min_left, max_right, maxdim
                )
                r0, r1 = center_position(
                    bound.top, bound.bottom, min_top, max_bottom, maxdim
                )
                f[r0:r1, c0:c1] = frame
            if any(
                a == b
                for a, b in zip(
                    [bound.left, bound.top, bound.right, bound.bottom], edge_bounds
                )
            ):
                dim = out_dim
            else:
                dim = maxdim - int((maxdim - out_dim) / 4 + 1) * 4
            f = cv2.resize(f, (dim, dim), interpolation=cv2.INTER_AREA)
        else:
            if bound.left == min_left:
                f = np.zeros((frame.shape[0], out_dim))
                f[: frame.shape[0], :] = frame
            elif bound.right == max_right:
                f = np.zeros((frame.shape[0], out_dim))
                f[:, out_dim - frame.shape[1] :] = frame
            if bound.top == min_top:
                f = np.zeros((out_dim, frame.shape[1]))
                f[:, : frame.shape[1]] = frame
            elif bound.bottom == max_bottom:
                f = np.zeros((out_dim, frame.shape[1]))
                f[out_dim - frame.shape[0] :, :] = frame
            else:
                f = frame
    f = center_frame(f, out_dim)
    f = f[:, :, np.newaxis]
    return f


def center_frame(frame, out_dim):
    out_frame = np.zeros((out_dim, out_dim), np.float32)
    y_start = (out_dim - frame.shape[0]) // 2
    x_start = (out_dim - frame.shape[1]) // 2

    out_frame[
        y_start : y_start + frame.shape[0], x_start : x_start + frame.shape[1]
    ] = frame
    return out_frame


def center_position(low_bound, high_bound, low_limit, high_limit, space):
    size = high_bound - low_bound
    extra = space - size
    if extra > 0:
        if low_bound == low_limit:
            return 0, size
        elif high_bound == high_limit:
            return space - size, space
        else:
            pad = int(extra / 2) + np.random.randint(extra % 2 + 1)
            return pad, size + pad
    else:
        return 0, size


def sum_weighted(predicts, weights):
    return np.matmul(weights.T, predicts)
