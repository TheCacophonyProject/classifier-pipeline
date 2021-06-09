ut_dim = 32
min_left = 1
min_top = 1
max_right = 159
max_bottom = 119
edge_bounds = [min_left, min_top, max_right, max_bottom]
classes = [
    "bird",
    "cat",
    "hedgehog",
    "human",
    "leporidae",
    "mustelid",
    "possum",
    "rodent",
    "unclassified",
    "vehicle",
    "wallaby",
]
unclassified_tags = {"false-positive", "insect"}
tag_class_map = {k: k for k in classes}
tag_class_map.update({k: "unclassified" for k in unclassified_tags})


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
            if any(a == b for a, b in zip(bound, edge_bounds)):
                dim = out_dim
            else:
                dim = maxdim - int((maxdim - out_dim) / 4 + 1) * 4
            f = cv2.resize(f, (dim, dim), interpolation=cv2.INTER_AREA)
        else:
            f = np.zeros((out_dim, out_dim), dtype=np.float32)
            if bound.left == min_left:
                f[: frame.shape[0], :] = frame
            elif bound.right == max_right:
                f[out_dim - frame.shape[0] :, :] = frame
            if bound.top == min_top:
                f[:, : frame.shape[1]] = frame
            elif bound.bottom == max_bottom:
                f[space - frame.shape[0] :, :] = frame
    f = jitter_crop(f, out_dim, 1)
    return f


def jitter_crop(frame, out_dim, jitter):
    out_frame = np.zeros((out_dim, out_dim), np.float32)

    center_position_y = (out_dim - frame.shape[0]) / 2
    center_position_x = (out_dim - frame.shape[1]) / 2
    out_frame[
        center_position_y + np.random.randint(1) : frame.shape[0],
        center_position_x + np.random.randint(1) : frame.shape[1],
    ] = frame
    return out_frame[:, :, np.newaxis]


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


sample_dims = (32, 32)
for track in tracks:
    tag = track.tag
    if tag in unclassified_tags:
        tag = "unclassified"
    actuals.append(class_to_index[tag])
    frames = track.frames
    frames = np.array(
        [_format_sample(sample_dims, f, maximum_offset, False) for f in frames]
    )
    predicts = model.predict(frames)
    predicts_squared = predicts ** 2
    pixelcount_weights = np.array([(f > 0).sum() for f in frames])
    label = np.argmax(sum_weighted(predicts_squared, pixelcount_weights))


def sum_weighted(predicts, weights):
    return np.matmul(weights.T, predicts)
