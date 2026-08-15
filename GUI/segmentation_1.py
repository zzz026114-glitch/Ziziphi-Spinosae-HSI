import scipy.io as sio
import numpy as np
import cv2

from scipy.ndimage import (
    binary_fill_holes,
    label,
    find_objects
)

from sklearn.cluster import KMeans


# =========================================================
# Load HSI
# =========================================================
def load_hsi(mat_path):

    mat = sio.loadmat(mat_path)

    cube = mat["data"]

    return cube.astype(np.float32)


# =========================================================
# Foreground Mask
# =========================================================
def get_foreground_mask(cube):

    band_idx = cube.shape[2] // 2

    band = cube[:, :, band_idx]

    img = cv2.normalize(
        band,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    th = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        61,
        2
    )

    th = binary_fill_holes(th)

    return th.astype(np.uint8)


# =========================================================
# Watershed
# =========================================================
def split_touching_objects(mask):

    binary = mask * 255

    dist = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )

    dist_norm = cv2.normalize(
        dist,
        None,
        0,
        1.0,
        cv2.NORM_MINMAX
    )

    _, sure_fg = cv2.threshold(
        dist_norm,
        0.35,
        1.0,
        cv2.THRESH_BINARY
    )

    sure_fg = (
        sure_fg * 255
    ).astype(np.uint8)

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    sure_bg = cv2.dilate(
        binary,
        kernel,
        iterations=3
    )

    unknown = cv2.subtract(
        sure_bg,
        sure_fg
    )

    _, markers = cv2.connectedComponents(
        sure_fg
    )

    markers = markers + 1

    markers[
        unknown == 255
    ] = 0

    color_img = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR
    )

    markers = cv2.watershed(
        color_img,
        markers
    )

    return markers > 1


# =========================================================
# Extract Regions
# =========================================================
def extract_regions(mask):

    lbl, num = label(mask)

    slices = find_objects(lbl)

    regions = []

    for i, sl in enumerate(slices):

        if sl is None:
            continue

        area = np.sum(
            lbl[sl] == (i + 1)
        )

        if area < 200:
            continue

        y0 = sl[0].start
        y1 = sl[0].stop

        x0 = sl[1].start
        x1 = sl[1].stop

        w = x1 - x0
        h = y1 - y0

        cx = x0 + w / 2
        cy = y0 + h / 2

        regions.append({

            "label": i + 1,

            "area": area,

            "slice": sl,

            "box": (
                x0,
                y0,
                w,
                h
            ),

            "center": (
                cx,
                cy
            )
        })

    return regions, lbl


# =========================================================
# Sort ROI
# 3 rows × 6 cols
# =========================================================
def sort_regions_grid(
        regions,
        n_rows=3
):

    centers_y = np.array([

        r["center"][1]

        for r in regions

    ]).reshape(-1, 1)

    kmeans = KMeans(

        n_clusters=n_rows,

        random_state=0,

        n_init=10

    )

    row_ids = kmeans.fit_predict(
        centers_y
    )

    row_centers = kmeans.cluster_centers_.flatten()

    row_order = np.argsort(
        row_centers
    )

    row_map = {

        old: new

        for new, old in enumerate(
            row_order
        )
    }

    for r, row in zip(
            regions,
            row_ids
    ):

        r["row"] = row_map[row]

    final_regions = []

    for row_idx in range(n_rows):

        row_regions = [

            r for r in regions

            if r["row"] == row_idx
        ]

        row_regions = sorted(

            row_regions,

            key=lambda x: x["center"][0]
        )

        final_regions.extend(
            row_regions
        )

    return final_regions


# =========================================================
# Extract ROI
# =========================================================
def extract_roi_list(
        cube,
        regions,
        lbl
):

    roi_list = []

    boxes = []

    for r in regions:

        idx = r["label"]

        sl = r["slice"]

        submask = (

                lbl[sl] == idx

        )

        roi = (

                cube[sl]

                * submask[:, :, None]

        )

        roi_list.append(
            roi
        )

        boxes.append(
            r["box"]
        )

    return roi_list, boxes


# =========================================================
# GUI Entry
# =========================================================
def segment_single_hsi(mat_path):

    cube = load_hsi(
        mat_path
    )

    mask = get_foreground_mask(
        cube
    )

    mask = split_touching_objects(
        mask
    )

    regions, lbl = extract_regions(
        mask
    )

    # ============================
    # Grid Sort
    # ============================

    regions = sort_regions_grid(

        regions,

        n_rows=3
    )

    roi_list, boxes = extract_roi_list(

        cube,

        regions,

        lbl
    )

    return (

        cube,

        mask,

        roi_list,

        boxes
    )