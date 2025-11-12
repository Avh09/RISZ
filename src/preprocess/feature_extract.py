import pandas as pd
import numpy as np
from math import sqrt, atan2, degrees
df = pd.read_csv("rawdata.csv")
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df = df.sort_values(["device_id", "user_id", "doc_type", "time"]).reset_index(drop=True)


def euclidean_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def stroke_features(stroke):
    """Compute all features for a given stroke (DataFrame)."""
   
    startX = stroke.iloc[0]["x_coor"]
    startY = stroke.iloc[0]["y_coor"]
    stopX = stroke.iloc[-1]["x_coor"]
    stopY = stroke.iloc[-1]["y_coor"]

    # --- Duration ---
    strokeDuration = stroke["time"].iloc[-1] - stroke["time"].iloc[0]

    # --- End-to-end distance ---
    directEndToEndDistance = euclidean_distance(startX, startY, stopX, stopY)

    # --- Trajectory length ---
    x, y = stroke["x_coor"].values, stroke["y_coor"].values
    lengthOfTrajectory = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

    # --- Mean resultant length (straightness) ---
    meanResultantLength = directEndToEndDistance / lengthOfTrajectory if lengthOfTrajectory != 0 else 0

    # --- Up/Down/Left/Right flag ---
    dx, dy = stopX - startX, stopY - startY
    if abs(dx) > abs(dy):
        upDownLeftRightFlag = "right" if dx > 0 else "left"
    else:
        upDownLeftRightFlag = "down" if dy > 0 else "up"

    # --- Direction of end-to-end line (angle in degrees) ---
    directionOfEndToEndLine = degrees(atan2(dy, dx)) if dx or dy else 0

    # --- Largest deviation from end-to-end line ---
    if directEndToEndDistance == 0:
        largestDeviationFromEndToEndLine = 0
    else:
        # Line equation Ax + By + C = 0
        A = stopY - startY
        B = startX - stopX
        C = (stopX * startY) - (startX * stopY)
        distances = np.abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)
        largestDeviationFromEndToEndLine = np.max(distances)

    # --- Average direction ---
    directions = []
    for i in range(1, len(stroke)):
        dx_i = x[i] - x[i - 1]
        dy_i = y[i] - y[i - 1]
        if dx_i != 0 or dy_i != 0:
            directions.append(degrees(atan2(dy_i, dx_i)))
    averageDirection = np.mean(directions) if directions else 0

    # --- Average velocity ---
    total_time = strokeDuration if strokeDuration != 0 else 1e-6
    averageVelocity = lengthOfTrajectory / total_time

    # --- Mid-stroke features ---
    mid_index = len(stroke) // 2
    midStrokePressure = stroke.iloc[mid_index]["pressure"]
    midStrokeArea = stroke.iloc[mid_index]["finger_area"]

    return pd.Series({
        "strokeDuration": strokeDuration,
        "startX": startX,
        "startY": startY,
        "stopX": stopX,
        "stopY": stopY,
        "directEndToEndDistance": directEndToEndDistance,
        "meanResultantLength": meanResultantLength,
        "upDownLeftRightFlag": upDownLeftRightFlag,
        "directionOfEndToEndLine": directionOfEndToEndLine,
        "largestDeviationFromEndToEndLine": largestDeviationFromEndToEndLine,
        "averageDirection": averageDirection,
        "lengthOfTrajectory": lengthOfTrajectory,
        "averageVelocity": averageVelocity,
        "midStrokePressure": midStrokePressure,
        "midStrokeArea": midStrokeArea
    })


# ============================
# 3. Stroke segmentation
# ============================
strokes = []
group_cols = ["device_id", "user_id", "doc_type"]
for _, group in df.groupby(group_cols):
    current_stroke = []
    for _, row in group.iterrows():
        action = row["action"]
        if action == 0:  # action_down
            current_stroke = [row]
        elif action == 2:  # action_move
            current_stroke.append(row)
        elif action == 1:  # action_up
            current_stroke.append(row)
            if len(current_stroke) > 1:
                stroke_df = pd.DataFrame(current_stroke)
                feat = stroke_features(stroke_df)
                for col in group_cols:
                    feat[col] = row[col]
                strokes.append(feat)
            current_stroke = []

# ============================
# 4. Output feature dataset
# ============================
features_df = pd.DataFrame(strokes)
features_df = features_df[group_cols + [c for c in features_df.columns if c not in group_cols]]

features_df.to_csv("features_extracted.csv", index=False)
print("Feature extraction complete! Saved as features_extracted.csv")
print(features_df.head())
