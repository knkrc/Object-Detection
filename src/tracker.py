"""Object tracking: give each object a persistent ID and follow it across frames.

How this differs from detection: `detect()` evaluates every frame from scratch,
so it can only answer "how many cars are there" per frame. Tracking recognises
the same car across frames under the same ID, which makes a real count possible:
"how many distinct cars passed through this video".

State (IDs, trails, counters) lives in `TrackSession`. A new session is created
for every video or webcam run.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.detector import Detector

# Tracker configurations that ship with ultralytics
BYTETRACK = "bytetrack.yaml"


@dataclass
class Track:
    """An identified object seen in one particular frame."""

    track_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


def color_for(track_id: int) -> tuple[int, int, int]:
    """A deterministic BGR colour from an ID, so an object keeps one colour."""
    hue = np.uint8([[[(track_id * 37) % 180, 200, 255]]])
    b, g, r = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


class LineCounter:
    """Counts objects crossing a virtual line, with direction.

    The method: the sign of the cross product tells us which side of the line a
    point is on. If an ID's sign flips between two frames it has crossed, and
    the direction of the flip separates one way from the other.
    """

    def __init__(
        self,
        p1: tuple[int, int],
        p2: tuple[int, int],
        names: tuple[str, str] = ("forward", "backward"),
    ):
        # names[0] labels a crossing to the line's positive side, names[1] the
        # negative one. Passed in so a horizontal or vertical line can use
        # meaningful names like "down/up" or "right/left".
        self.p1 = p1
        self.p2 = p2
        self.names = names
        self.counts = {names[0]: 0, names[1]: 0}
        self._last_side: dict[int, int] = {}
        self.crossed: set[int] = set()

    def _side(self, point: tuple[int, int]) -> int:
        (x1, y1), (x2, y2) = self.p1, self.p2
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return 1 if cross > 0 else -1 if cross < 0 else 0

    def update(self, track_id: int, center: tuple[int, int]) -> str | None:
        side = self._side(center)
        if side == 0:
            return None

        previous = self._last_side.get(track_id)
        self._last_side[track_id] = side

        if previous is None or previous == side:
            return None

        direction = self.names[0] if side > 0 else self.names[1]
        self.counts[direction] += 1
        self.crossed.add(track_id)
        return direction

    def draw(self, frame: np.ndarray) -> None:
        cv2.line(frame, self.p1, self.p2, (0, 255, 255), 2)
        label = "  ".join(f"{name}: {count}" for name, count in self.counts.items())
        cv2.putText(
            frame,
            label,
            (self.p1[0], max(self.p1[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )


@dataclass
class _Seen:
    """When and for how long an ID was seen."""

    label: str
    first_frame: int
    last_frame: int
    frames: int = 1


@dataclass
class TrackSession:
    """Tracking state for one video or webcam session.

    `fps` is used for the duration figures. If the video is processed with a
    stride, pass the *effective* fps here (e.g. 30 fps with stride 2 = 15).
    """

    detector: Detector
    conf: float = 0.35
    keep_classes: list[str] | None = None
    tracker_cfg: str = BYTETRACK
    fps: float = 25.0
    trail_length: int = 32
    draw_trails: bool = True
    line: LineCounter | None = None

    frame_index: int = field(default=0, init=False)
    unique_ids: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set), init=False)
    trails: dict[int, deque] = field(default_factory=dict, init=False)
    seen: dict[int, _Seen] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clears the tracker's internal state.

        The model is shared through `@st.cache_resource`, so IDs left over from
        a previous video can leak into the next one. We reset at session start.
        """
        self.frame_index = 0
        self.unique_ids = defaultdict(set)
        self.trails = {}
        self.seen = {}

        predictor = getattr(self.detector.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", []) or []:
            if hasattr(tracker, "reset"):
                tracker.reset()

    def step(self, frame: np.ndarray) -> tuple[np.ndarray, list[Track]]:
        """Processes one frame; returns the annotated frame and its objects."""
        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf,
            classes=self.detector.class_ids(self.keep_classes),
            verbose=False,
        )
        result = results[0]
        names = self.detector.model.names

        tracks: list[Track] = []
        for box in result.boxes:
            if box.id is None:
                # The tracker has not assigned an ID yet (new or unstable object).
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            tracks.append(
                Track(
                    track_id=int(box.id[0]),
                    label=names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=(x1, y1, x2, y2),
                )
            )

        annotated = result.plot()
        self._update_state(tracks)

        if self.draw_trails:
            self._draw_trails(annotated)
        if self.line:
            self.line.draw(annotated)

        self.frame_index += 1
        return annotated, tracks

    def _update_state(self, tracks: list[Track]) -> None:
        for track in tracks:
            tid = track.track_id
            self.unique_ids[track.label].add(tid)

            record = self.seen.get(tid)
            if record is None:
                self.seen[tid] = _Seen(track.label, self.frame_index, self.frame_index)
            else:
                record.last_frame = self.frame_index
                record.frames += 1

            trail = self.trails.setdefault(tid, deque(maxlen=self.trail_length))
            trail.append(track.center)

            if self.line:
                self.line.update(tid, track.center)

    def _draw_trails(self, frame: np.ndarray) -> None:
        for tid, points in self.trails.items():
            if len(points) < 2:
                continue
            cv2.polylines(
                frame,
                [np.array(points, dtype=np.int32)],
                isClosed=False,
                color=color_for(tid),
                thickness=2,
            )

    # --- summary ---------------------------------------------------------

    def unique_counts(self) -> dict[str, int]:
        """How many *distinct* objects were seen, per class."""
        counts = {label: len(ids) for label, ids in self.unique_ids.items()}
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def durations(self) -> list[dict]:
        """Time on screen for each ID, longest first."""
        rows = [
            {
                "id": tid,
                "object": record.label,
                "seconds": round(record.frames / self.fps, 2) if self.fps else 0.0,
                "frames": record.frames,
                "first_frame": record.first_frame,
                "last_frame": record.last_frame,
            }
            for tid, record in self.seen.items()
        ]
        return sorted(rows, key=lambda r: -r["seconds"])

    def summary(self) -> dict:
        return {
            "unique": self.unique_counts(),
            "total_objects": len(self.seen),
            "line": dict(self.line.counts) if self.line else None,
            "frames": self.frame_index,
        }


def line_from_ratio(width: int, height: int, orientation: str, position: float) -> LineCounter:
    """Turns the UI's 'horizontal/vertical + position' choice into pixels."""
    if orientation == "horizontal":
        # For a line drawn left to right, the positive side is below it.
        y = int(height * position)
        return LineCounter((0, y), (width, y), names=("down", "up"))

    # We draw the vertical line bottom to top so the positive side is the right;
    # otherwise rightward movement would be counted as "left".
    x = int(width * position)
    return LineCounter((x, height), (x, 0), names=("right", "left"))
