"""Nesne takibi: her nesneye kalici bir ID verip kareler boyunca izleme.

Tespitten farki: `detect()` her kareyi sifirdan degerlendirir, dolayisiyla
"kac araba var" sorusunu kare bazinda cevaplar. Takip ise ayni arabayi
kareler boyunca ayni ID ile tanir; bu sayede "bu videodan toplam kac farkli
araba gecti" gibi gercek bir sayim yapilabilir.

Durum (ID'ler, izler, sayaclar) `TrackSession` icinde tutulur. Her video veya
webcam oturumu icin yeni bir TrackSession olusturulur.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.detector import Detector

# Ultralytics'in hazir tracker yapilandirmalari
BYTETRACK = "bytetrack.yaml"


@dataclass
class Track:
    """Belirli bir karede gorulen, kimlikli bir nesne."""

    track_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2


def color_for(track_id: int) -> tuple[int, int, int]:
    """ID'den deterministik bir BGR renk — ayni nesne hep ayni renkte cizilir."""
    hue = np.uint8([[[(track_id * 37) % 180, 200, 255]]])
    b, g, r = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


class LineCounter:
    """Sanal bir cizgiyi gecen nesneleri yonuyle birlikte sayar.

    Yontem: cizginin hangi tarafinda oldugumuzu vektorel carpimin isaretinden
    buluyoruz. Bir ID'nin isareti bir kareden digerine degistiyse cizgiyi
    gecmis demektir; degisim yonu de giris/cikis ayrimini verir.
    """

    def __init__(
        self,
        p1: tuple[int, int],
        p2: tuple[int, int],
        names: tuple[str, str] = ("forward", "backward"),
    ):
        # names[0] cizginin pozitif tarafina gecisi, names[1] negatif tarafa
        # gecisi adlandirir. Yatay/dikey cizgide "down/up", "right/left"
        # gibi anlamli isimler verilebilsin diye disaridan aliniyor.
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
    """Bir ID'nin ne zaman ve ne kadar goruldugu."""

    label: str
    first_frame: int
    last_frame: int
    frames: int = 1


@dataclass
class TrackSession:
    """Bir video/webcam oturumunun takip durumu.

    `fps` sure hesabi icin kullanilir. Video kare atlayarak (stride) islenirse
    buraya *efektif* fps verilmeli (orn. 30 fps / stride 2 = 15).
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
        """Tracker'in ic durumunu temizler.

        Model `@st.cache_resource` ile paylasildigi icin onceki videodan kalan
        ID'ler yeni videoya sizabilir; her oturum basinda sifirliyoruz.
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
        """Tek bir kareyi isler; cizilmis kareyi ve o karedeki nesneleri doner."""
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
                # Tracker bu kutuya henuz kimlik atamadi (yeni/kararsiz nesne).
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

    # --- ozet ------------------------------------------------------------

    def unique_counts(self) -> dict[str, int]:
        """Sinif basina kac *farkli* nesne gorulduğu."""
        counts = {label: len(ids) for label, ids in self.unique_ids.items()}
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def durations(self) -> list[dict]:
        """Her ID icin ekranda kalma suresi, uzundan kisaya."""
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
    """Arayuzdeki 'horizontal/vertical + %konum' secimini piksel koordinatina cevirir."""
    if orientation == "horizontal":
        # Soldan saga cizilen cizgide pozitif taraf asagisi olur.
        y = int(height * position)
        return LineCounter((0, y), (width, y), names=("down", "up"))

    # Dikey cizgiyi asagidan yukari cizeriz ki pozitif taraf sag olsun;
    # aksi halde saga dogru hareket "left" olarak sayilir.
    x = int(width * position)
    return LineCounter((x, height), (x, 0), names=("right", "left"))
