import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class RecordInfo:
    frame_no: int
    kind: str
    viewport_h: int
    viewport_w: int
    render_target_h: int
    render_target_w: int
    path: str


PATTERN = re.compile(
    r"^(?P<frame>\d+)_"
    r"(?P<kind>SceneColor|SceneDepth|CameraMotion)_"
    r"(?P<viewport_h>\d+)x(?P<viewport_w>\d+)_in_"
    r"(?P<render_target_h>\d+)x(?P<render_target_w>\d+)\.data$"
)


def parse_records(record_dir: str) -> List[RecordInfo]:
    records: List[RecordInfo] = []
    for name in os.listdir(record_dir):
        match = PATTERN.match(name)
        if not match:
            continue
        records.append(
            RecordInfo(
                frame_no=int(match.group("frame")),
                kind=match.group("kind"),
                viewport_h=int(match.group("viewport_h")),
                viewport_w=int(match.group("viewport_w")),
                render_target_h=int(match.group("render_target_h")),
                render_target_w=int(match.group("render_target_w")),
                path=os.path.join(record_dir, name),
            )
        )
    return records


def index_records(records: Iterable[RecordInfo]) -> Dict[Tuple[int, str, int, int, int, int], str]:
    index: Dict[Tuple[int, str, int, int, int, int], str] = {}
    for rec in records:
        key = (rec.frame_no, rec.kind, rec.viewport_h, rec.viewport_w, rec.render_target_h, rec.render_target_w)
        index[key] = rec.path
    return index


def copy_triplet(
    index: Dict[Tuple[int, str, int, int, int, int], str],
    out_dir: str,
    frame_start: int,
    dest_frame_no: int,
    viewport_h: int,
    viewport_w: int,
    render_target_h: int,
    render_target_w: int,
    kind: str,
) -> bool:
    keys = [
        (frame_start + offset, kind, viewport_h, viewport_w, render_target_h, render_target_w) for offset in range(3)
    ]
    if not all(key in index for key in keys):
        return False
    dest_dir = os.path.join(
        out_dir,
        "anime",
        f"{dest_frame_no:06d}",
        f"{render_target_h}x{render_target_w}",
        f"{viewport_h}x{viewport_w}",
    )
    os.makedirs(dest_dir, exist_ok=True)
    for idx, key in enumerate(keys):
        src = index[key]
        if kind == "SceneColor":
            dst_name = f"color_{idx}.data"
        elif kind == "SceneDepth":
            dst_name = f"depth_{idx}.data"
        elif kind == "CameraMotion":
            dst_name = f"motion_{idx}.data"
        else:
            return False
        shutil.copy2(src, os.path.join(dest_dir, dst_name))
    return True


def generate_dataset(record_dir: str, out_dir: str) -> None:
    records = parse_records(record_dir)
    if not records:
        print("no records found")
        return
    index = index_records(records)
    combos = sorted(
        {
            (rec.viewport_h, rec.viewport_w, rec.render_target_h, rec.render_target_w)
            for rec in records
        }
    )
    frames = sorted({rec.frame_no for rec in records})
    frame_set = set(frames)

    valid_frame_starts = [
        frame_start
        for frame_start in frames
        if all((frame_start + offset) in frame_set for offset in range(3))
    ]

    copied = 0
    for dest_frame_no, frame_start in enumerate(valid_frame_starts):
        for viewport_h, viewport_w, render_target_h, render_target_w in combos:
            color_ok = copy_triplet(
                index,
                out_dir,
                frame_start,
                dest_frame_no,
                viewport_h,
                viewport_w,
                render_target_h,
                render_target_w,
                "SceneColor",
            )
            depth_ok = copy_triplet(
                index,
                out_dir,
                frame_start,
                dest_frame_no,
                viewport_h,
                viewport_w,
                render_target_h,
                render_target_w,
                "SceneDepth",
            )
            camera_motion_ok = copy_triplet(
                index,
                out_dir,
                frame_start,
                dest_frame_no,
                viewport_h,
                viewport_w,
                render_target_h,
                render_target_w,
                "CameraMotion",
            )
            
            if color_ok and depth_ok and camera_motion_ok:
                copied += 1
    print(f"done, copied triplets: {copied}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NRSRecord to train dataset")
    parser.add_argument("--record-dir", required=True, help="NRSRecord directory")
    parser.add_argument("--out-dir", default="train_data", help="output dataset root")
    parser.add_argument("--force", action="store_true", help="remove existing out-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.path.exists(args.out_dir):
        if not args.force:
            print(f"warning: out-dir exists, pass --force to overwrite: {args.out_dir}")
            return
        shutil.rmtree(args.out_dir)
    generate_dataset(args.record_dir, args.out_dir)


# python frame_extrap/record_to_dataset.py --record-dir Saved/NRSRecord --out-dir frame_extrap/train_data --force
if __name__ == "__main__":
    main()
