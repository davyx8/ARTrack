import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker


def export2onnx(tracker_param, input_video, output_path):
    """Run the tracker on a video.
    args:
        tracker_param: Name of parameter file.
        input_video: Path to the input video (mp4)
        output_path: Path to the output onnx model
    """
    print(f"{tracker_param=}, {input_video=}, {output_path=}")

    # Initializing tracker
    tracker = Tracker("artrackv2_seq", tracker_param)

    # beach.mp4
    init_bbox = [790, 1440, 171, 387]

    tracker.export2onnx(input_video=input_video, init_bbox=init_bbox, output_path=output_path)
    # torch.onnx.dynamo_export(tracker)

def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a video.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('input_video', type=str, help='path to the input video.')
    parser.add_argument('output_path', type=str, help='path to the output onnx model.')

    args = parser.parse_args()

    export2onnx(args.tracker_param, args.input_video, args.output_path)


if __name__ == '__main__':
    main()
