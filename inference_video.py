import cv2
import torch
import argparse
import time
import os

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from config import ALL_CLASSES
from segmentation_model import faster_vit_0_any_res

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input dir')
parser.add_argument(
    '--model',
    default='outputs/model.pth',
    help='path to the model checkpoint'
)
parser.add_argument(
    '--imgsz', 
    default=[512, 416],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--device',
    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    choices=['cuda', 'cpu']
)
args = parser.parse_args()

out_dir = os.path.join('outputs', 'inference_results_video')
os.makedirs(out_dir, exist_ok=True)

# Set computation device.
device = args.device

model = faster_vit_0_any_res(pretrained=True).to(device)
model.upsample_and_classify[13] = torch.nn.Conv2d(
        512, len(ALL_CLASSES), kernel_size=(1, 1), stride=(1, 1)
)
ckpt = torch.load(args.model, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

cap = cv2.VideoCapture(args.input)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))

save_name = f"{args.input.split('/')[-1].split('.')[0]}"
# define codec and create VideoWriter object
out = cv2.VideoWriter(f"{out_dir}/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
                      args.imgsz)

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if args.imgsz is not None:
            rgb_frame = cv2.resize(rgb_frame, (args.imgsz[0], args.imgsz[1]))
        
        orig_rgb_frame = rgb_frame.copy()
        rgb_frame = rgb_frame / 255.0
        # get the start time
        start_time = time.time()
        # Do forward pass and get the output dictionary.
        outputs = get_segment_labels(rgb_frame, model, device)
        segmented_image = draw_segmentation_map(outputs)
        
        final_image = image_overlay(orig_rgb_frame, segmented_image)

        # get the end time
        end_time = time.time()
        # get the current fps
        fps = 1 / (end_time - start_time)
        # add current fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # put the FPS text on the current frame
        cv2.putText(
            final_image, 
            f"{fps:.3f} FPS", 
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2,
            lineType=cv2.LINE_AA
        )
        # press `q` to exit
        cv2.imshow('image', final_image)
        out.write(final_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")