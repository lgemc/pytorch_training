import rm
import numpy as np
import os
import json




def evaluate_segmentation(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0



def check_mask_propagation(output):
    for video in output.keys():
        video_dir = f"./videos/{video}"
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        total_frames = len(frame_names)
        masks = output[video]
        if len(masks) != total_frames:
            return False
        
        first_frame_masks = np.array(masks[0])
        last_frame_masks = np.array(masks[-1])

        first_frame_has_mask = any(mask.any() for mask in first_frame_masks)
        last_frame_has_mask = any(mask.any() for mask in last_frame_masks)

        if not first_frame_has_mask or not last_frame_has_mask:
            return False
        return True


def get_ans_1(select_prompts):
    video_dir = "./videos"
    videos = [p for p in os.listdir(video_dir)if "video" in p]
    videos.sort()
    ans = []
    for video in videos:
        points, labels = select_prompts(video)
        points = np.array(points)
        labels = np.array(labels)
        ans.append(len(points) != 0 and len(labels) != 0)
    ans = np.array(ans)
    ans = ans.astype(int)*10
    ans = ans.tolist()
    return ans


def get_ans_2(f):
    ans = []
    video_test = 'video_009'
    output = f(video_test)
    check_out = check_mask_propagation(output)
    ans.append(check_out)
    ans = np.array(ans)   
    ans = ans.astype(int)*20
    ans = ans.tolist()
    return ans


def get_ans_3(track_object_videos):
    video_test = 'video_007'
    pred_mask = track_object_videos(video_test)
    pred_mask_f = pred_mask['video_007'][40:60]
    gt = json.load(open(os.path.join('videos','video_007', 'gt.json'), 'r'))
    gt = gt['gt']
    evaluation = evaluate_segmentation(np.array(gt), np.array(pred_mask_f))
    ans = [40 if evaluation > 0.6 and evaluation < 1 else 0]
    return ans



def verify_and_save(n, answer, code):
    f_answer = None
    if n == 1:
        f_answer = get_ans_1(answer)
    elif n == 2:
        f_answer = get_ans_2(answer)
    elif n == 3:
        f_answer = get_ans_3(answer)
    rm.save(f_answer, 2, 2, n, code)


def save_answers(code):
    rm.save(None, 2, 2, 4, code)

