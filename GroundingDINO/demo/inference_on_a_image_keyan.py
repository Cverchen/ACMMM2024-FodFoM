import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
sys.path.append("/home/2022/jiankang/Storage/Mycode/GroundingDINO")
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(flag, image_pil, tgt, per_image_path):
    # image_pil_orgin = Image.open(per_image_path).convert("RGB")
    image_pil_origin = cv2.imread(per_image_path)
    image_pil_temp = image_pil_origin.copy()
    image_area = Image.open(per_image_path).convert("RGB").width * Image.open(per_image_path).convert("RGB").height
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"
    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    # y0_min = 10000
    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box # 检测框的左上角和右下角坐标  坐标原点在左上角
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)   
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)    
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")
        area = (y1-y0) * (x1-x0)
        # if area/image_area <= 0.4 and area !=0 :
        if area/image_area <= 1:
            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
            face_image = image_pil_temp[y0:y1, x0:x1]
            if face_image.size == 0:
                flag = False
                continue
            face_image = cv2.blur(face_image, (50,50))
            image_pil_temp[y0:y1, x0:x1] = face_image
        else:
            flag = False

    return image_pil, mask, image_pil_temp, flag

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    # config_file = 'groundingdino/config/GroundingDINO_SwinT_OGC.py' 
    checkpoint_path = args.checkpoint_path  # change the path of the model
    # checkpoint_path = 'weights/groundingdino_swint_ogc.pth'
    image_path = args.image_path
    # image_path = 'aircondition/879889-0.jpeg'
    text_prompt = args.text_prompt
    # text_prompt = 'air condition'
    output_dir = args.output_dir
    # output_dir = 'output'
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    class_index = []  
    class_name = []
    with open('classindex.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            class_index.append(line)
    with open('classname.txt', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            class_name.append(line)

    for i in range(100):
        classindex = class_index[i]
        classname = class_name[i]
        print(classindex)
        image_path = []
        folder = 'Datasets/ImageNet100/train/{}'.format(classindex)
        for name in os.listdir('Datasets/ImageNet100/train/{}'.format(classindex)):
            image_path.append(os.path.join(folder, name))
        output_dir = 'ImageNet_background/output_{}'.format(classindex)
        text_prompt = classname
        index = 0
        for per_image_path in tqdm(image_path):
            # make dir[]
            os.makedirs(output_dir, exist_ok=True)
            # load image
            image_pil, image = load_image(per_image_path)
            # load model
            model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
            
            # visualize raw image
            # if not os.path.exists(os.path.join(output_dir,(per_image_path.split('/')[1]).split('.')[0])):
            #     os.makedirs(os.path.join(output_dir,(per_image_path.split('/')[1]).split('.')[0]))
            # image_pil.save(os.path.join(output_dir,(per_image_path.split('/')[1]).split('.')[0],"raw_image.jpg"))

            # set the text_threshold to None if token_spans is set.
            if token_spans is not None:
                text_threshold = None
                print("Using token_spans. Set the text_threshold to None.")


            # run model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=None
            )

            # visualize pred
            size = image_pil.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],  # H,W
                "labels": pred_phrases,
            }
            # import ipdb; ipdb.set_trace()
            flag = True
            image_with_box,_,image_clip,flag = plot_boxes_to_image(flag, image_pil, pred_dict, per_image_path)
            # pred_dict 预测的东西
            if flag:
                # image_with_box.save(os.path.join(output_dir,(per_image_path.split('/')[1]).split('.')[0],"pred.jpg"))
                cv2.imwrite(os.path.join(output_dir,"clip_{}.jpg".format(index)), image_clip)
            index += 1
            # if image_clip != None:
            #     image_clip.save(os.path.join(output_dir,(per_image_path.split('/')[1]).split('.')[0],"clip.jpg"))

    # # make dir
    # os.makedirs(output_dir, exist_ok=True)
    # # load image
    # image_pil, image = load_image(image_path)
    # # load model
    # model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # # visualize raw image
    # image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # # set the text_threshold to None if token_spans is set.
    # if token_spans is not None:
    #     text_threshold = None
    #     print("Using token_spans. Set the text_threshold to None.")


    # # run model
    # boxes_filt, pred_phrases = get_grounding_output(
    #     model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=None
    # )

    # # visualize pred
    # size = image_pil.size
    # pred_dict = {
    #     "boxes": boxes_filt,
    #     "size": [size[1], size[0]],  # H,W
    #     "labels": pred_phrases,
    # }
    # # import ipdb; ipdb.set_trace()
    # image_with_box,_,image_clip= plot_boxes_to_image(image_pil, pred_dict, per_image_path=image_path)
    # image_with_box.save(os.path.join(output_dir, "pred.jpg"))
    # if image_clip != None:
    #     image_clip.save(os.path.join(output_dir,"clip.jpg"))