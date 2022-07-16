#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from azureml.core.run import Run
from loguru import logger
import logging

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):

    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]

    print('image path: ' + path)

    files.sort()
    for image_name in files:

        print('image: ' + image_name)

        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            #save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            save_folder = vis_folder
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        #ch = cv2.waitKey(0)
        #if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #    break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def infer(conf, code_prefix = './'):
    """
    Main inference method.
    """

    logger.info(f'Starting inference with {conf.get("model_name")} ..')
    logger.info(f'Input dir {conf.get("indir")}')

    output_dir = os.path.join(conf.get('outdir'), conf.get("name") + '__' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    logger.info(f'Output dir {output_dir}')

    run = Run.get_context()  # Azure specific, can also give local instance
    run.tag('model_name', conf.get("model_name"))

    # -- Inititialize experiment --
    exp = get_exp(os.path.join(code_prefix, 'main_experiment.py'), None)
    exp.setup(conf)

    # TODO: Does this has any effect? Because the image input size
    #        is controlled by args.model_name and set inside exp.setup()
    exp.test_size = (640,640)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    model.cuda()
    model.eval()

    logger.info("loading checkpoint")
    ckpt = torch.load(conf.get("ckpt"), map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

#    logger.info("\tFusing model...")
#    model = fuse_model(model)

    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, ("bed"), trt_file, decoder, 'gpu')
    image_demo(predictor, output_dir, conf.get('indir'), time.localtime(), True)

    # imageflow_demo(predictor, vis_folder, current_time, args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help="input data directory")
    parser.add_argument('--outdir', type=str, help="output data directory")
    parser.add_argument('--ckpt', type=str, default=None, help="path to model weight file (for pretrained weights)")
    parser.add_argument('--config', type=str, default='run_config.yml', help="path to config file of the experiment")
    args = parser.parse_args()

    from experiment_config import ExperimentConfig
    conf = ExperimentConfig(args.config)
    conf.add(args)

    infer(conf)
