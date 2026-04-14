import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)

    mask_rgb[mask == 1] = [255, 255, 255]


    return mask_rgb


def img_writer(inp):
    (mask, mask_path, rgb) = inp
    if rgb:
        mask_name_path = mask_path + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_path, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)

        mask_png[mask_png == 1] = 255

        mask_name_png = mask_path + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    # 加载模型
    ckpt_path = os.path.join(config.weights_path, config.test_weights_name + '.ckpt')
    print(f"Loading weights from: {ckpt_path}")
    model = Supervision_Train.load_from_checkpoint(ckpt_path, config=config)

    model.cuda(config.gpus[0])
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip()
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[90]),
            tta.Scale(scales=[0.75, 1.0, 1.25], interpolation='bicubic', align_corners=False)
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        results = []
        print(f"Start inference on {len(test_dataset)} images...")

        for input in tqdm(test_loader):
            raw_predictions = model(input['img'].cuda(config.gpus[0]))

            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                gt = masks_true[i].cpu().numpy()

                evaluator.add_batch(pre_image=mask, gt_image=gt)


                save_path = str(args.output_path / image_ids[i])
                results.append((mask, save_path, args.rgb))

    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()

    precision_per_class = evaluator.Precision()
    recall_per_class = evaluator.Recall()

    OA = evaluator.OA()

    print("\n" + "=" * 70)

    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10}")
    print("-" * 70)


    for class_name, class_iou, class_f1, class_pre, class_rec in zip(
            config.classes,
            iou_per_class,
            f1_per_class,
            precision_per_class,
            recall_per_class
    ):
        print(f'{class_name:<12} {class_pre:.4f}     {class_rec:.4f}     {class_f1:.4f}     {class_iou:.4f}')

    print("-" * 70)

    mIoU = np.nanmean(iou_per_class)
    mF1 = np.nanmean(f1_per_class)
    mPre = np.nanmean(precision_per_class)
    mRec = np.nanmean(recall_per_class)

    print(f"Mean Precision : {mPre:.4f}")
    print(f"Mean Recall    : {mRec:.4f}")
    print(f"Mean F1        : {mF1:.4f}")
    print(f"Mean IoU       : {mIoU:.4f}")
    print(f"OA             : {OA:.4f}")

    if len(iou_per_class) > 1:
        print("-" * 70)
        print(">> Building Class Metrics (For Paper) <<")
        print(f"Precision : {precision_per_class[1]:.4f}")
        print(f"Recall    : {recall_per_class[1]:.4f}")
        print(f"F1-Score  : {f1_per_class[1]:.4f}")
        print(f"IoU       : {iou_per_class[1]:.4f}")

    print("=" * 70 + "\n")

    t0 = time.time()
    with mpp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('Images writing spent: {:.2f} s'.format(img_write_time))


if __name__ == "__main__":
    main()