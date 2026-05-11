"""
Utility functions for interpretation generation.
© copyright 2024 Bytedance Ltd. and/or its affiliates.
Modified from Tyler Lawson, Saeed khorram. https://github.com/saeed-khorram/IGOS
"""

import os
import sys
from io import BytesIO
from textwrap import fill

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# mean and standard deviation for the imagenet dataset
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

special_ids = [
    12,
    13,
    29871,
    29889,
    29892,
    1919,
    869,
    29899,
    29897,
    29898,
    278,
    322,
    310,
    263,
    385,
    445,
    393,
    338,
    411,
]
CHOICES = ["A", "B", "C", "D", "E", "F"]


def eprint(*args, **kwargs):
    """
        Prints to the std.err

    :param args:
    :param kwargs:
    :return:
    """
    print(*args, file=sys.stderr, **kwargs)


def get_data(args, row):
    if args.dataset == "MMVP":
        photo_id = row["lndex"]
        image_path = os.path.join(args.image_folder, f"{photo_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        if args.choices:
            opts = row["Options"].split(" ")
            cur_prompt = f"{row['Question']}\nA. {opts[1]}\nB. {opts[3]}\n"
            qs = (
                cur_prompt
                + "Answer only with the option's letter A or B from the given choices directly."
            )
        else:
            qs = row["Question"]
            cur_prompt = qs
    elif args.dataset == "cvbench":
        image_path = os.path.join(args.image_folder, row["filename"])
        image = Image.open(image_path).convert("RGB")
        if args.choices:
            choices = row["choices"]
            choices_text = ""
            for j, item in enumerate(choices):
                choices_text += f"\n{CHOICES[j]}. {item}"

            cur_prompt = row["question"] + choices_text
            qs = (
                cur_prompt
                + "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            qs = row["question"]
            cur_prompt = qs
    elif args.dataset == "mmstar":
        image_stream = BytesIO(row["image"]["bytes"])
        image = Image.open(image_stream).convert("RGB")
        if args.choices:
            cur_prompt = row["question"]
            qs = (
                cur_prompt
                + "\n"
                + "Answer with the option's letter from the given choices directly."
            )
        else:
            qs = row["question"].split("?")[0] + "?"
            cur_prompt = qs

    elif args.dataset == "llava-bench":
        image_path = os.path.join(args.image_folder, row["image"])
        image = Image.open(image_path).convert("RGB")
        cur_prompt = row["text"]
        qs = cur_prompt
    return image, qs, cur_prompt


def get_kernel_size(image_size):
    if image_size[0] >= 2000:
        kernel_size = 501
    elif image_size[0] >= 800:
        kernel_size = 301
    elif image_size[0] >= 500:
        kernel_size = 201
    else:
        kernel_size = 101
    return kernel_size


def save_heatmaps(
    masks,
    images,
    size,
    index,
    index_o,
    outdir,
    model_name,
    box,
    classes,
    labels,
    out=224,
):
    if isinstance(images, list):
        images = images[1]
    elif images.shape[0] != 1:
        images = images[0].unsqueeze(0)
    masks = masks.view(-1, 1, size, size)
    up = torch.nn.UpsamplingBilinear2d(size=images.shape[-2:]).cuda()

    u_mask = up(masks)
    u_mask = u_mask.permute((0, 2, 3, 1))

    # Normalize the mask
    u_mask = (u_mask - torch.min(u_mask)) / (torch.max(u_mask) - torch.min(u_mask))
    u_mask = u_mask.cpu().detach().numpy()

    # deprocess images
    images = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    images = images.numpy()

    for i, (image, u_msk) in enumerate(zip(images, u_mask, strict=False)):
        # get the color map and normalize to 0-1
        heatmap = cv2.applyColorMap(np.uint8(255 * u_msk), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap / 255)
        # overlay the mask over the image
        # overlay = (u_msk ** 0.8) *0.5* image + (1 - u_msk ** 0.8) * heatmap
        overlay = 0.5 * heatmap + 0.5 * image
        cv2.normalize(overlay.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
        overlay[overlay < 0] = 0

        plt.imsave(os.path.join(outdir, f"{index + i}_{index_o}_heatmap.jpg"), heatmap)
        plt.imsave(os.path.join(outdir, f"{index + i}_{index_o}_overlay.jpg"), overlay)


def save_masks(masks, index, categories, mask_name, outdir):
    """
        Saves the generated masks as numpy.ndarrays.

    :param masks:
    :param index:
    :param categories:
    :param mask_name:
    :param outdir:
    :return:
    """
    masks = masks.cpu().detach().numpy()
    for i, (mask, category) in enumerate(zip(masks, categories, strict=False), start=index):
        np.save(os.path.join(outdir, f"{mask_name}_{i + 1}_mask_{category}.npy"), mask)


def save_loss(
    loss_del,
    loss_ins,
    loss_l1,
    loss_tv,
    loss_l2,
    index,
    index_o,
    outdir,
    loss_comb_del=None,
    loss_comb_ins=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    iterations = np.arange(len(loss_del))

    if loss_comb_del:
        ax1.plot(iterations, loss_comb_del, label="loss comb del", marker="^")
    if loss_comb_ins:
        ax1.plot(iterations, loss_comb_ins, label="loss comb ins", marker="<")
    ax1.plot(iterations, loss_del, label="loss del", marker="o")
    ax1.plot(iterations, loss_ins, label="loss ins", marker="s")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss Value")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(iterations, loss_tv, label="loss TV", marker="o")
    ax2.plot(iterations, loss_l2, label="loss L2", marker="+")
    ax2.plot(iterations, loss_l1, label="loss L1", marker="x")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss Value")
    ax2.grid(True)
    ax2.legend()

    plt.savefig(
        os.path.join(outdir, f"{index}_{index_o}_losses.jpg"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def save_curves(del_curve, ins_curve, index_curve, index, index_o, outdir):
    """
        Save the deletion/insertion curves for the generated masks.

    :param del_curve:
    :param ins_curve:
    :param index_curve:
    :param index:
    :param index_o:
    :param outdir:
    :return:
    """
    for i in range(len(del_curve)):
        fig, (ax, ax1) = plt.subplots(2, 1)
        ax.plot(index_curve, del_curve[i], color="r", label="deletion")
        ax.fill_between(index_curve, del_curve[i], facecolor="maroon", alpha=0.4)
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(labelsize=14)
        ax.set_yticks(np.arange(0, 1.01, 1))
        ax.legend(["Deletion"], fontsize="x-large")
        ax.text(
            0.5,
            0.5,
            f"AUC: {auc(del_curve[i]):.4f}",
            fontsize=14,
            horizontalalignment="center",
            verticalalignment="center",
        )

        ax1.plot(index_curve, ins_curve[i], color="b", label="Insertion")
        ax1.fill_between(index_curve, ins_curve[i], facecolor="darkblue", alpha=0.4)
        ax1.set_ylim([-0.05, 1.05])
        ax1.tick_params(labelsize=14)
        ax1.set_yticks(np.arange(0, 1.01, 1))
        ax1.legend(["Insertion"], fontsize="x-large")
        ax1.text(
            0.5,
            0.5,
            f"AUC: {auc(ins_curve[i]):.4f}",
            fontsize=14,
            horizontalalignment="center",
            verticalalignment="center",
        )

        # save the plot
        plt.savefig(
            os.path.join(outdir, f"{index}_{index_o}_curves.jpg"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


def save_images(images, index, index_o, outdir, classes, labels, pred_data, text=None):
    """
        saves original images into output directory

    :param images:
    :param index:
    :param index_o:
    :param outdir:
    :param classes:
    :param labels:
    :return:
    """
    if isinstance(images, list):
        images = images[1]
    elif images.shape[0] != 1:
        images = images[0].unsqueeze(0)
    images_ = images.cpu().detach().permute((0, 2, 3, 1)) * std + mean
    for i, image in enumerate(images_):
        wrapped_text = fill(text, width=60)
        fig, ax = plt.subplots(figsize=(5, 5))

        plt.subplots_adjust(top=0.8)
        ax.imshow(image.numpy())
        ax.axis("off")
        fig.text(0.5, 0.9, wrapped_text, ha="center", va="top", wrap=True, fontsize=10)
        plt.savefig(
            os.path.join(outdir, f"{index + i}_{index_o}_image.jpg"),
            bbox_inches="tight",
            pad_inches=0.2,
        )


def auc(array):
    """
        calculates area under the curve (AUC)

    :param array:
    :return:
    """
    return (sum(array) - array[0] / 2 - array[-1] / 2) / len(array)


def get_initial(pred_data, k, init_posi, init_val, input_size, out_size):
    """
        filter the detection results by the threshold (predicted score)

    :param pred_data:
    :param k:
    :param initial_posi:
    :param init_val:
    :param input_size:
    :param out_size:
    :return:
    """
    interval_r = (pred_data["boxes"][:, 2] - pred_data["boxes"][:, 0]) / k
    interval_c = (pred_data["boxes"][:, 3] - pred_data["boxes"][:, 1]) / k
    num_row = init_posi // k
    num_col = init_posi - num_row * k
    init_boxes = np.concatenate(
        [
            [pred_data["boxes"][:, 0] + interval_r * num_row],  # x1
            [pred_data["boxes"][:, 1] + interval_c * num_col],  # y1
            [pred_data["boxes"][:, 0] + interval_r * (num_row + 1)],  # x2
            [pred_data["boxes"][:, 1] + interval_c * (num_col + 1)],  # y2
        ],
        axis=0,
    ).T

    pred_data["init_masks"] = []
    down = torch.nn.UpsamplingBilinear2d(size=(out_size, out_size))

    for ith, box in enumerate(init_boxes):
        init_mask = torch.zeros((input_size[0], input_size[1])).unsqueeze(0)
        init_mask[int(box[0]) : int(box[2]), int(box[1]) : int(box[3])] = 1

        if "masks" in pred_data:
            init_mask = init_mask * pred_data["masks"][ith]

        init_mask = down(init_mask.unsqueeze(0)) * init_val
        pred_data["init_masks"].append(1 - init_mask)
    return pred_data


def generate(args, model, input_ids, image, image_size):
    output_ids = model.generate(
        input_ids,
        images=image,
        # image_sizes=image_size,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    output_ids = output_ids[:, 1:-1]
    return output_ids


def match_keywords(lst, sublist):
    lst_str = ",".join(map(str, lst))
    sublist_str = ",".join(map(str, sublist))
    start_idx = lst_str.find(sublist_str)

    if start_idx == -1:
        return None

    start = lst_str[:start_idx].count(",")
    end = start + len(sublist) - 1
    return [start, end]
