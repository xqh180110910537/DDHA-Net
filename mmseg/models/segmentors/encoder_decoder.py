# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 interactive=False):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.interactive = interactive
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        # img[:, 4, :, :] = 0
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 交互式test

        # if self.interactive:
        #     # value_ = [0, 1, 2, 3, 200, 255]
        #     # flattened_tensor = gt_semantic_seg[0].flatten()
        #     # counts = {value: (flattened_tensor == value).sum().item() for value in value_}
        #     # print(counts)
        #     img[:, 3, :, :][img[:, 3, :, :] == 0] = 255
        #     img[:, 3, :, :][img[:, 3, :, :] == 1] = 0
        #     img[:, 3, :, :][img[:, 3, :, :] == 2] = 1
        #     img[:, 3, :, :][img[:, 3, :, :] == 3] = 2
        #     img[:, 3, :, :][img[:, 3, :, :] == 4] = 3
        #     pre_mask = img[:, 3, :, :].reshape(-1, 1, img.shape[2], img.shape[3]).to(torch.float32).to(img.device)
        #     # flattened_tensor = img[0, 3, :, :].flatten()
        #     # counts = {value: (pre_mask == value).sum().item() for value in value_}
        #     # print(counts)
        #     gt = gt_semantic_seg.clone().to(torch.float32).to(img.device)
        #     # flattened_tensor = gt.flatten()
        #     # counts = {value: (flattened_tensor == value).sum().item() for value in value_}
        #     # print(counts)
        #     diff = gt - pre_mask
        #     diff[diff != 0] = 1
        #     # prompt = self.gouzao(diff,gt).to(torch.float32)
        #     # flattened_tensor = diff.flatten()
        #     # counts = {value: (flattened_tensor == value).sum().item() for value in value_}
        #     # print(2*1280*1280-counts[0])
        #     prompt = self.gouzao(diff, gt).to(torch.float32).to(img.device)
        #     img = torch.cat([img[:, 0:3, :, :], pre_mask, prompt], dim=1)
        #     # flattened_tensor = prompt.flatten()
        #     # counts = {value: (flattened_tensor == value).sum().item() for value in value_}
        #     # print(counts)
        # x = img[:, 3, :, :].flatten()
        # unique_values, indices = torch.unique(x, return_inverse=True)
        # print(unique_values * 255)
        # if self.interactive:
        #     prompt = img[:, 4, :, :]
        #     # 定义窗口大小
        #     window_size = 64
        #
        #     # 计算每个维度上的窗口数量
        #     num_windows = prompt.shape[1] // window_size
        #
        #     # 计算总的窗口数量
        #     total_windows = num_windows * num_windows
        #
        #     # 计算需要遮挡的窗口数量（80%）
        #     precent = random.uniform(0, 0.75)
        #     # precent = 1
        #     num_windows_to_mask = int(total_windows * precent)
        #
        #     # 随机选择需要遮挡的窗口索引
        #     windows_to_mask = random.sample(range(total_windows), num_windows_to_mask)
        #
        #     # 遍历张量的每一个窗口并进行遮挡处理
        #     for window_index in windows_to_mask:
        #         row = (window_index // num_windows) * window_size
        #         col = (window_index % num_windows) * window_size
        #         # 遮挡窗口，将其值设为0
        #         prompt[:, row:row + window_size, col:col + window_size] = 0
        #     # prompt = prompt.view(prompt.shape[0], -1, prompt.shape[1], prompt.shape[2])
        #     img[:, 4, :, :] = prompt
        if self.interactive:
            img[:, 3:5, :, :] = 0
        #     print(1)
        x = self.extract_feat(img)
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        # print(loss_decode)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        # if self.interactive:
        #     img[:, 3:5, :, :] = 0

        #     img[:, 4, :, :] = 0
        #     print(1)
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        # if self.interactive:
        #     img[:, 4, :, :] = 0
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if self.out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        # if self.interactive:
        #     img[:, 4, :, :] = 0
        seg_logit = self.inference(img, img_meta, rescale)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def simple_test_logits(self, img, img_metas, rescale=True):
        """Test without augmentations.

        Return numpy seg_map logits.
        """
        seg_logit = self.inference(img[0], img_metas[0], rescale)
        seg_logit = seg_logit.cpu().numpy()
        return seg_logit

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test_logits(self, img, img_metas, rescale=True):
        """Test with augmentations.

        Return seg_map logits. Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale

        imgs = img
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit

        seg_logit /= len(imgs)
        seg_logit = seg_logit.cpu().numpy()
        return seg_logit

    def gouzao(self, input_tensor, gt, neglect_size=20):
        from scipy.ndimage import label
        input_np = input_tensor.clone().cpu().numpy()
        gt_np = gt.clone().cpu().numpy()

        # 对于 batch 中的每个图像进行处理
        for i in range(input_np.shape[0]):
            image = input_np[i, 0]  # 取出当前的图像, 形状为 (10, 10)

            # 标记连通区域
            labeled_array, num_features = label(image)

            # 创建一个新的掩码全是 200 的张量来存储 circle_tensor
            circle_tensor = np.full_like(image, 200, dtype=np.float32)
            # print(num_features)
            # 获取每个连通区域的像素点列表并生成圆形区域
            for region_num in range(1, num_features + 1):
                region = np.argwhere(labeled_array == region_num)  # 获取每个连通区域的像素点

                if len(region) < neglect_size:
                    continue  # 跳过空区域

                # # 获取连通区域的中心点坐标
                center_y = int(np.mean(region[:, 0]))
                center_x = int(np.mean(region[:, 1]))

                # # 随机选择一个非边界点作为圆心
                # valid_points = region[(region[:, 0] > 0) & (region[:, 0] < image.shape[0] - 1) &
                #                       (region[:, 1] > 0) & (region[:, 1] < image.shape[1] - 1)]
                # if len(valid_points) == 50:
                #     continue  # 如果没有非边界点，则跳过该区域
                #
                # random_index = np.random.randint(len(valid_points))
                # center_y, center_x = valid_points[random_index]

                # 获取圆心位置的 gt 值
                gt_value = gt_np[i, 0][center_y, center_x]

                # 计算到区域边界的最小距离，作为最大可能的半径
                distances = np.minimum.reduce([
                    center_y, center_x, image.shape[0] - 1 - center_y, image.shape[1] - 1 - center_x
                ])
                max_radius = int(np.min(distances))

                # 生成随机半径，确保不超出连通区域的边界
                if max_radius > 1:
                    radius = np.random.randint(1, max_radius)
                else:
                    radius = 1

                yy, xx = np.ogrid[-center_y:image.shape[0] - center_y, -center_x:image.shape[1] - center_x]
                circle = (yy ** 2 + xx ** 2 <= radius ** 2)

                # 更新 circle_tensor，圆心位置的值设置为 gt 值
                circle_tensor[circle > 0] = gt_value

            # 更新输入张量
            input_np[i, 0] = circle_tensor

        output_tensor = torch.tensor(input_np)
        return output_tensor
