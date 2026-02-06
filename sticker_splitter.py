# -*- coding: utf-8 -*-
"""
表情包自动分割工具 (Sticker Auto-Splitter)
使用连通域分析自动识别并裁切每个独立贴纸
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

def remove_green_screen(img_bgr, threshold=45):
    """
    移除绿幕背景，返回带透明通道的 BGRA 图像
    使用更严格的纯绿色检测，避免误抠角色
    """
    b, g, r = cv2.split(img_bgr)
    
    # 更严格的绿幕检测：
    # 1. 绿色通道必须很高 (>150)
    # 2. 绿色必须显著高于红色和蓝色
    # 3. 红色和蓝色都必须较低
    mask = (
        (g > 150) &                    # 绿色通道必须很亮
        (g > r + threshold) &          # 绿色比红色高出阈值
        (g > b + threshold) &          # 绿色比蓝色高出阈值
        (r < 150) &                    # 红色不能太高
        (b < 150)                      # 蓝色不能太高
    )
    
    # 创建 alpha 通道
    alpha = np.ones(g.shape, dtype=np.uint8) * 255
    alpha[mask] = 0
    
    # 合并为 BGRA
    bgra = cv2.merge([b, g, r, alpha])
    return bgra, alpha


def find_stickers(alpha_mask, min_area=500):
    """
    使用连通域分析找到所有独立贴纸区域
    返回每个贴纸的边界框列表 [(x, y, w, h), ...]
    """
    # 二值化
    _, binary = cv2.threshold(alpha_mask, 127, 255, cv2.THRESH_BINARY)
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    bboxes = []
    for i in range(1, num_labels):  # 跳过背景（标签 0）
        x, y, w, h, area = stats[i]
        if area >= min_area:  # 过滤掉太小的噪点
            bboxes.append((x, y, w, h))
    
    # 按从上到下、从左到右排序
    bboxes.sort(key=lambda b: (b[1] // 100, b[0]))  # 按行分组后按 x 排序
    
    return bboxes

def crop_and_save(img_bgra, bboxes, output_dir, max_dim=None, padding=5):
    """
    裁切每个贴纸并保存为透明动图 GIF
    通过添加几乎不可见的微动效，让微信识别为真正的动图
    """
    from PIL import Image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (x, y, w, h) in enumerate(bboxes):
        # 添加 padding，防止边缘被切掉
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_bgra.shape[1], x + w + padding)
        y2 = min(img_bgra.shape[0], y + h + padding)
        
        cropped = img_bgra[y1:y2, x1:x2]
        
        # 等比例缩放
        if max_dim and max(cropped.shape[:2]) > max_dim:
            scale = max_dim / max(cropped.shape[:2])
            new_w = int(cropped.shape[1] * scale)
            new_h = int(cropped.shape[0] * scale)
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为 PIL Image (BGRA -> RGBA)
        rgba = cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA)
        pil_img = Image.fromarray(rgba)
        
        # 创建完全相同的两帧（肉眼完全不可见的"动效"）
        # 这样 GIF 是真正的动图格式，但视觉上完全静止
        frame1 = pil_img.copy()
        frame2 = pil_img.copy()
        
        # 保存为动图 GIF
        filename = output_dir / f"sticker_{idx + 1:02d}.gif"
        frame1.save(
            str(filename),
            save_all=True,
            append_images=[frame2],
            duration=100,  # 每帧 100ms
            loop=0,        # 无限循环
            transparency=0,
            disposal=2     # 清除前一帧
        )

        print(f"保存: {filename} ({cropped.shape[1]}x{cropped.shape[0]})")
    
    print(f"\n✅ 共提取 {len(bboxes)} 个动图贴纸到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="表情包自动分割工具")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", default="output_stickers", help="输出目录 (默认: output_stickers)")
    parser.add_argument("-t", "--threshold", type=int, default=45, help="绿幕阈值 (默认: 45)")
    parser.add_argument("-m", "--max-dim", type=int, default=None, help="最长边缩放尺寸 (如 160 用于微信)")
    parser.add_argument("--min-area", type=int, default=500, help="最小贴纸面积 (默认: 500)")
    args = parser.parse_args()
    
    print(f"📂 读取图片: {args.input}")
    img = cv2.imread(args.input)
    if img is None:
        print("❌ 无法读取图片!")
        return
    
    print("🎨 移除绿幕背景...")
    img_bgra, alpha = remove_green_screen(img, args.threshold)
    
    print("🔍 检测独立贴纸区域...")
    bboxes = find_stickers(alpha, args.min_area)
    print(f"   找到 {len(bboxes)} 个贴纸")
    
    print("✂️ 裁切并生成微动效 GIF...")
    crop_and_save(img_bgra, bboxes, args.output, args.max_dim)

if __name__ == "__main__":
    main()
