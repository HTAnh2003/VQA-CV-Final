<!-- ---
license: apache-2.0
language:
- vi
base_model:
- OpenGVLab/InternVL3-1B
- 5CD-AI/Vintern-1B-v3_5
pipeline_tag: visual-question-answering
--- -->
# VQA-CV-Final

Dự án cuối kỳ môn Thị giác Máy tính (Computer Vision), tập trung vào Visual Question Answering (VQA). Mục tiêu là xây dựng một hệ thống trả lời câu hỏi dựa trên đầu vào là hình ảnh và văn bản.

## Dataset : 
- https://www.kaggle.com/datasets/tienanh2003/vitextvqa
- https://huggingface.co/datasets/minhquan6203/ViTextVQA

    Paper của dataset : https://arxiv.org/abs/2404.10652


## Training 
Chi tiết quá trình finetune ở notebook [VQA-CV-Final-Training.ipynb](VQA-CV-Final-Training.ipynb).


## Inference 

Checkpoint đã fine tune được lưu tại [TienAnh/Finetune_VQA_1B](https://huggingface.co/TienAnh/Finetune_VQA_1B).


```python
import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

model = AutoModel.from_pretrained(
    "TienAnh/Finetune_VQA_1B",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=False,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained("TienAnh/Finetune_VQA_1B", trust_remote_code=True, use_fast=False)

test_image = 'test-image.jpg'

pixel_values = load_image(test_image, max_num=6).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens= 1024, do_sample=False, num_beams = 3, repetition_penalty=2.5)

question = '<image>\nvăn bản trong hình ảnh là gì'

response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

#question = "Câu hỏi khác ......"
#response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
#print(f'User: {question}\nAssistant: {response}')
```
## Evaluation
Link submit đánh giá :
- [EM](https://www.kaggle.com/competitions/ViTextVQA-evaluation)
- [F1](https://www.kaggle.com/competitions/ViTextVQA-F1-evaluation)

## Reference
- [InternVL3](https://huggingface.co/OpenGVLab/InternVL3-1B)
- [Vintern](https://huggingface.co/5CD-AI/Vintern-1B-v3_5)
- [Dataset ViTextVQA](https://arxiv.org/abs/2404.10652)

