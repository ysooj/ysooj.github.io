from diffusers import StableDiffusionPipeline
import torch

# Stable Diffusion 파이프라인 로드
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # CPU 사용
# 나는 GPU가 아니라 CPU라서 해당 부분을 "cuda"가 아니라 "CPU"라고 적었다.

# 텍스트 설명을 기반으로 이미지 생성
prompt = "A futuristic cityscape with flying cars at sunset"
image = pipe(prompt).images[0]

# 생성된 이미지 저장 및 출력
image.save("generated_image.png")
image.show()
