import torch as t
import numpy as np
data = [[1,2],[2,0]]
t_data = t.tensor(data)
t_ones = t.ones_like(t_data)
print(f"device: {t_data.device}")
if t.cuda.is_available():
    t_data.to("cuda")
    print(f"device: {t_data.device}")
else :
    print(f"cuda is not available")
print(f"t_data: {t_data}")
t_image = t.rand(3,300,400,dtype=t.float32, requires_grad=True)
t_param = t.ones(1,dtype=t.float32,requires_grad=True)
t_image = t_image+t_param
t_image_pow2 = t_image.pow(2)
t_image = t.log(t_image.pow(2))
L = t_image.mean()
L.backward()
print(f"image:{t_image}")
t_image_normal = t.randn(3,300,400,dtype=t.float32)
if t_image_normal.max != t_image.min:
    t_image_normal_n = (t_image_normal - t_image_normal.min())/(t_image_normal.max()-t_image_normal.min())
else:
    print("無効なデータです")
