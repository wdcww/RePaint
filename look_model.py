### cd到runs/unet_model
# # 然后 tensorboard --logdir=./

import os
import conf_mgt
from guided_diffusion import dist_util
from guided_diffusion.script_util import create_model_and_diffusion, select_args, model_and_diffusion_defaults
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import yamlread



conf = conf_mgt.conf_base.Default_Conf()
conf.update(yamlread('confs/face_example1.yml'))
device = dist_util.dev(None)

model, _ = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )


if __name__ == "__main__":
    # 创建一个TensorBoard的SummaryWriter
    # writer = SummaryWriter(log_dir="runs/unet_model")
    #
    # # 创建一个随机的输入张量，大小与模型的输入一致
    # # 这里假设输入大小为(batch_size, in_channels, image_size, image_size)，根据需要调整
    # input_tensor = torch.randn(1, 3, 256, 256)  # 假设输入大小为256x256的RGB图像
    # # timesteps = torch.randint(0, 1000, (1,))
    # timesteps = torch.tensor([1000])
    # writer.add_graph(model, (input_tensor, timesteps))
    #
    # # 关闭TensorBoard
    # writer.close()
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")  # checkpoint地址conf.model_path
    )



# D:\myenv\python.exe D:\py_github\RePaint\look_model.py
# D:\py_github\RePaint\guided_diffusion\unet.py:666: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   if timesteps[0].item() > self.conf.diffusion_steps:
# D:\py_github\RePaint\guided_diffusion\unet.py:155: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   assert x.shape[1] == self.channels
# D:\py_github\RePaint\guided_diffusion\unet.py:360: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   assert width % (2 * self.n_heads) == 0
# D:\py_github\RePaint\guided_diffusion\unet.py:364: TracerWarning: Converting a tensor to a Python float might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   scale = 1 / math.sqrt(math.sqrt(ch))
# D:\py_github\RePaint\guided_diffusion\unet.py:117: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#   assert x.shape[1] == self.channels
#
# Process finished with exit code 0