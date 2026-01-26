# python lpips_2dirs.py -d0 ./250_10_10_niter1/gt -d1 ./250_10_10_niter1/inpainted -o ./250_10_10_niter1/lpips.txt --use_gpu

import os
import argparse
import lpips  # ensure lpips is installed
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0', '--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1', '--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o', '--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

# 初始化 LPIPS 模型
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
if opt.use_gpu:
    loss_fn.cuda()

# 获取两个目录中的图片文件
def get_sorted_image_files(folder):
    exts = ['.png', '.jpg', '.jpeg', '.bmp']
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ])

files0 = get_sorted_image_files(opt.dir0)
files1 = get_sorted_image_files(opt.dir1)

# 确保数量一致
assert len(files0) == len(files1), f"文件数量不一致：{len(files0)} vs {len(files1)}"

# 开始比较
dist_list = []

with open(opt.out, 'w') as f:
    for idx, (f0, f1) in enumerate(zip(files0, files1)):
        img0 = lpips.im2tensor(lpips.load_image(f0))  # [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(f1))

        if opt.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()

        dist01 = loss_fn.forward(img0, img1)
        dist_value = dist01.item()
        dist_list.append(dist_value)

        print(f'{idx:05d}.png: {dist_value:.6f}')
        f.write(f'{idx:05d}.png {dist_value:.6f}\n')

    # 计算平均值
    mean_dist = sum(dist_list) / len(dist_list)
    print(f'Average LPIPS: {mean_dist:.6f}')
    f.write(f'Average: {mean_dist:.6f}\n')




# # # # # 下面这部分必须保证 d0/ 与 d1/ 中的图片名字是一样的,例如d0/1.png 与 d1/1.png
# import argparse
# import os
# import lpips
#
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
# parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
# parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
# parser.add_argument('-v','--version', type=str, default='0.1')
# parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
#
# opt = parser.parse_args()
#
# ## Initializing the model
# loss_fn = lpips.LPIPS(net='alex',version=opt.version)
# if(opt.use_gpu):
# 	loss_fn.cuda()
#
# # crawl directories
# f = open(opt.out,'w')
# files = os.listdir(opt.dir0)
#
# for file in files:
# 	if(os.path.exists(os.path.join(opt.dir1,file))):
# 		# Load images
# 		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
# 		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))
#
# 		if(opt.use_gpu):
# 			img0 = img0.cuda()
# 			img1 = img1.cuda()
#
# 		# Compute distance
# 		dist01 = loss_fn.forward(img0,img1)
# 		print('%s: %.3f'%(file,dist01))
# 		f.writelines('%s: %.6f\n'%(file,dist01))
#
# f.close()
