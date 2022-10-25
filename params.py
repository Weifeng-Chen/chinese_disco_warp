useCPU = False  # @param {type:"boolean"}
skip_augs = False  # @param{type: 'boolean'}
perlin_init = False  # @param{type: 'boolean'}
perlin_mode = 'mixed'  # @param ['mixed', 'color', 'gray']

use_secondary_model = False  # @param {type: 'boolean'}    # set false if you want to use the custom model
diffusion_model = "custom"  # @param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]
width_height = [512, 512]
side_x = (width_height[0]//64)*64
side_y = (width_height[1]//64)*64
if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')


diffusion_sampling_mode = 'ddim'  # @param ['plms','ddim']
use_checkpoint = True  # @param {type: 'boolean'}
ViTB32 = False  # @param{type:"boolean"}
ViTB16 = False  # @param{type:"boolean"}
ViTL14 = True  # @param{type:"boolean"}
ViTL14_336px = False  # @param{type:"boolean"}
RN101 = False  # @param{type:"boolean"}
RN50 = False  # @param{type:"boolean"}
RN50x4 = False  # @param{type:"boolean"}
RN50x16 = False  # @param{type:"boolean"}
RN50x64 = False  # @param{type:"boolean"}


# @markdown #####**OpenCLIP settings:**
ViTB32_laion2b_e16 = False  # @param{type:"boolean"}
ViTB32_laion400m_e31 = False  # @param{type:"boolean"}
ViTB32_laion400m_32 = False  # @param{type:"boolean"}
ViTB32quickgelu_laion400m_e31 = False  # @param{type:"boolean"}
ViTB32quickgelu_laion400m_e32 = False  # @param{type:"boolean"}
ViTB16_laion400m_e31 = False  # @param{type:"boolean"}
ViTB16_laion400m_e32 = False  # @param{type:"boolean"}
RN50_yffcc15m = False  # @param{type:"boolean"}
RN50_cc12m = False  # @param{type:"boolean"}
RN50_quickgelu_yfcc15m = False  # @param{type:"boolean"}
RN50_quickgelu_cc12m = False  # @param{type:"boolean"}
RN101_yfcc15m = False  # @param{type:"boolean"}
RN101_quickgelu_yfcc15m = False  # @param{type:"boolean"}

# @markdown ####**Basic Settings:**
# batch_name = 'OUTPUT-CN'  # @param{type: 'string'}
steps = 100  # @param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
tv_scale = 0  # @param{type: 'number'}
range_scale = 150  # @param{type: 'number'}
sat_scale = 0  # @param{type: 'number'}
cutn_batches = 1  # @param{type: 'number'}  # NOTE 这里会对图片做数据增强，累计计算n次CLIP的梯度，以此作为guidance。
skip_augs = False  # @param{type: 'boolean'}



# @markdown ####**Saving:**

intermediate_saves = 0  # @param{type: 'raw'}
intermediates_in_subfolder = True  # @param{type: 'boolean'}

perlin_init = False  # @param{type: 'boolean'}
perlin_mode = 'mixed'  # @param ['mixed', 'color', 'gray']
set_seed = 'random_seed'  # @param{type: 'string'}
eta = 0.8  # @param{type: 'number'}
clamp_grad = True  # @param{type: 'boolean'}
clamp_max = 0.05  # @param{type: 'number'}


# EXTRA ADVANCED SETTINGS:
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05


# @markdown ---

cut_overview = "[12]*400+[4]*600"  # @param {type: 'string'}
cut_innercut = "[4]*400+[12]*600"  # @param {type: 'string'}
cut_ic_pow = "[1]*1000"  # @param {type: 'string'}
cut_icgray_p = "[0.2]*400+[0]*600"  # @param {type: 'string'}


# @markdown ####**Transformation Settings:**
use_vertical_symmetry = False  # @param {type:"boolean"}
use_horizontal_symmetry = False  # @param {type:"boolean"}
transformation_percent = [0.09]  # @param


# @markdown ####**Animation Mode:**
animation_mode = 'None'  # @param ['None', '2D', '3D', 'Video Input'] {type:'string'}

key_frames = True  # @param {type:"boolean"}
max_frames = 1  # @param {type:"number"}


interp_spline = 'Linear'  # Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
angle = "0:(0)"  # @param {type:"string"}
zoom = "0: (1), 10: (1.05)"  # @param {type:"string"}
translation_x = "0: (0)"  # @param {type:"string"}
translation_y = "0: (0)"  # @param {type:"string"}
translation_z = "0: (10.0)"  # @param {type:"string"}
rotation_3d_x = "0: (0)"  # @param {type:"string"}
rotation_3d_y = "0: (0)"  # @param {type:"string"}
rotation_3d_z = "0: (0)"  # @param {type:"string"}
midas_depth_model = "dpt_large"  # @param {type:"string"}
midas_weight = 0.3  # @param {type:"number"}
near_plane = 200  # @param {type:"number"}
far_plane = 10000  # @param {type:"number"}
fov = 40  # @param {type:"number"}
padding_mode = 'border'  # @param {type:"string"}
sampling_mode = 'bicubic'  # @param {type:"string"}


image_prompts = {
    # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
}


display_rate = 3  # @param{type: 'number'}
n_batches = 1  # @param{type: 'number'}

# @markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False  # @param{type:"boolean"}
interp_spline = 'Linear'  # Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
resume_run = False
batch_size = 1


diff_model_map = {
    'FeiArt_Handpainted_CG_Diffusion': {'downloaded': False, 'sha': '85f95f0618f288476ffcec9f48160542ba626f655b3df963543388dcd059f86a', 'uri_list': ['https://huggingface.co/Feiart/FeiArt-Handpainted-CG-Diffusion/resolve/main/FeiArt-Handpainted-CG-Diffusion.pt']},
    '256x256_diffusion_uncond': {'downloaded': False, 'sha': 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1', 'uri_list': ['https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt']},
    '512x512_diffusion_uncond_finetune_008100': {'downloaded': False, 'sha': '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648', 'uri_list': ['https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt', 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt']},
    'portrait_generator_v001': {'downloaded': False, 'sha': 'b7e8c747af880d4480b6707006f1ace000b058dd0eac5bb13558ba3752d9b5b9', 'uri_list': ['https://huggingface.co/felipe3dartist/portrait_generator_v001/resolve/main/portrait_generator_v001_ema_0.9999_1MM.pt']},
    'pixelartdiffusion_expanded': {'downloaded': False, 'sha': 'a73b40556634034bf43b5a716b531b46fb1ab890634d854f5bcbbef56838739a', 'uri_list': ['https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt']},
    'pixel_art_diffusion_hard_256': {'downloaded': False, 'sha': 'be4a9de943ec06eef32c65a1008c60ad017723a4d35dc13169c66bb322234161', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_hard_256/resolve/main/pixel_art_diffusion_hard_256.pt']},
    'pixel_art_diffusion_soft_256': {'downloaded': False, 'sha': 'd321590e46b679bf6def1f1914b47c89e762c76f19ab3e3392c8ca07c791039c', 'uri_list': ['https://huggingface.co/KaliYuga/pixel_art_diffusion_soft_256/resolve/main/pixel_art_diffusion_soft_256.pt']},
    'pixelartdiffusion4k': {'downloaded': False, 'sha': 'a1ba4f13f6dabb72b1064f15d8ae504d98d6192ad343572cc416deda7cccac30', 'uri_list': ['https://huggingface.co/KaliYuga/pixelartdiffusion4k/resolve/main/pixelartdiffusion4k.pt']},
    'watercolordiffusion_2': {'downloaded': False, 'sha': '49c281b6092c61c49b0f1f8da93af9b94be7e0c20c71e662e2aa26fee0e4b1a9', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion_2/resolve/main/watercolordiffusion_2.pt']},
    'watercolordiffusion': {'downloaded': False, 'sha': 'a3e6522f0c8f278f90788298d66383b11ac763dd5e0d62f8252c962c23950bd6', 'uri_list': ['https://huggingface.co/KaliYuga/watercolordiffusion/resolve/main/watercolordiffusion.pt']},
    'PulpSciFiDiffusion': {'downloaded': False, 'sha': 'b79e62613b9f50b8a3173e5f61f0320c7dbb16efad42a92ec94d014f6e17337f', 'uri_list': ['https://huggingface.co/KaliYuga/PulpSciFiDiffusion/resolve/main/PulpSciFiDiffusion.pt']},
    'secondary': {'downloaded': False, 'sha': '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a', 'uri_list': ['https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth', 'https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth']},
}