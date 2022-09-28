useCPU = False  # @param {type:"boolean"}
skip_augs = False  # @param{type: 'boolean'}
perlin_init = False  # @param{type: 'boolean'}
perlin_mode = 'mixed'  # @param ['mixed', 'color', 'gray']

# @markdown ####**Models Settings (note: For pixel art, the best is pixelartdiffusion_expanded):**
# diffusion_model = "512x512_diffusion_uncond_finetune_008100" #@param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]
use_secondary_model = False  # @param {type: 'boolean'}    # set false if you want to use the custom model

# diffusion_model = "512x512_diffusion_uncond_finetune_008100" #@param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]
diffusion_model = "custom"  # @param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]
# diffusion_model = "256x256_diffusion_uncond"  # @param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100", "portrait_generator_v001", "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256", "pixelartdiffusion4k", "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion", "custom"]
width_height_for_256x256_models = [256, 256]  # @param{type: 'raw'}
kaliyuga_pixel_art_model_names = ['pixelartdiffusion_expanded', 'pixel_art_diffusion_hard_256', 'pixel_art_diffusion_soft_256', 'pixelartdiffusion4k', 'PulpSciFiDiffusion']
kaliyuga_watercolor_model_names = ['watercolordiffusion', 'watercolordiffusion_2']
kaliyuga_pulpscifi_model_names = ['PulpSciFiDiffusion']
diffusion_models_256x256_list = ['256x256_diffusion_uncond'] + kaliyuga_pixel_art_model_names + kaliyuga_watercolor_model_names + kaliyuga_pulpscifi_model_names
width_height_for_512x512_models = [512, 512]  # @param{type: 'raw'}
width_height = width_height_for_256x256_models if diffusion_model in diffusion_models_256x256_list else width_height_for_512x512_models
# Get corrected sizes
side_x = (width_height[0]//64)*64
side_y = (width_height[1]//64)*64
if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.')


diffusion_sampling_mode = 'ddim'  # @param ['plms','ddim']
# @markdown #####**Custom model:**
# custom_path = '/content/drive/MyDrive/deep_learning/ddpm/ema_0.9999_058000.pt'#@param {type: 'string'}
# custom_path = '/home/chenweifeng/scripts_t2i/disco_new_warp/models/nature_ema_160000.pt'  # @param {type: 'string'}

# @markdown #####**CLIP settings:**
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
batch_name = 'TimeToDisco'  # @param{type: 'string'}
steps = 100  # @param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
# width_height_for_512x512_models = [1280, 768]  # @param{type: 'raw'}
# clip_guidance_scale = 7000  # @param{type: 'number'}
tv_scale = 0  # @param{type: 'number'}
range_scale = 150  # @param{type: 'number'}
sat_scale = 0  # @param{type: 'number'}
cutn_batches = 1  # @param{type: 'number'}  # NOTE 这里会对图片做数据增强，累计计算n次CLIP的梯度，以此作为guidance。
skip_augs = False  # @param{type: 'boolean'}

# @markdown ####**Image dimensions to be used for 256x256 models (e.g. pixelart models):**
width_height_for_256x256_models = [512, 448]  # @param{type: 'raw'}

# @markdown ####**Video Init Basic Settings:**
video_init_steps = 100  # @param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
video_init_clip_guidance_scale = 1000  # @param{type: 'number'}
video_init_tv_scale = 0.1  # @param{type: 'number'}
video_init_range_scale = 150  # @param{type: 'number'}
video_init_sat_scale = 300  # @param{type: 'number'}
video_init_cutn_batches = 4  # @param{type: 'number'}
video_init_skip_steps = 50  # @param{type: 'integer'}

# @markdown ---

# @markdown ####**Init Image Settings:**
# init_image = None  # @param{type: 'string'}
# init_scale = 1000  # @param{type: 'integer'}
# skip_steps = 10  # @param{type: 'integer'}
# @markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.*


# @markdown ####**Saving:**

intermediate_saves = 0  # @param{type: 'raw'}
intermediates_in_subfolder = True  # @param{type: 'boolean'}
# @markdown Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps

# @markdown A value of `2` will save a copy at 33% and 66%. 0 will save none.

# @markdown A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. (Make sure to include the brackets)


# @markdown ####**Advanced Settings:**
# @markdown *There are a few extra advanced settings available if you double click this cell.*

# @markdown *Perlin init will replace your init, so uncheck if using one.*

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

# @markdown ####**Cutn Scheduling:**
# @markdown Format: `[40]*400+[20]*600` = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000

# @markdown cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.

cut_overview = "[12]*400+[4]*600"  # @param {type: 'string'}
cut_innercut = "[4]*400+[12]*600"  # @param {type: 'string'}
cut_ic_pow = "[1]*1000"  # @param {type: 'string'}
cut_icgray_p = "[0.2]*400+[0]*600"  # @param {type: 'string'}

# @markdown KaliYuga model settings. Refer to [cut_ic_pow](https://ezcharts.miraheze.org/wiki/Category:Cut_ic_pow) as a guide. Values between 1 and 100 all work.
pad_or_pulp_cut_overview = "[15]*100+[15]*100+[12]*100+[12]*100+[6]*100+[4]*100+[2]*200+[0]*200"  # @param {type: 'string'}
pad_or_pulp_cut_innercut = "[1]*100+[1]*100+[4]*100+[4]*100+[8]*100+[8]*100+[10]*200+[10]*200"  # @param {type: 'string'}
pad_or_pulp_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300"  # @param {type: 'string'}
pad_or_pulp_cut_icgray_p = "[0.87]*100+[0.78]*50+[0.73]*50+[0.64]*60+[0.56]*40+[0.50]*50+[0.33]*100+[0.19]*150+[0]*400"  # @param {type: 'string'}

watercolor_cut_overview = "[14]*200+[12]*200+[4]*400+[0]*200"  # @param {type: 'string'}
watercolor_cut_innercut = "[2]*200+[4]*200+[12]*400+[12]*200"  # @param {type: 'string'}
watercolor_cut_ic_pow = "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300"  # @param {type: 'string'}
watercolor_cut_icgray_p = "[0.7]*100+[0.6]*100+[0.45]*100+[0.3]*100+[0]*600"  # @param {type: 'string'}


# @markdown ####**Transformation Settings:**
use_vertical_symmetry = False  # @param {type:"boolean"}
use_horizontal_symmetry = False  # @param {type:"boolean"}
transformation_percent = [0.09]  # @param


# @markdown ####**Animation Mode:**
animation_mode = 'None'  # @param ['None', '2D', '3D', 'Video Input'] {type:'string'}
# @markdown *For animation, you probably want to turn `cutn_batches` to 1 to make it quicker.*


# @markdown ---

# @markdown ####**Video Input Settings:**
# if is_colab:
#     video_init_path = "/content/drive/MyDrive/init.mp4" #@param {type: 'string'}
video_init_path = "init.mp4"  # @param {type: 'string'}
extract_nth_frame = 2  # @param {type: 'number'}
persistent_frame_output_in_batch_folder = True  # @param {type: 'boolean'}
video_init_seed_continuity = False  # @param {type: 'boolean'}
# @markdown #####**Video Optical Flow Settings:**
video_init_flow_warp = True  # @param {type: 'boolean'}
# Call optical flow from video frames and warp prev frame with flow
video_init_flow_blend = 0.999  # @param {type: 'number'} #0 - take next frame, 1 - take prev warped frame
video_init_check_consistency = False  # Insert param here when ready
video_init_blend_mode = "optical flow"  # @param ['None', 'linear', 'optical flow']
# Call optical flow from video frames and warp prev frame with flow


# @markdown ####**2D Animation Settings:**
# @markdown `zoom` is a multiplier of dimensions, 1 is no zoom.
# @markdown All rotations are provided in degrees.

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

# ======= TURBO MODE
# @markdown ---
# @markdown ####**Turbo Mode (3D anim only):**
# @markdown (Starts after frame 10,) skips diffusion steps and just uses depth map to warp images for skipped frames.
# @markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames.
# @markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

turbo_mode = False  # @param {type:"boolean"}
turbo_steps = "3"  # @param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 10  # frames

# @markdown ####**Coherency Settings:**
# @markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
frames_scale = 1500  # @param{type: 'integer'}
# @markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
frames_skip_steps = '60%'  # @param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}

# @markdown ####**Video Init Coherency Settings:**
# @markdown `frame_scale` tries to guide the new frame to looking like the old one. A good default is 1500.
video_init_frames_scale = 15000  # @param{type: 'integer'}
# @markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
video_init_frames_skip_steps = '70%'  # @param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}


# Note: If using a pixelart diffusion model, try adding "#pixelart" to the end of the prompt for a stronger effect. It'll tend to work a lot better!
# text_prompts = {
#     0: ["蓝天白云"],
# }

image_prompts = {
    # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
}


# @title Do the Run!
# @markdown `n_batches` ignored with animation modes.
display_rate = 2  # @param{type: 'number'}
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
