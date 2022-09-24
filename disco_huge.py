from utils import *
import random
import json
import lpips
import gc
from secondary_model import SecondaryDiffusionImageNet2
from transformers import BertModel, BertTokenizer
import clip
from types import SimpleNamespace
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from ipywidgets import Output
from datetime import datetime
from tqdm.notebook import tqdm
from glob import glob
import time
import open_clip

class Diffuser:
    def __init__(self, cutom_path='/home/chenweifeng/disco_project/models/nature_ema_160000.pt'):
        self.model_setup(cutom_path)
        # self.current_image = None
        pass

    def model_setup(self, custom_path):
        # LOADING MODEL
        self.lpips_model = lpips.LPIPS(net='vgg').to(device)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        print(f'Prepping model...model name: {diffusion_model}')
        self.model, self.diffusion = create_model_and_diffusion(**model_config)
        if diffusion_model == 'custom':
            self.model.load_state_dict(torch.load(custom_path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(f'{model_path}/{get_model_filename(diffusion_model)}', map_location='cpu'))
        self.model.requires_grad_(False).eval().to(device)
        for name, param in self.model.named_parameters():
            if 'qkv' in name or 'norm' in name or 'proj' in name:
                param.requires_grad_()
        if model_config['use_fp16']:
            self.model.convert_to_fp16()
        print(f'Diffusion_model Loaded {diffusion_model}')

        # NOTE Directly Load The Text Encoder From Hugging Face
        print(f'Prepping model...model name: CLIP')
        self.taiyi_tokenizer = BertTokenizer.from_pretrained("/home/chenweifeng/fengshen/Taiyi-CLIP-Roberta-326M-ViT-H-Chinese")
        self.taiyi_transformer = BertModel.from_pretrained("/home/chenweifeng/fengshen/Taiyi-CLIP-Roberta-326M-ViT-H-Chinese").eval().to(device)
        self.clip_models = []
        # if ViTB32:
        #     self.clip_models.append(clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device))
        # if ViTB16:
        #     self.clip_models.append(clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device))
        # if ViTL14:
        #     self.clip_models.append(clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device))
        # if ViTL14_336px:
        # self.clip_models.append(clip.load('ViT-L/14@336px', jit=False)[0].eval().requires_grad_(False).to(device))
        self.clip_models.append(open_clip.create_model('ViT-H-14', pretrained='laion2b_s32b_b79k').eval().requires_grad_(False).to(device))
        print(f'CLIP Loaded')

    def generate(self, 
                    input_text_prompts=['夕阳西下'], 
                    init_image=None, 
                    skip_steps=10,
                    clip_guidance_scale=7500,
                    init_scale=2000,
                    st_dynamic_image=None,
                    seed = None,
                    ):

        seed = seed
        frame_num = 0
        init_image = init_image
        init_scale = init_scale
        skip_steps = skip_steps
        loss_values = []
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        target_embeds, weights = [], []
        frame_prompt = input_text_prompts

        print(args.image_prompts_series)
        if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
            image_prompt = args.image_prompts_series[-1]
        elif args.image_prompts_series is not None:
            image_prompt = args.image_prompts_series[frame_num]
        else:
            image_prompt = []

        print(f'Frame {frame_num} Prompt: {frame_prompt}')

        model_stats = []
        for clip_model in self.clip_models:
            cutn = 16
            model_stat = {"clip_model": None, "target_embeds": [], "make_cutouts": None, "weights": []}
            model_stat["clip_model"] = clip_model

            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                # txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
                # NOTE use chinese CLIP
                txt = self.taiyi_transformer(self.taiyi_tokenizer(txt, return_tensors='pt')['input_ids'].to(device))[1]
                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0, 1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs)
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert('RGB')
                    img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
                    batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                    embed = clip_model.encode_image(normalize(batch)).float()
                    if fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0, 1))
                            weights.extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        init = None
        if init_image is not None:
            # init = Image.open(fetch(init_image)).convert('RGB')   # 传递的是加载好的图片。而非地址~
            init = init_image
            init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

            if args.perlin_init:
                # NOTE在原始图像上加perlin（柏林噪声）
                if args.perlin_mode == 'color':
                    init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
                    init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
                elif args.perlin_mode == 'gray':
                    init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
                    init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
                else:
                    init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
                    init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
                # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
                init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
                del init2

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if use_secondary_model is True:
                    alpha = torch.tensor(self.diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    sigma = torch.tensor(self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = self.diffusion.p_mean_variance(self.model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                    fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out['pred_xstart'] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = int(t.item())+1  # errors on last step without +1, need to find source
                        # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution = model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution = 224

                        cuts = MakeCutoutsDango(input_resolution,
                                                Overview=args.cut_overview[1000-t_int],
                                                InnerCrop=args.cut_innercut[1000-t_int],
                                                IC_Size_Pow=args.cut_ic_pow[1000-t_int],
                                                IC_Grey_P=args.cut_icgray_p[1000-t_int],
                                                args=args,
                                                )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([args.cut_overview[1000-t_int]+args.cut_innercut[1000-t_int], n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item())  # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / cutn_batches
                tv_losses = tv_loss(x_in)
                if use_secondary_model is True:
                    range_losses = range_loss(out)
                else:
                    range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
                if init is not None and init_scale:
                    init_losses = self.lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any() == False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=args.clamp_max) / magnitude  # min=-0.02, min=-clamp_max,
            return grad

        if args.diffusion_sampling_mode == 'ddim':
            sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = self.diffusion.plms_sample_loop_progressive

        for i in range(args.n_batches):
            current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
            
            batchBar = tqdm(range(args.n_batches), desc="Batches")
            batchBar.n = i
            batchBar.refresh()
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = self.diffusion.num_timesteps - skip_steps - 1
            total_steps = cur_t

            if perlin_init:
                init = regen_perlin(device)

            if args.diffusion_sampling_mode == 'ddim':
                samples = sample_fn(
                    self.model,
                    (batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    eta=eta,
                    transformation_fn=symmetry_transformation_fn,
                    transformation_percent=args.transformation_percent
                )
            else:
                samples = sample_fn(
                    self.model,
                    (batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    order=2,
                )

            for j, sample in enumerate(samples):
                cur_t -= 1
                intermediateStep = False
                if args.steps_per_checkpoint is not None:
                    if j % steps_per_checkpoint == 0 and j > 0:
                        intermediateStep = True
                elif j in args.intermediate_saves:
                    intermediateStep = True
                if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                    for k, image in enumerate(sample['pred_xstart']):
                        # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                        percent = math.ceil(j/total_steps*100)
                        if args.n_batches > 0:
                            filename = f'{current_time}-{parse_prompt(prompt)[0]}.png'
                        image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                        if j % args.display_rate == 0 or cur_t == -1:
                            image.save(f'{outDirPath}/{filename}')
                            if st_dynamic_image:
                                st_dynamic_image.image(image, use_column_width=True)
                            # self.current_image = image
        return image
    # def get_current_image(self, ):
    #     return self.current_image
    

if __name__ == '__main__':
    dd = Diffuser('./models/nature_ema_160000.pt')    # 自然风格图像的模型。
    image_scale = 1000
    text_scale = 5000
    skip_steps = 10
    dd.generate(['白茫茫的一片都是雪'] , 
                # init_image=Image.open(fetch('./sunset.jpg')).convert('RGB'),
                clip_guidance_scale=text_scale,
                init_scale=image_scale,
                skip_steps=skip_steps,)