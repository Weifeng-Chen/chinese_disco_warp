# from disco_huge import Diffuser
from turtle import width
from disco import Diffuser
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
@st.cache(show_spinner=False, allow_output_mutation=True)   # 加装饰器， 只加载一次。
class ST_Diffuser(Diffuser):
    def __init__(self, custom_path):
        super().__init__(custom_path)

    # @st.cache(show_spinner=False)   # 加装饰器， 只加载一次模型
    # def model_setup(self, custom_path):
    #     return super().model_setup(custom_path)

if __name__ == '__main__':
    dd = ST_Diffuser(custom_path='./models/cyberpunk_ema_160000.pt')  # 初始化
    # dd = ST_Diffuser(custom_path='./models/512x512_diffusion_uncond_finetune_008100.pt')  # 初始化
    # dd = ST_Diffuser(custom_path='./models/nature_ema_160000.pt')  # 初始化
    # dd = ST_Diffuser(custom_path='./models/animal_ema_160000.pt')

    form = st.form("参数设置")
    input_text = form.text_input('输入文本生成图像:',value='',placeholder='你想象的一个画面')
    seg_text = form.text_input('The area you want to regenerate:',value='',placeholder='Specify the area you want to regenerate')
    form.form_submit_button("提交")
    # uploaded_file = st.file_uploader("上传初始化图片（可选）", type=["jpg","png","jpeg"])

    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("笔刷: ", 5, 25, 15)
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    image_scale = st.sidebar.slider('图像尺度', 0, 10000,)
    text_scale = st.sidebar.slider('文本尺度', 0, 10000, 5000)
    skip_steps = st.sidebar.slider('起始点', 0, 60, 10)

    clip_seg_mask = None

    c1, c2 = st.columns(2)    
    with c1:
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="#eee",
            background_color="",
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=256,
            width=256,
            drawing_mode="freedraw",
            point_display_radius= 0,
            key="canvas",
        )
    with c2:
        image_status = st.empty()

    mask = None
    with st.spinner('正在生成中...(预计1min内结束)'):
        capture_img = None
        if bg_image is not None:
        # To read file as bytes:
            bytes_data = bg_image.getvalue()
            #将字节数据转化成字节流
            bytes_data = BytesIO(bytes_data)
            #Image.open()可以读字节流
            capture_img = Image.open(bytes_data).convert('RGB').resize((512, 512))
            if seg_text:
                # 文本区域的选区优先级更高
                mask = dd.get_seg_mask(capture_img, seg_text) 
                mask = mask.cpu().numpy()       # 0.5分割区域太小了貌似

            elif canvas_result.image_data is not None:
                mask = canvas_result.image_data
                mask = mask[:,:,0]
                # print(mask)
            # image_status.image(bg_image, use_column_width=True)
        
        if input_text:
            # global text_prompts
            input_text_prompts = [input_text]
            # print(mask, type(mask))
            if not np.any(mask):
                print('global generate')
                # mask区域为空，此时直接执行全局的生成
                # image_status.image(mask[:,:,0], use_column_width=True)
                image = dd.generate(input_text_prompts,
                                    capture_img,
                                    inpainting_mode = False,
                                    clip_guidance_scale=text_scale,
                                    init_scale=image_scale,
                                    skip_steps=skip_steps,
                                    st_dynamic_image=image_status)   # 最终结果。实时显示修改generate里面的内容。
            else:
                # print(mask)
                print('local generate')
                image = dd.generate(input_text_prompts,
                                    capture_img,
                                    inpainting_mode = True,
                                    inpainting_mask = Image.fromarray(mask),
                                    clip_guidance_scale=text_scale,
                                    init_scale=image_scale,
                                    skip_steps=skip_steps,
                                    st_dynamic_image=image_status)   # 最终结果。实时显示修改generate里面的内容。
            # image_status.image(image, use_column_width=True)
        