from disco import Diffuser
import streamlit as st
from io import BytesIO
from PIL import Image

@st.cache(show_spinner=False, allow_output_mutation=True)   # 加装饰器， 只加载一次。
class ST_Diffuser(Diffuser):
    def __init__(self):
        super().__init__()

    # @st.cache(show_spinner=False)   # 加装饰器， 只加载一次模型
    # def model_setup(self, custom_path):
    #     return super().model_setup(custom_path)

if __name__ == '__main__':
    dd = ST_Diffuser()  # 初始化
    
    form = st.form("参数设置")
    input_text = form.text_input('输入文本生成图像:',value='',placeholder='你想象的一个画面（描述景色）')
    form.form_submit_button("提交")
    uploaded_file = st.file_uploader("上传初始化图片（可选）", type=["jpg","png","jpeg"])

    image_scale = 1000
    text_scale = 5000
    skip_steps = 10
    
    with st.spinner('正在生成中...(预计40s)'):
        capture_img = None
        if uploaded_file is not None:
        # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            #将字节数据转化成字节流
            bytes_data = BytesIO(bytes_data)
            #Image.open()可以读字节流
            capture_img = Image.open(bytes_data).convert('RGB').resize((512, 512))
            col1, col2, col3 = st.columns(3)
            with col1:
                image_scale = st.slider('图像尺度', 1000, 10000,)
            with col2:
                text_scale = st.slider('文本尺度', 1000, 10000, 5000)
            with col3:
                skip_steps = st.slider('起始点', 10, 60,)
            image_status = st.empty()
            image_status.image(capture_img, use_column_width=True)
        else:
            image_status = st.empty()

        if input_text:
            # global text_prompts
            input_text_prompts = [input_text]
            image = dd.generate(input_text_prompts,
                                capture_img,
                                clip_guidance_scale=text_scale,
                                init_scale=image_scale,
                                skip_steps=skip_steps,
                                st_dynamic_image=image_status)   # 最终结果。实时显示修改generate里面的内容。
            image_status.image(image, use_column_width=True)
    
