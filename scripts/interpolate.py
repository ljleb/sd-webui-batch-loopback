import math
from pathlib import Path
from typing import List

import numpy

from modules import scripts, processing, images, shared
import gradio as gr
from PIL import Image
import importlib
from copy import copy

from modules.shared import opts


class Script(scripts.Script):
    def title(self):
        return "Interpolate"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        last_image = gr.Image(label="Last image", type="pil")
        steps = gr.Number(label="Intermediate steps")
        output_directory = gr.Textbox(label="Output directory")
        return [last_image, steps, output_directory]

    def run(self, p, last_image: Image.Image, steps: float, output_directory: str):
        controlnet = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
        temporalnet_model = [m for m in controlnet.get_models() if 'temporalnet' in m.lower()][0]
        canny_model = [m for m in controlnet.get_models() if 'canny' in m.lower()][0]
        self.cn_units = [
            controlnet.ControlNetUnit(
                model=temporalnet_model,
                pixel_perfect=True,
                control_mode=controlnet.ControlMode.CONTROL,
            ),
            controlnet.ControlNetUnit(
                model=temporalnet_model,
                pixel_perfect=True,
                control_mode=controlnet.ControlMode.CONTROL,
            ),
            # controlnet.ControlNetUnit(
            #     model=canny_model,
            #     module="canny",
            #     pixel_perfect=True,
            #     control_mode=controlnet.ControlMode.CONTROL,
            # ),
            # controlnet.ControlNetUnit(
            #     model=canny_model,
            #     module="canny",
            #     pixel_perfect=True,
            #     control_mode=controlnet.ControlMode.CONTROL,
            # ),
        ]
        controlnet.update_cn_script_in_processing(p, self.cn_units)
        # processing.fix_seed(p)

        if shared.state.job_count == -1:
            shared.state.job_count = steps

        self.total_steps = steps
        start_image = p.init_images[0]
        all_images = self.bisect_run(p, p.init_images[0], last_image, 0, int(steps) - 1)
        all_images = [start_image] + all_images + [last_image]
        if not p.do_not_save_samples:
            for i, image in enumerate(all_images):
                image.save(str(Path(output_directory) / f"{i:06d}.png"))

        return processing.Processed(p, all_images)

    def bisect_run(self, p, start_image: Image.Image, end_image: Image.Image, start_step: int, end_step: int) -> List[Image.Image]:
        if end_step - start_step == 1 or shared.state.interrupted:
            return []

        middle_step = (end_step + start_step) // 2
        middle_ratio = (middle_step - start_step) / (end_step - start_step)
        freedom = math.log(end_step - start_step, self.total_steps - 1)
        print(freedom)

        next_p = copy(p)
        next_p.do_not_save_samples = True
        next_p.init_images[0] = Image.blend(start_image, end_image, middle_ratio)

        self.cn_units[0].image = numpy.array(start_image)
        self.cn_units[0].weight = (1 - middle_ratio) * (1 - (middle_step - start_step) / self.total_steps)

        # self.cn_units[2].image = numpy.array(start_image)
        # self.cn_units[2].weight = (1 - middle_ratio) * (1 - (middle_step - start_step) / self.total_steps)

        self.cn_units[1].image = numpy.array(end_image)
        self.cn_units[1].weight = middle_ratio * (1 - (end_step - middle_step) / self.total_steps)

        # self.cn_units[3].image = numpy.array(end_image)
        # self.cn_units[3].weight = middle_ratio * (1 - (end_step - middle_step) / self.total_steps)

        next_p.denoising_strength *= freedom
        middle_image = processing.process_images(next_p).images[0]

        before_images = self.bisect_run(p, start_image, middle_image, start_step, middle_step)
        after_images = self.bisect_run(p, middle_image, end_image, middle_step, end_step)
        return before_images + [middle_image] + after_images
