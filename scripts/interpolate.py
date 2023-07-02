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
        min_denoising_strength = gr.Slider(label="Min denoising strength", minimum=0.0, maximum=1.0, value=0.35)
        controlnet_weight_master = gr.Slider(label="Controlnet weight master", minimum=-2.0, maximum=2.0, value=1.0)
        steps = gr.Number(label="Intermediate steps")
        output_directory = gr.Textbox(label="Output directory")
        return [last_image, min_denoising_strength, controlnet_weight_master, steps, output_directory]

    def run(self, p, last_image: Image.Image, min_denoising_strength: float, controlnet_weight_master: float, steps: float, output_directory: str):
        controlnet = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
        temporalnet_model = [m for m in controlnet.get_models() if 'temporalnet' in m.lower()][0]
        canny_model = [m for m in controlnet.get_models() if 'canny' in m.lower()][0]
        self.controlnet_weight_master = controlnet_weight_master
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

        self.output_directory = output_directory
        self.min_denoising_strength = min_denoising_strength
        self.total_steps = steps
        start_image = p.init_images[0]
        start_image.save(str(Path(output_directory) / f"{0:06d}.png"))
        last_image.save(str(Path(output_directory) / f"{int(steps)-1:06d}.png"))
        all_images = self.bisect_run(p, p.init_images[0], last_image, 0, int(steps) - 1, 0, 0)
        all_images = [start_image] + all_images + [last_image]

        return processing.Processed(p, all_images)

    def bisect_run(
        self,
        p,
        start_image: Image.Image,
        end_image: Image.Image,
        start_step: int,
        end_step: int,
        start_depth: int,
        end_depth: int,
    ) -> List[Image.Image]:
        if end_step - start_step == 1 or shared.state.interrupted:
            return []

        depth_difference = start_depth - end_depth
        middle_step = (end_step + start_step) // 2
        max_depth = max(1, math.ceil(math.log2(self.total_steps)))
        freedom = math.log(end_step - start_step, self.total_steps - 1)
        print(freedom)

        next_p = copy(p)
        next_p.do_not_save_samples = True
        # bezier interpolation
        next_p.init_images[0] = Image.blend(start_image, end_image, middle_step / (self.total_steps - 1))

        self.cn_units[0].image = numpy.array(start_image)
        self.cn_units[0].weight = self.controlnet_weight_master * (1 - ((depth_difference / max_depth) + 1) / 2)

        # self.cn_units[2].image = numpy.array(start_image)
        # self.cn_units[2].weight = self.controlnet_weight_master * (1 - middle_ratio) * (1 - (middle_step - start_step) / self.total_steps)

        self.cn_units[1].image = numpy.array(end_image)
        self.cn_units[1].weight = self.controlnet_weight_master * ((depth_difference / max_depth) + 1) / 2

        # self.cn_units[3].image = numpy.array(end_image)
        # self.cn_units[3].weight = self.controlnet_weight_master * middle_ratio * (1 - (end_step - middle_step) / self.total_steps)

        min_denoising_strength = self.min_denoising_strength * ((depth_difference / max_depth) + 1) / 2
        next_p.denoising_strength *= min_denoising_strength + freedom * (1 - min_denoising_strength)
        middle_image = processing.process_images(next_p).images[0]
        middle_image.save(str(Path(self.output_directory) / f"{middle_step:06d}.png"))

        middle_depth = max(start_depth, end_depth) + 1
        before_images = self.bisect_run(p, start_image, middle_image, start_step, middle_step, start_depth, middle_depth)
        after_images = self.bisect_run(p, middle_image, end_image, middle_step, end_step, middle_depth, end_depth)
        return before_images + [middle_image] + after_images
