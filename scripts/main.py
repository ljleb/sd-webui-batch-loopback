from modules import scripts, processing, shared
import gradio as gr


class BatchLoopbackScript(scripts.Script):
    def __init__(self):
        self.init_seed = -1
        self.init_subseed = -1
        self.seed = -1
        self.subseed = -1

    def title(self):
        return 'Batch Loopback'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        extension_name = 'batch_loopback'
        def format_elem_id(name):
            return f'{extension_name}_{name}'

        with gr.Accordion(label='Batch Loopback', open=False, elem_id=extension_name):
            increment_seed = gr.Checkbox(
                label='Increment seed during external batch processing',
                value=False,
                elem_id=format_elem_id('increment_seed'),
            )

        return [increment_seed]

    def process(self, p, increment_seed):
        if shared.state.job_no == 0:
            self.batch_tab_process(p, increment_seed)

    def process_batch(self, p, increment_seed, **kwargs):
        if shared.state.job_no > 0 and p.iteration == 0:
            self.batch_tab_process_each(p, increment_seed)

    def postprocess(self, p, processed, increment_seed):
        if shared.state.job_no >= shared.state.job_count:
            self.batch_tab_postprocess_each(p, increment_seed)

    def batch_tab_process(self, p, increment_seed):
        if increment_seed:
            processing.fix_seed(p)
            self.init_seed = self.seed = p.seed
            self.init_subseed = self.subseed = p.subseed

    def batch_tab_process_each(self, p, increment_seed):
        if increment_seed:
            if self.seed != -1:
                self.seed += p.n_iter
            p.seed = self.seed
            if self.subseed != -1:
                self.subseed += p.n_iter
            p.subseed = self.subseed

    def batch_tab_postprocess_each(self, p, increment_seed):
        if increment_seed:
            p.seed = self.init_seed
            p.subseed = self.subseed
            p.all_seeds = [self.init_seed for _ in range(len(p.all_seeds))]
            p.all_subseeds = [self.init_seed for _ in range(len(p.all_subseeds))]
