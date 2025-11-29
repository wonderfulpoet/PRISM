from PRISM.utils import Logger
from PRISM.tester import Tester
from PRISM.models.PRISM import PRISMTest
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class PRISMTester(Tester):
    def __init__(self, **params):
        super(PRISMTester, self).__init__(**params)
    
    def get_image(self, data):
        data_opts = self.opts.dataset
        dataset_type = data_opts["dataset_type"]
        if dataset_type == "deep_lesion":
            if data_opts[dataset_type]["load_mask"]:
                return data['A'], data['D'], data['C'], data['data_name']
            else:
                return data['A'], data['D'], data['C'], data['data_name']
        elif dataset_type == "spineweb":
            return data['a'], data['b'], data['c'], data["data_name"]
        elif dataset_type == "nature_image":
            return data['artifact'], data['no_artifact'], data["data_name"]

    def get_metric(self, metric):
        def measure(x, y):
            x = self.dataset.to_numpy(x, False)
            y = self.dataset.to_numpy(y, False)
            x = x * 0.5 + 0.5
            y = y * 0.5 + 0.5

            return metric(x, y, data_range=1.0)
        return measure

    def get_pairs(self):
        if hasattr(self.model, 'mask'):
            mask = self.model.mask
            img_low = self.model.img_low * mask
            img_high = self.model.img_high * mask
            pred_lh = self.model.pred_lh * mask
        else:
            img_low = self.model.img_low
            img_high = self.model.img_high
            pred_lh = self.model.pred_lh

        return [
            ("before", (img_low, img_high)), 
            ("after", (pred_lh, img_high))], self.model.name

    def get_visuals(self, n=8):
        lookup = [
            ("l", "x_low"), ("lh", "l_h"),
            ("h", "x_high"), ("hl", "h_l"), ("hh", "h_h")]
        visual_window = self.opts.visual_window
       
        def visual_func(x):
            x = x * 0.5 + 0.5
            x[x < visual_window[0]] = visual_window[0]
            x[x > visual_window[1]] = visual_window[1]
            x = (x - visual_window[0]) / (visual_window[1] - visual_window[0])
            return x

        return self.model._get_visuals(lookup, n, visual_func, False)

    def evaluate(self, model, data):
        model.evaluate(*data)     

if __name__ == "__main__":
    tester = PRISMTester(
        name="PRISM", model_class=PRISMTest,
        description="Test PRISM network")
    tester.run()
