import os
import os.path as path
import argparse
from PRISM.utils import get_config, update_config, save_config, get_last_checkpoint, add_post, Logger
from PRISM.datasets import get_dataset_train
from PRISM.models import PRISMTrain
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PRISM network")
    parser.add_argument("run_name", help="name of the run")
    parser.add_argument("--default_config", default="config/PRISM.yaml", help="default configs")
    parser.add_argument("--run_config", default="runs/PRISM.yaml", help="run configs")
    args = parser.parse_args()

    opts = get_config(args.default_config)
    run_opts = get_config(args.run_config)
    if args.run_name in run_opts and "train" in run_opts[args.run_name]:
        run_opts = run_opts[args.run_name]["train"]
        update_config(opts, run_opts)
    run_dir = path.join(opts["checkpoints_dir"], args.run_name)
    if not path.isdir(run_dir): os.makedirs(run_dir)

    def get_image(data):
        dataset_type = dataset_opts['dataset_type']
        if dataset_type == "deep_lesion":
            return data['data_name'], data['A'], data['D'], data['C']
        elif dataset_type == "spineweb":
            return data['a'], data['b'], data['c']
        elif dataset_type == "nature_image":
            return data["artifact"], data["no_artifact"]
        else:
            raise ValueError("Invalid dataset type!")

    dataset_opts = opts['dataset']
    train_dataset = get_dataset_train(**dataset_opts)
    train_loader = DataLoader(train_dataset,
        batch_size=opts["batch_size"], num_workers=opts['num_workers'], shuffle=True)
    train_loader = add_post(train_loader, get_image)

    if opts['last_epoch'] == 'last':
        checkpoint, start_epoch = get_last_checkpoint(run_dir)
    else:
        start_epoch = opts['last_epoch']
        checkpoint = path.join(run_dir, "net_{}".format(start_epoch))
        if type(start_epoch) is not int: start_epoch = 0

    model = PRISMTrain(opts['learn'], opts['loss'], **opts['model'])
    if opts['use_gpu']: model.cuda()

    logger = Logger(run_dir, start_epoch, args.run_name)
    logger.add_loss_log(model.get_loss, opts["print_step"], opts['window_size'])
    logger.add_iter_visual_log(model.get_visuals, opts['visualize_step'], "train_visuals")
    logger.add_save_log(model.save, opts['save_step'])

    for epoch in range(start_epoch, opts['num_epochs']):
        for data in logger(train_loader):
            model.optimize(*data)
        model.update_lr()

