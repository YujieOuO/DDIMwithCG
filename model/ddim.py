
"""
 This code is based on the implementation of DDIM with Classifier-Guidance in the following repository:
 https://github.com/Project-MONAI/GenerativeModels
 tutorials/generative/anomaly_detection/anomalydetection_tutorial_classifier_guidance.py 
"""

import os
import torch
import pytorch_lightning as pl
from torch import Tensor
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelEncoder, DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
from time import time

torch.multiprocessing.set_sharing_strategy("file_system")

# enable deterministic training
set_determinism(42)


def get_diffusion_model(config, load_weights=True):
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(64, 64, 64),
        attention_levels=(False, False, True),
        num_res_blocks=1,
        num_head_channels=64,
        with_conditioning=False,
    )
    if load_weights:
        weights_path = os.path.join(config['pretrained_dir'], "best_diffusion_model.pth")
        model.load_state_dict(torch.load(weights_path))
    return model


def get_classifier(config, load_weights=True):

    model = DiffusionModelEncoder(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        num_res_blocks=(1, 1, 1),
        num_head_channels=64,
        with_conditioning=False,
    )
    if load_weights:
        weights_path = os.path.join(config['pretrained_dir'], "best_classifier_model.pth")
        model.load_state_dict(torch.load(weights_path))
    return model


class DDIMwCG:

    """ DDIM model with classifier-guidance
     this class is used for inference of anomaly detection (not training)
     对已经训练好的DM利用已经训练好的Classifier的梯度进行infer采样指导
    """

    def __init__(self, config):

        super().__init__()
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## 导入训练好的DM
        self.diffusion_model = DDIM(config)
        weights_path = os.path.join(config['ddim_log_dir'], "weights.pth")
        self.diffusion_model.load_state_dict(torch.load(weights_path))
        self.diffusion_model = self.diffusion_model.to(self.device).eval()
        self.diffusion_model = self.diffusion_model.diffusion_model

        ## 导入训练好的分类器
        self.classifier = Classifier(config)
        weights_path = os.path.join(config['classifier_log_dir'], "weights.pth")
        self.classifier.load_state_dict(torch.load(weights_path))
        self.classifier = self.classifier.to(self.device).eval()
        self.classifier = self.classifier.classifier


        ## scheduler决定了infer时的加噪方式
        self.scheduler = DDIMScheduler(num_train_timesteps=1000)

        ## 这里的infer指的什么？
        self.inferer = DiffusionInferer(self.scheduler)

        self.L = self.config['L']      # noise level L
        self.scale = self.config['s']  # gradient-scale s

    def eval(self):
        self.diffusion_model.eval()
        self.classifier.eval()

    def detect_anomaly(self, x: Tensor):
        """ detect anomaly using diffusion model with classifier-guidance """
        t_start = time()
        x = x.to(self.device)
        rec = x.clone()
        self.scheduler.set_timesteps(num_inference_steps=1000)

        ## 这里存在一个差异，该模型是输入一个原图
        ## 然后反转得到噪声图，对噪声图进行CG指导去噪
        ## 我们的是纯噪声图由text指导生成任意图，CG指导的是text约束下任意图的生成
        print("\nnoising process...")
        progress_bar = tqdm(range(self.L-1))  # go back and forth L timesteps
        for t in progress_bar:  # go through the noising process
            ## 自动混合精度autocase
            with autocast(enabled=False):
                with torch.no_grad():
                    model_output = self.diffusion_model(rec, timesteps=torch.Tensor((t,)).to(rec.device))
            rec, _ = self.scheduler.reversed_step(model_output, t, rec)
            rec = torch.clamp(rec, -1, 1)


        ## 下面才是真正的分类器指导去噪
        print("denoising process...")

        ## 分类器标签
        y = torch.tensor(0)  # define the desired class label
        progress_bar = tqdm(range(self.L-1,-1,-1))  # go back and forth L timesteps
        for t in progress_bar:  # go through the denoising process
            # t = self.L - i
            with autocast(enabled=True):
                with torch.no_grad():
                    ## 对于输入的噪声图，经过DM预测噪声epsilon
                    model_output = self.diffusion_model(
                        rec, timesteps=torch.Tensor((t,)).to(rec.device)
                    ).detach()  # this is supposed to be epsilon

                with torch.enable_grad():

                    ## 将每一步的输入的噪声图作为分类器的输入
                    ## detach获取rec的值, 创建新张量x_in，同时赋予x_in梯度属性
                    ## 因为要利用分类器的输出的梯度影响每一步的预测噪声的结果
                    x_in = rec.detach().requires_grad_(True)

                    ## 得到分类器的logits，概率
                    logits = self.classifier(x_in, timesteps=torch.Tensor((t,)).to(rec.device))
                    log_probs = F.log_softmax(logits, dim=-1)

                    ## 关键一步，提取标签为1的概率大小
                    ## 这个概率反映了输入数据趋向于正类的幅度，概率越大，说明输入的噪声图越像正样本
                    selected = log_probs[range(len(logits)), y.view(-1)]

                    # get gradient C(x_t) regarding x_t 

                    ## 计算输入数据在标签为1的方向上的梯度a
                    a = torch.autograd.grad(selected.sum(), x_in)[0]

                    ## 得到当前时间步的倍数因子alpha
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]

                    ## 利用标签1的梯度指导模型预测的噪声往梯度方向更新
                    ## 自定义梯度的权重参数self.scale
                    updated_noise = (
                        model_output - (1 - alpha_prod_t).sqrt() * self.scale * a
                    )  # update the predicted noise epsilon with the gradient of the classifier
            ## 利用每一步梯度优化后的预测噪声，来更新下一步的输入
            rec, _ = self.scheduler.step(updated_noise, t, rec)
            rec = torch.clamp(rec, -1, 1)

            ## 清空CUDA缓存，释放所有未使用的GPU内存
            torch.cuda.empty_cache()

        # anomaly detection
        anomaly_map = torch.abs(x - rec)
        anomaly_score = torch.sum(anomaly_map, dim=(1, 2, 3))
        print(f'total inference-time: {time() - t_start:.2f}sec\n')
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }


class DDIM(pl.LightningModule):
    
    ## 前向加噪，利用每一步的加噪图训练分类器
    """ DDIM (Denoising Diffusion Implicit Model) model 
     Class for training DDIM model.
    """

    def __init__(self, config, load_weights=False):
        super().__init__()
        self.save_hyperparameters('config')

        # use custom optimization in the training loop
        self.automatic_optimization = False

        self.config = config

        self.diffusion_model = get_diffusion_model(config, load_weights)
        self.diffusion_model.to(self.device)

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.inferer = DiffusionInferer(self.scheduler)

        # diffusion training
        self.scaler = GradScaler()
        self.optimizer = self.configure_optimizers()
        self.loss_fn = F.mse_loss


    def forward(self, x: Tensor, timesteps: Tensor):
        self.diffusion_model(x, timesteps)
        return x

    def training_step(self, batch: Tensor, batch_idx):
        
        images = batch
        self.optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(self.device)  # pick a random time step t

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(self.device)

            # Get model prediction (1) adds noise to image (2) predicts noise
            noise_pred = self.inferer(inputs=images, diffusion_model=self.diffusion_model, 
                                      noise=noise, timesteps=timesteps)
            loss = self.loss_fn(noise_pred.float(), noise.float())

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log('loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx):

        images = batch
        timesteps = torch.randint(0, 1000, (len(images),)).to(self.device)
        with torch.no_grad():
            with autocast(enabled=True):
                noise = torch.randn_like(images).to(self.device)
                noise_pred = self.inferer(inputs=images, diffusion_model=self.diffusion_model, 
                                        noise=noise, timesteps=timesteps)
                val_loss = self.loss_fn(noise_pred.float(), noise.float())

        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True, on_step=False)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])



class Classifier(pl.LightningModule):

    """ Classifier model 
     Class for training classifier model.
    """

    def __init__(self, config, load_weights=False):
        super().__init__()
        self.save_hyperparameters('config')

        # use custom optimization in the training loop
        self.automatic_optimization = False
        
        self.config = config

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = get_classifier(config, load_weights)
        self.classifier.to(self.device)

        ## 这里采用的是DDIM的前向加噪，但是DDIM一般为反向去噪
        self.scheduler = DDIMScheduler(num_train_timesteps=1000)

        self.optimizer = self.configure_optimizers()
        self.loss_fn = F.cross_entropy

    def forward(self, x: Tensor, timesteps: Tensor):
        pred = self.classifier(x, timesteps)
        return pred

    def training_step(self, batch: dict, batch_idx):
        images = batch["image"]
        classes = batch["slice_label"]
        self.optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(self.device)

        with autocast(enabled=False):
            # Generate random noise
            noise = torch.randn_like(images).to(self.device)

            # Get model prediction
            noisy_img = self.scheduler.add_noise(images, noise, timesteps)  # add t steps of noise to the input image
            pred = self(noisy_img, timesteps)
            loss = self.loss_fn(pred, classes.long())

            loss.backward()
            self.optimizer.step()
        
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        images = batch["image"]
        classes = batch["slice_label"]
        timesteps = torch.randint(0, 1, (len(images),), device=self.device)
        # check validation accuracy on the original images, i.e., do not add noise

        with torch.no_grad():
            with autocast(enabled=False):
                pred = self(images, timesteps)
                val_loss = self.loss_fn(pred, classes.long())
        
        self.log('val_loss', val_loss, prog_bar=True, on_epoch=True, on_step=False)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr_cls'])

