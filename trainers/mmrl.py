import os
import os.path as osp
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss
# import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import random
import math
import numpy as np
_tokenizer = _Tokenizer()

dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}
CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.'
}



def shrink_and_block_tile(imgs):
    """
    输入: imgs: [B, 3, 224, 224]
    输出: tiled_imgs: [B, 3, 224, 224] 由4个缩小图拼接而成的图
    """
    B, C, H, W = imgs.shape
    assert H == 224 and W == 224, "图像尺寸必须是 224x224"

    # 1. 缩小每张图像为 56x56
    imgs_small = F.interpolate(imgs, size=(112, 112), mode='bilinear', align_corners=False)  # [B, 3, 56, 56]

    # 2. 创建4个 tile（你也可以换成不同的图像，但这里我们用同一张重复4次）
    tile_0 = imgs_small
    tile_1 = imgs_small
    tile_2 = imgs_small
    tile_3 = imgs_small

    # 3. 拼成 2×2 block → [B, 3, 224, 224]
    top = torch.cat([tile_0, tile_1], dim=-1)     # [B, 3, 56, 112]
    bottom = torch.cat([tile_2, tile_3], dim=-1)  # [B, 3, 56, 112]
    tiled_imgs = torch.cat([top, bottom], dim=-2) # [B, 3, 112, 224]
    tiled_imgs = F.interpolate(tiled_imgs, size=(224, 224), mode='bilinear', align_corners=False)  # [B, 3, 56, 56]    
    return tiled_imgs



def load_clip_to_cpu(cfg, model_name="CLIP"):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"model": model_name,
                      "rep_tokens_layers": cfg.TRAINER.MMRL.REP_LAYERS,
                      "n_rep_tokens": cfg.TRAINER.MMRL.N_REP_TOKENS,
                      "proj_lora_dim": cfg.TRAINER.MMRL.PROJ_LORA_DIM,
                      "beta": cfg.TRAINER.MMRL.BETA}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os

    os.makedirs("cache/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache/{dataset_name}_clip_weights_random.pt")
    return clip_weights

def hpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os

    os.makedirs("cache_hpt/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache_hpt/{dataset_name}_clip_weights_random.pt")
    return clip_weights

def hpt_clip_classifier_all(classnames, gpt_prompts, clip_model, dataset_name):
    import os

    os.makedirs("cache_all/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache_all/{dataset_name}_clip_weights_random.pt")
    return clip_weights

class TextEncoder_MMRL(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text):

        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        eot_index = tokenized_prompts.argmax(dim=-1)
        combined = [x, compound_rep_tokens_text, 0, eot_index]  # third argument is the counter which denotes depth of representation tokens
        outputs = self.transformer(combined)

        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), eot_index + n_rep_tokens] @ self.text_projection
 
        
        return x




class TextEncoder_CLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        outputs = self.transformer(x)

        x = outputs
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  
        return x


def _get_text_base_features_zero_shot(cfg, classnames, clip_model, text_encoder, gpt_prompts):
    device = next(text_encoder.parameters()).device

    text_encoder = text_encoder.cuda()
    dataset = cfg.DATASET.NAME
    template = CUSTOM_TEMPLATES[dataset]
 
    with torch.no_grad():
        tokenized_prompts = []
        for text in tqdm(classnames, desc="Extracting text features"):
            classname = text.replace("_", " ")
            # i = random.randint(0, len(gpt_prompts[classname])-1)
            # text_random = gpt_prompts[classname][i]
            # text = template.format(text.replace('_', ' ')) + ', ' + text_random + '. '
            tokens = clip.tokenize(template.format(text.replace('_', ' ')))  #(n_tokens)
            tokens = tokens.to(device)
            tokenized_prompts.append(tokens) 
        tokenized_prompts = torch.cat(tokenized_prompts) # (n_classes, n_tokens)  

        embeddings = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype) # (n_classes, n_tokens, embed_dim)
        outputs = text_encoder(embeddings.cuda(), tokenized_prompts.cuda()) 

        text_embeddings = outputs

    text_encoder = text_encoder.to(device)
    return text_embeddings

    
class Residual_Aligner(nn.Module):
    def __init__(self, weight, bias, rank):
        super(Residual_Aligner, self).__init__()
        self.weight_shape = weight.shape
        self.rank = rank

        self.A = nn.Parameter(torch.zeros(self.weight_shape[0], self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(self.rank, self.weight_shape[-1]))
        self.bias = nn.Parameter(bias.clone().detach()) if bias is not None else None

    def forward(self, x, weight):
        return F.linear(x, weight + (self.A @ self.B), self.bias)


class Shared_Residual_Representation_Aligner(nn.Module):
    def __init__(self, base_linear, num_layers, rank):
        super(Shared_Residual_Representation_Aligner, self).__init__()
        self.weight = nn.Parameter(base_linear.weight.clone().detach())
        self.srra = nn.ModuleList([Residual_Aligner(weight=self.weight, 
                                                                bias=base_linear.bias, 
                                                                rank=rank) for _ in range(num_layers)])
    def forward(self, x, idx):
        return self.srra[idx](x, self.weight)
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiModalRepresentationLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_rep_tokens = cfg.TRAINER.MMRL.N_REP_TOKENS
        self.dtype = clip_model.dtype

        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        rep_dim = cfg.TRAINER.MMRL.REP_DIM

        self.rep_layers_length = len(cfg.TRAINER.MMRL.REP_LAYERS)  # max=12
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        dataset = cfg.DATASET.NAME

        template = CUSTOM_TEMPLATES[dataset]
        
        tokenized_prompts = []
        for text in classnames:
            tokens = clip.tokenize(template.format(text.replace('_', ' ')))  # (n_tokens)
            tokenized_prompts.append(tokens)
        self.tokenized_prompts = torch.cat(tokenized_prompts)  # (n_classes, n_tokens)

        with torch.no_grad():
            self.prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype) # (n_classes, n_tokens, embed_dim)

        self.compound_rep_tokens = nn.Parameter(torch.empty(n_rep_tokens, rep_dim))
        nn.init.normal_(self.compound_rep_tokens, std=0.02)

        self.compound_rep_tokens_1 = nn.Parameter(torch.zeros(n_rep_tokens, rep_dim))
        nn.init.normal_(self.compound_rep_tokens_1, std=0.02)


        shared_layer_r2v = nn.Linear(rep_dim, visual_dim)
        single_layer_r2t = nn.Linear(rep_dim, text_dim)
        res_lora_dim = cfg.TRAINER.MMRL.RES_LORA_DIM
        
        self.srra_r2vproj = Shared_Residual_Representation_Aligner(shared_layer_r2v, self.rep_layers_length, res_lora_dim)
        self.compound_rep_tokens_r2tproj = _get_clones(single_layer_r2t, self.rep_layers_length)
        self.compound_rep_tokens_r2vproj = _get_clones(shared_layer_r2v, self.rep_layers_length)


    def forward(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []
 
        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            rep_tokens1 = self.compound_rep_tokens_1
            rep_mapped_to_text = self.compound_rep_tokens_r2tproj[index](rep_tokens)
            rep_mapped_to_visual_ori = self.compound_rep_tokens_r2vproj[index](rep_tokens1)

            rep_mapped_to_visual = self.srra_r2vproj(rep_tokens, index)
            rep_mapped_to_visual = 0.2*rep_mapped_to_visual_ori + rep_mapped_to_visual                       
            compound_rep_tokens_text.append(rep_mapped_to_text.type(self.dtype))
            compound_rep_tokens_visual.append(rep_mapped_to_visual.type(self.dtype))  

        return compound_rep_tokens_text, compound_rep_tokens_visual




class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, clip_weights):
        super().__init__()
        self.SCALE = cfg.TRAINER.MMRL.SCALE
        self.classnames = classnames
        self.ori_embedding = clip_weights
        self.representation_learner = MultiModalRepresentationLearner(cfg, classnames, clip_model).type(clip_model.dtype)
        self.tokenized_prompts = self.representation_learner.tokenized_prompts
        self.register_buffer("prompt_embeddings", self.representation_learner.prompt_embeddings)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MMRL(clip_model)
        self.dtype = clip_model.dtype
        self.text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None


    def forward(self, image):
        pre_trained_text_features = (
                self.ori_embedding
                / self.ori_embedding.norm(dim=-1, keepdim=True)
            )
        if self.representation_learner.training:
            compound_rep_tokens_text, compound_rep_tokens_visual = self.representation_learner()
            text_features = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts, compound_rep_tokens_text)
        else:
            if self.text_features_for_inference is None:
                self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference = self.representation_learner()
                self.text_features_for_inference = self.text_encoder(self.prompt_embeddings, self.tokenized_prompts, self.compound_rep_tokens_text_for_inference)

            compound_rep_tokens_text, compound_rep_tokens_visual = self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference
            text_features = self.text_features_for_inference

        image_features, image_features_rep = self.image_encoder([image.type(self.dtype), compound_rep_tokens_visual])
    
        scale = self.SCALE
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_rep = image_features_rep / image_features_rep.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = 100. * image_features @ text_features.t()
        logits_rep = 100. * image_features_rep @ text_features.t()
        logits_fusion = scale * logits + (1 - scale) * logits_rep
        logits_text = 100. * image_features_rep @ pre_trained_text_features.t()
        std_logits = logits.std()
        dynamic_scale = 1 / ((std_logits + 1e-6) * logits.numel())
        L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(logits_text / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * dynamic_scale
        
        return logits, logits_rep, logits_fusion, image_features, text_features, L_SCL_logits



class MMRL_Loss(_Loss):
    def __init__(self, reg_weight=1.0, scale=0.7):
        super(MMRL_Loss, self).__init__()
        self.reg_weight = reg_weight
        self.scale = scale 

    def forward(self, logits, logits_rep, 
                image_features, text_features, 
                image_features_clip, text_features_clip, 
                label, loss1):
        xe_loss1 = F.cross_entropy(logits, label)
        xe_loss2 = F.cross_entropy(logits_rep, label)
      
        # dynamic_scale = compute_dynamic_scale(logits, self.ep, self.m_ep, batch_size=4)
     
       
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-07)
        cossim_reg_img = 1 - torch.mean(F.cosine_similarity(image_features, image_features_clip, dim=1))
        cossim_reg_text = 1 - torch.mean(F.cosine_similarity(text_features, text_features_clip, dim=1))


        return self.scale * xe_loss1 + (1-self.scale) * xe_loss2 + self.reg_weight * cossim_reg_img + self.reg_weight * cossim_reg_text + loss1 #+ kl_loss #+ 0.1*cossim_reg_img #+ # 

import json

def load_classnames_from_txt(file_path):
    classnames = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:  # 跳过空行
                classnames.append(name)
    return classnames

@TRAINER_REGISTRY.register()
class MMRL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMRL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_clip_to_cpu(cfg)
        classnames_all = load_classnames_from_txt(f'classname/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.txt')
        if cfg.TRAINER.MMRL.PREC == "fp32" or cfg.TRAINER.MMRL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype
        with open(
            f"hpt/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json"
        ) as f:
            hpt_prompt = json.load(f)

        with open(
            f"gpt_file/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json"
        ) as f:
            gpt3_prompt = json.load(f)

        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_zero_shot, cfg.DATASET.NAME
        )
        clip_weights_hpt = hpt_clip_classifier(
            classnames, hpt_prompt, clip_model_zero_shot, cfg.DATASET.NAME
        )
       
        pre_trained_text_features = clip_weights[random.randint(0, clip_weights.shape[0] - 1)]
        pre_trained_text_features = pre_trained_text_features / pre_trained_text_features.norm(dim=-1, keepdim=True)

        clip_weights_hpt = clip_weights_hpt[random.randint(0, clip_weights_hpt.shape[0] - 1)]
        clip_weights_hpt = clip_weights_hpt / clip_weights_hpt.norm(dim=-1, keepdim=True)

      

        self.pre_trained_text_features = pre_trained_text_features + 0.001*clip_weights_hpt
        with torch.no_grad():
            self.text_encoder_clip = TextEncoder_CLIP(clip_model_zero_shot)
            text_features_clip = _get_text_base_features_zero_shot(cfg, classnames, clip_model_zero_shot, self.text_encoder_clip, gpt3_prompt)
           
            # text_features_clip = 0.1*pre_trained_text_features + text_features_clip
            self.text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)
          
            
        self.image_encoder_clip = clip_model_zero_shot.visual  
       
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.pre_trained_text_features)


        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["representation_learner", "image_encoder.proj_rep", "image_encoder.A", "image_encoder.B"]

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
            param.requires_grad_(update)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        

        self.image_encoder_clip.to(self.device)    
    
        reg_weight = cfg.TRAINER.MMRL.REG_WEIGHT
        scale = cfg.TRAINER.MMRL.SCALE
        self.criterion = MMRL_Loss(reg_weight=reg_weight, scale=scale)

        # NOTE: only give representation_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMRL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.image_encoder_clip = nn.DataParallel(self.image_encoder_clip)
        
        
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        # image_representation_cat = shrink_and_block_tile(image)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.MMRL.PREC
        if prec == "amp":
            with autocast():
                with torch.no_grad():
                    
                    image_features_clip = self.image_encoder_clip(image.type(self.dtype))
                    image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)
                    
                    image_cat = shrink_and_block_tile(image)
                    image_cat_features_clip = self.image_encoder_clip(image_cat.type(self.dtype))
                    image_cat_features_clip = image_cat_features_clip / image_cat_features_clip.norm(dim=-1, keepdim=True)

        
                logits, logits_rep, logits_fusion, image_features, text_features, loss1 = model(image)
                text_features = text_features[0:self.num_classes] #Crop the returned text_features for multi-GPU compatibility

                loss = self.criterion(logits, logits_rep, 
                                      image_features, text_features, 
                                      image_features_clip, self.text_features_clip,
                                      label, loss1)
            
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            with torch.no_grad():
                image_features_clip = self.image_encoder_clip(image.type(self.dtype))
                image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)

            logits, logits_rep, logits_fusion, image_features, text_features, loss1 = model(image)
            text_features = text_features[0:self.num_classes] #Crop the returned text_features for multi-GPU compatibility

            loss = self.criterion(logits,logits_rep, 
                                      image_features, text_features, 
                                      image_features_clip, self.text_features_clip,
                                      label, loss1)

            optim.zero_grad()
            loss.backward()
            optim.step()


        output = logits_fusion
        loss_summary = {"loss": loss.item(),
                        'acc': compute_accuracy(output, label)[0].item()
                        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None):
        all_logits = []
        all_labels = []
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        sub_cls = self.cfg.DATASET.SUBSAMPLE_CLASSES
        dataset = self.cfg.DATASET.NAME
        task = self.cfg.TASK

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            logits, logits_rep, logits_fusion, image_features, text_features, loss1= self.model(input)
            # all_logits.append(logits_fusion.detach().cpu())
            # all_labels.append(label.detach().cpu())
            if task == "B2N":
                output = logits_fusion if sub_cls == "base" else logits
            elif task == "FS":
                output = logits
            elif task == "CD":
                output = logits if dataset == "ImageNet" else logits
            else:
                raise ValueError("The TASK must be either B2N, CD, or FS.")

            self.evaluator.process(output, label)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        # Concat collected logits and labels
        # all_logits = torch.cat(all_logits, dim=0)  # [N, C]
        # all_labels = torch.cat(all_labels, dim=0)  # [N]

        # # Apply TSNE
        # tsne = TSNE(n_components=2, random_state=42)
        # reduced = tsne.fit_transform(all_logits.numpy())

        # # 数据集名称
        # dataset_name = self.cfg.DATASET.NAME

        # # 创建保存目录
        # os.makedirs("tsne_plots", exist_ok=True)

        # # 绘图
        # plt.figure(figsize=(8, 6))

        # # 判断类别数量
        # num_classes = len(set(all_labels.numpy()))
        # max_classes_to_show = 50

        # if num_classes > max_classes_to_show:
        #     # 多于50类，仅做散点图，不加 colorbar
        #     scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=all_labels.numpy(), cmap="tab20", s=10)
        #     plt.colorbar(scatter, ticks=range(10), label="Class Index")
        #     plt.title(f"{dataset_name}")
        # else:
        #     scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=all_labels.numpy(), cmap="tab20", s=10)
        #     plt.colorbar(scatter, ticks=range(10), label="Class Index")
        #     plt.title(f"{dataset_name}")

        # plt.tight_layout()
        # save_path = f"tsne_plots/tsne_logits_{dataset_name}.png"
        # plt.savefig(save_path)
        # plt.close()
        # print(f"[TSNE] Saved TSNE plot to {save_path}")

        return list(results.values())[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        # model_file = 'model-best.pth.tar'

        # if epoch is not None:
        #     model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            #model_path = osp.join(directory, name, model_file)
            model_path_prefix = osp.join(directory, name)
            if not osp.exists(model_path_prefix):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path_prefix)
                )
            for file in os.listdir(model_path_prefix):
                if "model-best.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
                    break
                if "model.pth" in file:
                    model_path = osp.join(model_path_prefix, file)
 
            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )            


            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            state_dict = {k: v for k, v in state_dict.items() if "prompt_embeddings" not in k}

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)