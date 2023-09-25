#################### Setup ####################

wheels_path = "/kaggle/input/clip-interrogator-0-5-4"
clip_interrogator_whl_path = f"{wheels_path}/clip_interrogator-0.5.4-py3-none-any.whl"
!pip install --no-index --find-links $wheels_path  $clip_interrogator_whl_path -q 
!pip install --no-index --no-deps /kaggle/input/lavis-pretrained/salesforce-lavis/transformers* 
!cp -r /kaggle/input/transformers-master/src/transformers/generation/ /opt/conda/lib/python3.7/site-packages/transformers/


#################### Sentence Transformer setup ####################

import torch
import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
sys.path.append('../input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models

comp_path = Path('../input/stable-diffusion-image-to-prompts/')
sample_submission = pd.read_csv(comp_path / 'sample_submission.csv', index_col='imgId_eId')
images = os.listdir(comp_path/"images")
image_ids = [i.split('.')[0] for i in images]

EMBEDDING_LENGTH = 384
eIds = list(range(EMBEDDING_LENGTH))

imgId_eId = [
    '_'.join(map(str, i)) for i in zip(
        np.repeat(image_ids, EMBEDDING_LENGTH),
        np.tile(range(EMBEDDING_LENGTH), len(image_ids)))]
# load sentence transformer model
st_model = SentenceTransformer('/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2')


#################### CLIP Interrogater setup ####################

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
def make_batches(l, batch_size=16):
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import inspect
import importlib

# replace tokenizer path to prevent downloading
blip_path = "/opt/conda/lib/python3.7/site-packages/blip/models/blip.py"

fin = open(blip_path, "rt")
data = fin.read()
data = data.replace(
    "BertTokenizer.from_pretrained('bert-base-uncased')", 
    "BertTokenizer.from_pretrained('/kaggle/input/clip-interrogator-models-x/bert-base-uncased')"
)

fin.close()

fin = open(blip_path, "wt")
fin.write(data)
fin.close()

from clip_interrogator import clip_interrogator
# fix clip_interrogator bug
clip_interrogator_path = inspect.getfile(clip_interrogator.Interrogator)

fin = open(clip_interrogator_path, "rt")
data = fin.read()
data = data.replace(
    'open_clip.get_tokenizer(clip_model_name)', 
    'open_clip.get_tokenizer(config.clip_model_name.split("/", 2)[0])'
)
fin.close()

fin = open(clip_interrogator_path, "wt")
fin.write(data)
fin.close()

importlib.reload(clip_interrogator)


#################### CLIP ans BLIP Config ####################

from blip.models import blip
import os
import sys
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt 


import pandas as pd
import torch
import open_clip

comp_path = Path('/kaggle/input/stable-diffusion-image-to-prompts/')
class CFG:
    device = "cuda"
    seed = 42
    embedding_length = 384
    sentence_model_path = "/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2"
    blip_model_path = "/kaggle/input/clip-interrogator-models-x/model_large_caption.pth"
    ci_clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
    clip_model_name = "ViT-H-14"
    clip_model_path = "/kaggle/input/clip-interrogator-models-x/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
    cache_path = "/kaggle/input/vlomme-clip-iter"


model_config = clip_interrogator.Config(clip_model_name=CFG.ci_clip_model_name)
model_config.cache_path = CFG.cache_path
model_config.blip_num_beams = 1
configs_path = os.path.join(os.path.dirname(os.path.dirname(blip_path)), 'configs')
med_config = os.path.join(configs_path, 'med_config.json')
blip_model = blip.blip_decoder(
    pretrained=CFG.blip_model_path,
    image_size=model_config.blip_image_eval_size, 
    vit=model_config.blip_model_type, 
    med_config=med_config
)
blip_model.eval()
blip_model = blip_model.to(model_config.device)
model_config.blip_model = blip_model
clip_model = open_clip.create_model(CFG.clip_model_name, precision='fp16' if model_config.device == 'cuda' else 'fp32')
open_clip.load_checkpoint(clip_model, CFG.clip_model_path)
clip_model.to(model_config.device).eval()
model_config.clip_model = clip_model
clip_preprocess = open_clip.image_transform(
    clip_model.visual.image_size,
    is_train = False,
    mean = getattr(clip_model.visual, 'image_mean', None),
    std = getattr(clip_model.visual, 'image_std', None),
)
model_config.clip_preprocess = clip_preprocess
ci = clip_interrogator.Interrogator(model_config)
cos = torch.nn.CosineSimilarity(dim=1)

mediums_features_array = torch.stack([torch.from_numpy(t) for t in ci.mediums.embeds]).to(ci.device)
movements_features_array = torch.stack([torch.from_numpy(t) for t in ci.movements.embeds]).to(ci.device)
flavors_features_array = torch.stack([torch.from_numpy(t) for t in ci.flavors.embeds]).to(ci.device)


#################### Inference (CLIP) ####################

BATCH_SIZE = 32
clip_text = []
submissions00 = []
for batch in make_batches(images, BATCH_SIZE):
    images_batch = []
    for i, image in enumerate(batch):
        images_batch.append(clip_preprocess(Image.open(comp_path/"images"/image).convert("RGB")).unsqueeze(0))
    images_batch = torch.cat(images_batch,0).to(device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = clip_model.encode_image(images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    for i in range(len(image_features)):
        medium = [ci.mediums.labels[i] for i in cos(image_features[i], mediums_features_array).topk(1).indices][0]
        movement = [ci.movements.labels[i] for i in cos(image_features[i], movements_features_array).topk(1).indices][0]
        flaves = ", ".join([ci.flavors.labels[i] for i in cos(image_features[i], flavors_features_array).topk(3).indices])
        prompt = f", {medium}, {movement}, {flaves}" + ', fine details, masterpiece'
        clip_text.append(prompt)
        prompt = clip_interrogator._truncate_to_fit(prompt, ci.tokenize)
        submissions00.append(prompt)

submissions00 = st_model.encode(submissions00).flatten()
submissions01 = st_model.encode(clip_text).flatten()


#################### Inference (BLIP) ####################

!pip install --no-index --no-deps /kaggle/input/lavis-pretrained/salesforce-lavis/hugging*

processor = AutoProcessor.from_pretrained("/kaggle/input/blip-pretrained-model/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("/kaggle/input/blip-pretrained-model/blip-image-captioning-large")
model.to(device);
BATCH_SIZE = 16

submissions1 = []
for ix,batch in enumerate(make_batches(images, BATCH_SIZE)):
    images_batch = []
    for i, image in enumerate(batch):
        images_batch.append(Image.open(comp_path/"images"/image).convert("RGB"))

    pixel_values = processor(images=images_batch, return_tensors="pt").pixel_values.to(device)
    out = model.generate(pixel_values=pixel_values, max_length=20, num_return_sequences=5,
                         num_beams=5, min_length=5)
    prompts = processor.batch_decode(out, skip_special_tokens=True)
    for i in range(len(images_batch)):
        for j in range(5):
            caption = prompts[i*5+j] 
            submissions1.append(caption)

submissions1 = st_model.encode(submissions1,show_progress_bar=False)
submissions1 = np.reshape(submissions1, (-1,5,384)).mean(1).flatten()


#################### Inference (CoCa) ####################

wheels_path = "/kaggle/input/open-clip-wheels/open_clip_wheels"
open_clip_whl_path = f"{wheels_path}/open_clip_torch-2.14.0-py3-none-any.whl"
!pip install --no-index --find-links $wheels_path $open_clip_whl_path -q

comp_path = Path('/kaggle/input/stable-diffusion-image-to-prompts/')

class CFG:
    device = "cuda"
    seed = 42
    embedding_length = 384
    model_name = "coca_ViT-L-14"
    model_checkpoint_path = '/kaggle/input/coca-finetuning/logs/2023_04_05-10_06_28-model_coca_ViT-L-14-lr_1e-05-b_48-j_2-p_amp/checkpoints/epoch_3.pt'
model = open_clip.create_model(CFG.model_name)
model.to(device)
open_clip.load_checkpoint(model, CFG.model_checkpoint_path)
transform = open_clip.image_transform(
    model.visual.image_size,
    is_train = False,
    mean = getattr(model.visual, 'image_mean', None),
    std = getattr(model.visual, 'image_std', None),
)
BATCH_SIZE = 16 
submissions2 = []
for ix,batch in enumerate(make_batches(images, BATCH_SIZE)):
    images_batch = []
    for i, image in enumerate(batch):
        images_batch.append(transform(Image.open(comp_path/"images"/image).convert("RGB")).unsqueeze(0))

    images_batch = torch.cat(images_batch,0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(images_batch)
        #generated = model.generate(images_batch, beam_search_kwargs={"process_beam_indices": None})
        
    for i in range(len(images_batch)):
        caption = open_clip.decode(generated[i]).split("<end_of_text>")[0].replace("<start_of_text>", "").rstrip(" .,")
        submissions2.append(caption)

submissions2 = st_model.encode(submissions2).flatten()


#################### Inference (ViT) ####################

from torch import nn
from transformers import AutoModel

class Net_laion(nn.Module):
    def __init__(self):
        super(Net_laion, self).__init__()
        
        clip = open_clip.create_model(CFG.model_name)
        
        self.positional_embedding = nn.Parameter(clip.positional_embedding)
        self.text_projection = nn.Parameter(clip.text_projection)
        self.logit_scale = nn.Parameter(clip.logit_scale)
        self.visual = clip.visual  # self.vision = clip.vision_model
        self.transformer = clip.transformer
        self.token_embedding = clip.token_embedding
        self.ln_final = clip.ln_final
        
        if CFG.model_name == 'ViT-H-14':
            self.fc = nn.Linear(1024, 384)  # (1280, 384)
        elif CFG.model_name == 'ViT-L-14':
            self.fc = nn.Linear(768, 384)
        elif CFG.model_name == 'ViT-B-32':
            self.fc = nn.Linear(512, 384)

    def forward(self, x):
        out = self.visual(x)
        return self.fc(out)
    
    
class Net_Erii(nn.Module):
    def __init__(self):
        super(Net_Erii, self).__init__()
        
        clip = open_clip.create_model(CFG.model_name)
        
        self.model = clip
        if CFG.model_name == 'ViT-L-14':
            self.dense0 = nn.Linear(768, 384)
        elif CFG.model_name == 'ViT-B-16':
            self.dense0 = nn.Linear(512, 384)

    def forward(self, x):
        out = self.model.visual(x)
        return self.dense0(out)


class Net_OpenAI(nn.Module):
    def __init__(self):
        super(Net_OpenAI, self).__init__()
        
        clip = AutoModel.from_pretrained(CFG.base_model_path ,local_files_only=True)
        
        self.visual = clip.vision_model  # self.vision = clip.vision_model
        if CFG.model_name == 'ViT-L-14':
            self.fc = nn.Linear(1024, 384)  # (1280, 384)
        elif CFG.model_name == 'ViT-B-16':
            self.fc = nn.Linear(768, 384)

    def forward(self, x):
        out = self.visual(x)['pooler_output']
        return self.fc(out)


class IMGDataset:
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = self.transform(image)
        return image
    
    
def inference(images, batch_size):
    
    if CFG.model_type == 'laion':
        nn_model = Net_laion()
    if CFG.model_type == 'OpenAI':
        nn_model = Net_OpenAI()
    if CFG.model_type == 'Erii':
        nn_model = Net_Erii()
        
    nn_model.load_state_dict(torch.load(CFG.trained_model_path))
    nn_model.to(device)
    nn_model.eval()
    
    input_size = 224
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataloader = DataLoader(dataset=IMGDataset(images, transform),
                                 batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    
    
    preds = []
    for batch_images in tqdm(test_dataloader):
        batch_images = batch_images.to(device)
            
        with torch.no_grad():
            X_out = nn_model(batch_images).cpu().numpy()
            # L2 normalize -- Start
            X_out = X_out / ( np.abs(X_out).max(axis=-1, keepdims=True) + 0.0000001)  # To avoid to overflow at normalize()
            X_out = normalize( X_out )
            # L2 normalize -- End
            preds.append(X_out)
            
    prediction = np.vstack(preds).flatten()
    
    return prediction


class CFG:
    model_type = 'OpenAI'
    model_name = 'ViT-L-14'
    base_model_path = '/kaggle/input/openai-clip/'
    trained_model_path = '/kaggle/input/clip-vit-h-14-finetuned/ViT-L-14_OpenAI_f18_2epoch.pt'
    batch_size = 64
    
submissions3 = inference(images, CFG.batch_size)

class CFG:
    model_type = 'OpenAI'
    model_name = 'ViT-B-16'
    base_model_path = '/kaggle/input/openai-clip-vit-b-16/'
    trained_model_path = '/kaggle/input/clip-vit-h-14-finetuned/ViT-B-16_OpenAI_f10_3epoch.pt'
    batch_size = 64
    
submissions4 = inference(images, CFG.batch_size)

class CFG:
    model_type = 'Erii'
    model_name = 'ViT-B-16'
    trained_model_path = '/kaggle/input/vit-b-16-laion2b-s34b-b88k-53/ViT-B-16_laion2b_s34b_b88k_53.pth'
    batch_size = 64
    
submissions5 = inference(images, CFG.batch_size)

class CFG:
    model_type = 'Erii'
    model_name = 'ViT-B-16'
    trained_model_path = '/kaggle/input/vit-b-16-laion2b-s34b-b88k-54/ViT-B-16_laion2b_s34b_b88k_54.pth'
    batch_size = 64
    
submissions6 = inference(images, CFG.batch_size)


#################### Ensemble ####################

import torch.nn.functional as F

def normalize(embeds):
    embeds = embeds.reshape(-1, 384)
    return (embeds / np.linalg.norm(embeds, ord=2, axis=1, keepdims=True)).reshape(-1)

ratio_1 = 0.15
ratio_2 = 0.15
ratio_3 = 0.15
ratio_4 = 0.15
ratio_5 = 0.20
ratio_6 = 0.20

submissions = (
               ratio_1 * (normalize(submissions1)) +
               ratio_2 * (normalize(submissions2)) + 
               ratio_3 * (normalize(submissions3)) +
               ratio_4 * (normalize(submissions4)) +
               ratio_5 * (normalize(submissions5)) +
               ratio_6 * (normalize(submissions6))
               )


submission = pd.DataFrame({"imgId_eId":imgId_eId, "val": submissions})
submission.to_csv("submission.csv", index=False)
submission.head()


#################### CV ####################

if len(submissions)<3000:
    images = os.listdir(comp_path / 'images')
    imgIds = [i.split('.')[0] for i in 
              images]
    EMBEDDING_LENGTH = 384
    eIds = list(range(EMBEDDING_LENGTH))

    imgId_eId = [
        '_'.join(map(str, i)) for i in zip(
            np.repeat(imgIds, EMBEDDING_LENGTH),
            np.tile(range(EMBEDDING_LENGTH), len(imgIds)))]

    assert sorted(imgId_eId) == sorted(submission.imgId_eId)
    ground_truth = pd.read_csv('/kaggle/input/stable-diffusion-image-to-prompts/prompts.csv')
    ground_truth = pd.merge(pd.DataFrame(imgIds,columns=['imgId']),ground_truth,on='imgId',how='left')
    st_model = SentenceTransformer('/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2')

    ground_truth_embeddings = st_model.encode(ground_truth.prompt).flatten()

    gte = pd.DataFrame(
                    index=imgId_eId,
                    data=ground_truth_embeddings,
                    columns=['val']).rename_axis('imgId_eId')

    cv_prompts = pd.read_csv('/kaggle/working/submission.csv') # example: LB 0.418 submission
    from scipy import spatial
    vec1 = gte['val']
    vec2 = cv_prompts['val']
    cos_sim = 1 - spatial.distance.cosine(vec1, vec2)
    print(cos_sim)
