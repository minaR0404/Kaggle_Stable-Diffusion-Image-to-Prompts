# Kaggle_Stable-Diffusion-Image-to-Prompts

<br>

https://www.kaggle.com/competitions/stable-diffusion-image-to-prompts

<br>

I would like to thank the organizers and participants of this contest.

43rd place (of 1200 teams) in this competition, I won a medal for the first time.

I would like to record my efforts for two month.

<br>

## About
This competition's theme is **Image to text** .

This is the field of computer vision.

We generate the sentences based on image from Stable Diffusion. 

Compete on the accuracy of the cosine similarity between the generated sentence and the correct sentence.

![phonto](https://github.com/minaR0404/Kaggle_Stable-Diffusion-Image-to-Prompts/assets/49789283/85c6e1f9-5053-4534-9947-d5c8b56cdaa7)

<br>

## Summary
I used two types of models.

**Caption Model :** BLIP, CoCa

**Vision Transformer Model :** CLIP-ViT

I created about 50 models in total, and an ensemble of 8 models achieved the best score.

<br>

## Preprocess
I used image and prompt dataset from DiffusionDB-2M.

Since the amount of data is large(2TB), I preprocessed it by dividing the size into 224 * 224, 320 * 320.., etc. for each model.

There were data augmentation methods such as creating a large amount of images ourselves and using Chat-GPT to create large amounts of text, but due to computational resources, I was not able to focus on that.

<br>

## Caption
A caption is a descriptive text for an image, and this model generates the caption using a trained model.

For caption models, trained model is available on Huggingface, Open-AI.

The following two models are compatible with ViT.

<br>

**BLIP :** Chose a medium model rather than a large model. Due to training and execution time issue.

**CoCa :** Fine tuning coca-ViT, but wasn't as effective as clip.

<br>

BLIP's score : 0.418

CoCa's score : 0.449

<br>

## ViT
ViT(Vision Transformer) is a model that applies the Transformer used in NLP(language) models to images.

These model are highly accurate, and the main model at this time.

More than hundreds of ViT pre-trained models are provided by multiple organizations such as Open-Ai, Facebook.

Among them, I would like to mention two models that have made a major contribution.

<br>

**CLIP-ViT-L-14 :** This is an orthodox type, Large model. Used Open-AI ver.

**CLIP-ViT-B-16 :** Training a model takes a lot of time and computational resources. The base size model was cost effective.

<br>

The settings of the neural network during model training made a large difference in model accuracy.

Below is the difference in prediction accuracy using different learning methods for the same model.

<br>

Timm(library)'s score : 0.519

My own NN's score : 0.551

<br>

## Ensemble
The ensemble provided a huge boost to the score.

Although similar models were not effective, but I obtained a large gain by merging the caption model and the ViT model.

<br>

The ratio of each model is as follows.

BLIP          : 10%

CoCa          : 15%

ViT(6 models) : 75%

Score : 0.607

<br>

## Not worked
These method were not worked for me.

- Other caption models (ex.BLIP2, GiT, DALL-E)
- Other ViT models (ex.ViT-H-14,ViT-G-14)
- Huge models(ex.BLIP2, ViT-H-14), the training epochs took too long(15~20h/epoch), so we had to reduce the model size.

<br>

