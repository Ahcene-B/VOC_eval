# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random
import pickle

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image

from networks import get_model
from datasets import ImageDataset, Dataset, bbox_iou
from visualizations import visualize_fms, visualize_predictions, visualize_seed_expansion
from object_discovery import lost, detect_box, dino_seg

######################################################

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

if CUDA:
    from fast_pytorch_kmeans import KMeans
    from torch_pca import PCA
else:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

import cv2

from skimage import measure
from skimage.segmentation import mark_boundaries

from scipy.ndimage import binary_dilation

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


######################################################
# These functions should return a list of tensors of shape:
# (BSIZE, CHANNELS, UPSCALED_IMG_WIDTH * UPSCALED_IMG_HEIGHT)

def get_PATH_RESNET(image, model, layers):
    PATH = []
    S = []
    size = [-1,-1]
    with torch.inference_mode():
        z = image.clone()
        z = model.features[0](z)
        z = model.features[1](z)
        z = model.features[2](z)
        z = model.features[3](z)

        if 0 in layers:
            if size[0] == -1:
                size = [a for a in list(z.shape[-2:])]
            S.append( z.shape )
            u = nn.Upsample(size=size, mode='bicubic')(z)
            PATH.append(u.flatten(-2).detach())
        #for il,ly in enumerate([model.layer1,model.layer2,model.layer3,model.layer4]):
        for il in range(4,8):
            z = model.features[il](z)
            if il-3 in layers:
                if size[0] == -1:
                    size = [a for a in list(z.shape[-2:])]
                S.append( z.shape )
                u = nn.Upsample(size=size, mode='bicubic')(z)
                PATH.append(u.flatten(-2).detach())
    #print(S)
    return PATH,S

def get_PATH_VGG(image, model, layers):
    PATH = []
    S = []
    size = [-1,-1]
    layers = np.array( [2,7,14,21,28] )[layers]
    with torch.inference_mode():
        z = image.clone()
        for il,ly in enumerate(model.features):
            z = ly(z)
            if il in layers:
                if size[0] == -1:
                    size = [a for a in list(z.shape[-2:])]
                S.append( z.shape )
                u = nn.Upsample(size=size, mode='bicubic')(z)
                PATH.append(u.flatten(-2).detach())
    #print(S)
    return PATH,S

def get_PATH_CLIP(image, model, layers):
    PATH = []
    S = []
    size = [-1,-1]
    with torch.inference_mode():
        z = image.clone()
        z = z.type(model.conv1.weight.dtype)

        z = model.relu1(model.bn1(model.conv1(z)))
        z = model.relu2(model.bn2(model.conv2(z)))
        z = model.relu3(model.bn3(model.conv3(z)))
        z = model.avgpool(z)

        if 0 in layers:
            S.append( z.shape )
            if size[0] == -1:
                size = [a for a in list(z.shape[-2:])]
            u = nn.Upsample(size=size, mode='bicubic')(z)
            PATH.append(u.flatten(-2).detach())

        for il,ly in enumerate([model.layer1,model.layer2,model.layer3,model.layer4]):
            z = ly(z)
            if il+1 in layers:
                S.append( z.shape )
                if size[0] == -1:
                    size = [a for a in list(z.shape[-2:])]
                u = nn.Upsample(size=size, mode='bicubic')(z)
                PATH.append(u.flatten(-2).detach())
    #print(S)
    return PATH,S


def get_PATH_VIT(image, model, layers):
    _,_,h,w = image.shape
    size = [-1,-1]
    PATH = []
    S = []

    #image = nn.Upsample(size=(224,224), mode='bicubic')(image)

    z = model.prepare_tokens(image)

    pd = z.shape[-1]
    pw = image.shape[-2] // args.patch_size
    ph = image.shape[-1] // args.patch_size

    for ib,blk in enumerate(model.blocks):
        z = blk(z)
        if ib in layers:
#            if size[0] == -1:
#                size = [a for a in list(z.shape[-2:])]
            u = z[:,1:]
            u = torch.permute(u,(0,2,1))
#            print('xx', u.shape)
            PATH.append(u.detach())
            u = u.reshape((-1,pd,pw,ph))
            S.append( u.shape )
#            u = nn.Upsample(size=size, mode='bicubic')(u)
#            PATH.append(u.flatten(-2).detach())

    #print(S)
    return PATH,S


def get_PATH_DINO(image, model, layers):
    PATH = []
    S = []
    size = [-1,-1]
    if size[0] == -1:
        num_patches_h = image.shape[2] // model.patch_size
        num_patches_w = image.shape[2] // model.patch_size
        size = [num_patches_h * 2, num_patches_w * 2]

    upsampler = nn.Upsample(size=size, mode='bicubic')
    with torch.inference_mode():
        # This returns [(1, EMB_DIM, NUM_PATCHES_H, NUM_PATCHES_W) ... ]
        tokens = model.get_intermediate_layers(image, layers, reshape=True)
        S = [t.shape for t in tokens]

        # Now upsample into [(1, EMB_DIM, size[0], size[1]) ... ]
        tokens = [upsampler(t) for t in tokens]

        # End with flattening
        PATH = [t.flatten(-2).detach() for t in tokens]
    #print(S)
    return PATH,S


def get_PATH_DINO_ATTN(image, model, layers):
    # Inspired by https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L216
    # ~/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py
    # ~/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py

    PATH = []
    S = []
    size = [-1,-1]
    num_patches_h = image.shape[2] // model.patch_size
    num_patches_w = image.shape[3] // model.patch_size
    if size[0] == -1:
        size = [num_patches_h * 2, num_patches_w * 2]

    upsampler = nn.Upsample(size=size, mode='bicubic')
    with torch.inference_mode():
        # This returns [(1, EMB_DIM, NUM_PATCHES_H, NUM_PATCHES_W) ... ]
        tokens = model.prepare_tokens_with_masks(image)
        for bk in model.blocks[:-1]:
            tokens = bk(tokens)
        tokens = model.blocks[-1].norm1(tokens)

        attention = model.blocks[-1].attn
        B, N, C = tokens.shape
        H = attention.num_heads

        qkv = attention.qkv(tokens).reshape(B, N, 3, H, C // H)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * attention.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attention.attn_drop(attn)

        attn = attn[:,:,0,1 + model.num_register_tokens:]
        attn = attn.reshape((B,attention.num_heads,num_patches_h,num_patches_w))
        attn = attn[:,layers,:]
        S = [attn.shape]

        attn = upsampler(attn)

        PATH = [attn.flatten(-2)]
    #print(S)
    return PATH,S


######################################################


def get_NAVE(
        image,
        model,
        get_path,
        kept_layers=[2,3,4],
        nb_clusters=5,
        method='km',
        ):

    n_images = len(image)

    PATH,S = get_path(image,model,kept_layers)
#    print(0,S)
#    print(1,[a.shape for a in PATH])
    size = [a for a in list(S[0])[-2:]]
#    print(2,size)

    WGT = np.array([a.shape[1] for a in PATH])
    WGT = WGT[0]/(1+WGT)
    WGT = WGT/WGT[0]

    X = [a / torch.sqrt( torch.sum( a**2,1))[:,None] for a in PATH]
    X = [ X[i] * WGT[i] for i in range(len(X))]
    X = torch.cat(X,1)
    X = torch.permute(X,(1,2,0))
    X = X.flatten(1).T
#    print(3,X.shape)

    if method == 'km':
        if CUDA:
            proj    = KMeans(n_clusters=nb_clusters, init_method='kmeans++', max_iter=300, tol=1e-5)
            segmt = proj.fit_predict( X ).reshape( list(size)+[n_images] )
            segmt = segmt.detach().cpu().numpy()
        else:
            proj    = KMeans(n_clusters=nb_clusters)
            segmt = proj.fit_predict( X ).reshape( list(size)+[n_images] )

    elif method == 'pca':
        proj  = PCA(n_components=nb_clusters)
        segmt = proj.fit_transform( X )
        if CUDA:
            segmt = segmt.detach().cpu().numpy()
        segmt = np.abs(segmt).argmax(-1)
        segmt = segmt.reshape( list(size)+[n_images] )

    elif method in 'attn':
        segmt = X.argmax(-1).detach().cpu().numpy()
        segmt = segmt.reshape( list(size)+[n_images] )

    elif method == 'agg':
        proj = AgglomerativeClustering(nb_clusters)
        proj.fit( X.detach().cpu().numpy() )
        segmt = proj.labels_
        segmt = segmt.reshape( list(size)+[n_images] )

    else:
        raise NotImplementedError

    sppx = np.moveaxis(segmt,2,0)
#    print(4,segmt.shape, sppx.shape, image.shape)

    _,_,h,w = image.shape
    if sppx.shape[-2:] != (h,w):
        sppx = np.array([cv2.resize( s, (w,h), interpolation=cv2.INTER_NEAREST) for s in sppx])

    return sppx


def get_MASK(sppx):
    h,w   = sppx.shape
    idcc  = np.unique(sppx.flatten())
    masks = -np.ones((len(idcc),h,w))
    for ik,k in enumerate(idcc):
        masks[ik,:,:] = (sppx==k).astype(float)

    return masks

def get_BORD(sppx):
    masks = get_MASK(sppx)
    for k in range(len(masks)):
        masks[k] = binary_dilation(masks[k],iterations=1) - masks[k]
    bord = (masks.sum(0) != 0).astype(float)
    return bord

def get_CC(sppx):
    msk = get_MASK(sppx)
    cc = np.zeros((0,*list(msk[0].shape)))
    for m in msk:
        mc = measure.label(m,connectivity=1)
        mc = np.eye( mc.max()+1)[mc]
        mc = np.moveaxis(mc,2,0)
        mc = mc[1:]
        mc = mc[ mc.sum((-1,-2)) > 14*14*4 ]
        cc = np.concatenate([cc,mc],0)
    cc[:,:,:14] = 0
    cc[:,:,-14:] = 0
    cc[:,:14,:] = 0
    cc[:,-14:,:] = 0

    cc = cc[cc.sum((1,2))>0]

    return cc



######################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unsupervised object discovery with NAVE.")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "resnet50",
            "vgg16_imagenet",
            "resnet50_imagenet",
        ],
        help="Model architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )

    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--set",
        default="train",
        type=str,
        choices=["val", "train", "trainval", "test","person_val","person_train"],
        help="Path of the image to load.",
    )
    # Or use a single image
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="If want to apply only on one image, give file path.",
    )

    # Folder used to output visualizations and
    parser.add_argument(
        "--output_dir", type=str, default="outputs_nave", help="Output directory to store predictions and visualizations."
    )

    # Evaluation setup
    parser.add_argument("--no_hard", action="store_true", help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["fms", "seed_expansion", "pred", None],
        default=None,
        help="Select the different type of visualizations.",
    )

    # For ResNet dilation
    parser.add_argument("--resnet_dilate", type=int, default=2, help="Dilation level of the resnet model.")

    # Use dino-seg proposed method
    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)


    # For NAVE
    parser.add_argument("-L","--layers", nargs='+', type=int, default=[2, 3, 4],
                    help="A combination of desired layer activations to be used")
    parser.add_argument('-K','--nb_clusters', type=int, default=5,
                    help="Number of clusters used to compute the segmentation.")
    parser.add_argument('-P','--projector', type=str, default='km',
                    help="Projection method: Kmeans (km) or PCA (pca).")

    # Custom
    parser.add_argument("--split_boxes", action="store_true", help="IoU splits the boxes.")

    args = parser.parse_args()

    if args.image_path is not None:
        args.save_predictions = False
        args.no_evaluation = True
        args.dataset = None

    if type(args.layers) == int:
        args.layers = [args.layers]

    if 'attn+' in args.projector:
        get_PATH_DINO = get_PATH_DINO_ATTN
        args.projector = args.projector[5:]

    # -------------------------------------------------------------------------------------------------------
    # Dataset

    # If an image_path is given, apply the method only to the image
    if args.image_path is not None:
        dataset = ImageDataset(args.image_path)
    else:
        dataset = Dataset(args.dataset, args.set, args.no_hard)

    # -------------------------------------------------------------------------------------------------------
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args.arch, args.patch_size, args.resnet_dilate, device)

    if 'resnet' in args.arch:
        get_PATH = get_PATH_RESNET
    elif 'vit' in args.arch:
        get_PATH = get_PATH_VIT
    elif 'vgg' in args.arch:
        get_PATH = get_PATH_VGG
    # -------------------------------------------------------------------------------------------------------
    # Directories
    if args.image_path is None:
        args.output_dir = os.path.join(args.output_dir, dataset.name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    if args.dinoseg:
        # Experiment with the baseline DINO-seg
        if "vit" not in args.arch:
            raise ValueError("DINO-seg can only be applied to tranformer networks.")
        exp_name = f"{args.arch}-{args.patch_size}_dinoseg-head{args.dinoseg_head}"
    else:
        # Experiment with NAVE
        exp_name = f"NAVE-{args.arch}"
        if "resnet" in args.arch:
            exp_name += f"dilate{args.resnet_dilate}"
        elif "vit" in args.arch:
            exp_name += f"{args.patch_size}"

    print(f"Running NAVE on the dataset {dataset.name} (exp: {exp_name})")
    print(f"Projector: {args.projector}, Layers: {args.layers}, Nb Clusters {args.nb_clusters}.")

    # Visualization
    if args.visualize:
        vis_folder = f"{args.output_dir}/visualizations/{exp_name}"
        os.makedirs(vis_folder, exist_ok=True)

    # -------------------------------------------------------------------------------------------------------
    # Loop over images
    preds_dict = {}
    cnt = 0
    corloc = np.zeros(len(dataset.dataloader))

    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):

        # ------------ IMAGE PROCESSING -------------------------------------------
        img = inp[0]
        init_image_size = img.shape

        # Get the name of the image
        im_name = dataset.get_image_name(inp[1])

        # Pass in case of no gt boxes in the image
        if im_name is None:
            continue

        # Padding the image with zeros to fit multiple of patch-size
        size_im = (
            img.shape[0],
            int(np.ceil(img.shape[1] / args.patch_size) * args.patch_size),
            int(np.ceil(img.shape[2] / args.patch_size) * args.patch_size),
        )
        paded = torch.zeros(size_im)
        paded[:, : img.shape[1], : img.shape[2]] = img
        img = paded

        # Move to gpu
        if CUDA:
            img = img.cuda(non_blocking=True)
        # Size for transformers
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        # ------------ GROUND-TRUTH -------------------------------------------
        if not args.no_evaluation:
            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)

            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

        # ------------ EXTRACT FEATURES -------------------------------------------
        with torch.no_grad():
            sppx = get_NAVE( img[None,:],
                                        model,
                                        get_PATH,
                                        kept_layers=args.layers,
                                        nb_clusters=args.nb_clusters,
                                        method=args.projector)

            # Split sppx per connected component
            bords = get_BORD(sppx[0])
            feats = get_CC(sppx[0])

        # Evaluation
        if args.no_evaluation:
            continue
        elif args.split_boxes:
            # Compare prediction to GT boxes
            all_ious = []
            preds = []
            for g_bbx in gt_bbxs:
                t_gt = torch.from_numpy(g_bbx[None,:])
                ious = []
                for cc in feats:
                    nnz = (cc>0).nonzero()
                    ymin,ymax = nnz[0].min(),nnz[0].max()
                    xmin,xmax = nnz[1].min(),nnz[1].max()
                    box = np.array([xmin,ymin,xmax,ymax])
                    iu = bbox_iou(torch.from_numpy(box), t_gt)
                    ious.append( [_iu[0].item() for _iu in iu] )

                all_ious.append( ious.max() )
                ic = ious.argmax()
                cc = feats[ic]
                nnz = (cc>0).nonzero()
                ymin,ymax = nnz[0].min(),nnz[0].max()
                xmin,xmax = nnz[1].min(),nnz[1].max()
                preds.append( np.array([xmin,ymin,xmax,ymax]) )


            # Save the prediction
            all_ious = np.array(all_ious)
            corloc[ im_id ] = (all_ious > .5).mean()
            preds_dict[im_name] = preds

            # ------------ Visualizations -------------------------------------------
            if args.visualize == "pred":
                image = dataset.load_image(im_name)

                h,w,_ = image.shape
                _,hh,ww = img.shape
                ph = (hh-h)//2
                pw = (ww-w)//2

                image = cv2.copyMakeBorder(image, ph, hh-h-ph, pw, ww-w-pw,
                            cv2.BORDER_CONSTANT,value=[0,0,0])

                msk = (cc[:,:,None]>0).astype(float)
                msk = (msk+.5)/2
                image = image * msk

                image = image*(1-bords[:,:,None])
                image[:,:,:2] = image[:,:,:2] + bords[:,:,None]*255

                image = image.astype(np.uint8)

                for bx in gt_bbxs:
                    cv2.rectangle(
                        image,
                        (int(bx[0]), int(bx[1])),
                        (int(bx[2]), int(bx[3])),
                        (0, 255, 0), 1,
                    )

                for pred in preds:
                    cv2.rectangle(
                        image,
                        (int(pred[0]), int(pred[1])),
                        (int(pred[2]), int(pred[3])),
                        (255, 0, 0), 1,
                    )

                pltname = f"{vis_folder}/NAVE_{im_name}.png"
                Image.fromarray(image).save(pltname)
            else:
                raise NotImplementedError("Only pred visualization.")

        else:
            # Compare prediction to GT boxes
            t_gt = torch.from_numpy(gt_bbxs)
            ious = []
            for cc in feats:
                nnz = (cc>0).nonzero()
                ymin,ymax = nnz[0].min(),nnz[0].max()
                xmin,xmax = nnz[1].min(),nnz[1].max()
                box = np.array([xmin,ymin,xmax,ymax])
                iu = bbox_iou(torch.from_numpy(box), t_gt, split=True)
                ious.append( [_iu[0].item() for _iu in iu] )

            ious = np.array(ious)
            ic = (ious[:,0]/ious[:,1]).argmax()
            if ious[ic,0]/ious[ic,1] >= 0.5:
                corloc[im_id] = 1

            # Save the prediction
            cc = feats[ic]
            nnz = (cc>0).nonzero()
            ymin,ymax = nnz[0].min(),nnz[0].max()
            xmin,xmax = nnz[1].min(),nnz[1].max()
            pred = np.array([xmin,ymin,xmax,ymax])

            preds_dict[im_name] = pred

            # ------------ Visualizations -------------------------------------------
            if args.visualize == "pred":
                image = dataset.load_image(im_name)

                h,w,_ = image.shape
                _,hh,ww = img.shape
                ph = (hh-h)//2
                pw = (ww-w)//2

                image = cv2.copyMakeBorder(image, ph, hh-h-ph, pw, ww-w-pw,
                            cv2.BORDER_CONSTANT,value=[0,0,0])

                msk = (cc[:,:,None]>0).astype(float)
                msk = (msk+.5)/2
                image = image * msk

                image = image*(1-bords[:,:,None])
                image[:,:,:2] = image[:,:,:2] + bords[:,:,None]*255

                image = image.astype(np.uint8)

                for bx in gt_bbxs:
                    cv2.rectangle(
                        image,
                        (int(bx[0]), int(bx[1])),
                        (int(bx[2]), int(bx[3])),
                        (0, 255, 0), 1,
                    )

                cv2.rectangle(
                    image,
                    (int(pred[0]), int(pred[1])),
                    (int(pred[2]), int(pred[3])),
                    (255, 0, 0), 1,
                )


                pltname = f"{vis_folder}/NAVE_{im_name}.png"
                Image.fromarray(image).save(pltname)
            else:
                raise NotImplementedError("Only pred visualization.")

            cnt += 1
            if cnt % 50 == 0:
                pbar.set_description("Found {:.1f}@{:d}".format(100*np.sum(corloc)/cnt,cnt))


    # Save predicted bounding boxes
    if args.save_predictions:
        folder = f"{args.output_dir}/{exp_name}"
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, "preds.pkl")
        with open(filename, "wb") as f:
            pickle.dump(preds_dict, f)
        print("Predictions saved at %s" % filename)

    # Evaluate
    if not args.no_evaluation:
        print(f"corloc: {100*np.sum(corloc)/cnt:.2f} ({int(np.sum(corloc))}/{cnt})")
        result_file = os.path.join(folder, 'results.txt')
        with open(result_file, 'w') as f:
            f.write('corloc,%.1f,,\n'%(100*np.sum(corloc)/cnt))
        print('File saved at %s'%result_file)
