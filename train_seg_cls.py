####################
# Import Libraries
####################
import os
import sys
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader

#from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
import albumentations as A
import timm
from omegaconf import OmegaConf

from sklearn.metrics import roc_auc_score
####################
# Utils
####################

def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:,i], y_pred[:,i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores

import random
import gc
from scipy.sparse import coo_matrix
import time
def multi_label_stratified_group_k_fold(label_arr: np.array, gid_arr: np.array, n_fold: int, seed: int=42):
    """
    create multi-label stratified group kfold indexs.
    reference: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    input:
        label_arr: numpy.ndarray, shape = (n_train, n_class)
            multi-label for each sample's index using multi-hot vectors
        gid_arr: numpy.array, shape = (n_train,)
            group id for each sample's index
        n_fold: int. number of fold.
        seed: random seed.
    output:
        yield indexs array list for each fold's train and validation.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()
    n_train, n_class = label_arr.shape
    gid_unique = sorted(set(gid_arr))
    n_group = len(gid_unique)

    # # aid_arr: (n_train,), indicates alternative id for group id.
    # # generally, group ids are not 0-index and continuous or not integer.
    gid2aid = dict(zip(gid_unique, range(n_group)))
#     aid2gid = dict(zip(range(n_group), gid_unique))
    aid_arr = np.vectorize(lambda x: gid2aid[x])(gid_arr)

    # # count labels by class
    cnts_by_class = label_arr.sum(axis=0)  # (n_class, )

    # # count labels by group id.
    col, row = np.array(sorted(enumerate(aid_arr), key=lambda x: x[1])).T
    cnts_by_group = coo_matrix(
        (np.ones(len(label_arr)), (row, col))
    ).dot(coo_matrix(label_arr)).toarray().astype(int)
    del col
    del row
    cnts_by_fold = np.zeros((n_fold, n_class), int)

    groups_by_fold = [[] for fid in range(n_fold)]
    group_and_cnts = list(enumerate(cnts_by_group))  # pair of aid and cnt by group
    np.random.shuffle(group_and_cnts)
    print("finished preparation", time.time() - start_time)
    for aid, cnt_by_g in sorted(group_and_cnts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for fid in range(n_fold):
            # # eval assignment.
            cnts_by_fold[fid] += cnt_by_g
            fold_eval = (cnts_by_fold / cnts_by_class).std(axis=0).mean()
            cnts_by_fold[fid] -= cnt_by_g

            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = fid

        cnts_by_fold[best_fold] += cnt_by_g
        groups_by_fold[best_fold].append(aid)
    print("finished assignment.", time.time() - start_time)

    gc.collect()
    idx_arr = np.arange(n_train)
    for fid in range(n_fold):
        val_groups = groups_by_fold[fid]

        val_indexs_bool = np.isin(aid_arr, val_groups)
        train_indexs = idx_arr[~val_indexs_bool]
        val_indexs = idx_arr[val_indexs_bool]

        print("[fold {}]".format(fid), end=" ")
        print("n_group: (train, val) = ({}, {})".format(n_group - len(val_groups), len(val_groups)), end=" ")
        print("n_sample: (train, val) = ({}, {})".format(len(train_indexs), len(val_indexs)))

        yield train_indexs, val_indexs

from scipy import interpolate
def interpolate_mask(data):
    f = interpolate.interp1d(data[:, 0], data[:, 1])
    xnew = np.arange(data[:, 0].min(), data[:, 0].max(), 1)
    fnew = f(xnew)
    return np.concatenate([xnew[:, None], fnew[:, None]], axis = -1).astype(int)

COLOR_MAP = {'ETT - Abnormal': 0,
             'ETT - Borderline': 0,
             'ETT - Normal': 0,
             'NGT - Abnormal': 0,
             'NGT - Borderline': 0,
             'NGT - Incompletely Imaged': 0,
             'NGT - Normal': 0,
             'CVC - Abnormal': 0,
             'CVC - Borderline': 0,
             'CVC - Normal': 0,
             'Swan Ganz Catheter Present': 0,
            }

####################
# Config
####################

conf_dict = {'batch_size': 32, 
             'epoch': 30,
             'image_size': 640,
             'model_name': 'timm_efficientnet_b0',
             'lr': 0.001,
             'fold': 0,
             'data_dir': '../input/ranzcr-clip-catheter-line-classification',
             'output_dir': './'}
conf_base = OmegaConf.create(conf_dict)


####################
# Dataset
####################
import ast
class RANZCRSegDataset(Dataset):
    def __init__(self, df, df_annotations, annot_size=10, transform=None, mode = 'train'):
        target_cols=['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                         'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
                         'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                         'Swan Ganz Catheter Present']    
        self.df = df
        self.df_annotations = df_annotations
        self.annot_size = annot_size
        self.file_names = df['file_path'].values
        self.patient_id = df['StudyInstanceUID'].values
        self.labels = df[target_cols].values
        self.transform = transform
        self.mode = mode
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        patient_id = self.patient_id[idx]
        no_anno = 1
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        seg_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype = np.float32)
        query_string = f"StudyInstanceUID == '{patient_id}'"
        df = self.df_annotations.query(query_string)
        if len(df) == 0:
            no_anno = 0
        for i, row in df.iterrows():
            label = row["label"]
            data = np.array(ast.literal_eval(row["data"]))
            for data_point in range(len(data)):
                point_pairs = data[data_point: data_point + 2]
                if len(point_pairs) < 2:
                    continue
                for d in interpolate_mask(point_pairs):
                    seg_mask[d[1]-self.annot_size//2:d[1]+self.annot_size//2,
                          d[0]-self.annot_size//2:d[0]+self.annot_size//2,
                          COLOR_MAP[label]] = 1
        if self.transform:
            augmented = self.transform(image=image, mask = seg_mask)
            image = augmented['image'].transpose(2, 0, 1)
            mask = augmented['mask'].transpose(2, 0, 1)
        else:
            mask = seg_mask
        if self.mode == 'test':
            return torch.tensor(image).float()
        else:
            label = torch.tensor(self.labels[idx]).float()
            no_anno = torch.tensor(no_anno)
            return file_name, torch.tensor(image).float(), torch.tensor(mask).float(), label, no_anno.float()
           
####################
# Data Module
####################

class RANZCRDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.target_cols=['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                         'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
                         'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                         'Swan Ganz Catheter Present']       

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            df = pd.read_csv(os.path.join(self.conf.data_dir, "train.csv"))
            df_ann = pd.read_csv(os.path.join(self.conf.data_dir, "train_annotations.csv"))
            df['file_path'] = df.StudyInstanceUID.apply(lambda x: os.path.join(self.conf.data_dir, f'train/{x}.jpg'))
            
            label_arr = df[self.target_cols].values
            group_id = df['PatientID'].values
            train_val_indexs = list(multi_label_stratified_group_k_fold(label_arr, group_id, 5, 2021))
            df["fold"] = -1
            for fold_id, (trn_idx, val_idx) in enumerate(train_val_indexs):
                df.loc[val_idx, "fold"] = fold_id
                
            train_df = df[df['fold'] != self.conf.fold]
            valid_df = df[df['fold'] == self.conf.fold]
            
            train_df = train_df[train_df['StudyInstanceUID'].isin(df_ann['StudyInstanceUID'].unique())].reset_index(drop=True)
            #train_df, valid_df = model_selection.train_test_split(df, test_size=0.2, random_state=42)

            train_transform = A.Compose([
                        A.RandomResizedCrop(height=self.conf.image_size, width=self.conf.image_size, scale=(0.75, 1), p=1), 
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(p=0.5),
                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                        A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
                        A.CLAHE(clip_limit=(1,4), p=0.5),
                        A.OneOf([
                            A.OpticalDistortion(distort_limit=1.0),
                            A.GridDistortion(num_steps=5, distort_limit=1.),
                            A.ElasticTransform(alpha=3),
                        ], p=0.20),
                        A.OneOf([
                            A.GaussNoise(var_limit=[10, 50]),
                            A.GaussianBlur(),
                            A.MotionBlur(),
                            A.MedianBlur(),
                        ], p=0.20),
                        #A.Resize(size, size),
                        A.OneOf([
                            A.JpegCompression(quality_lower=95, quality_upper=100, p=0.50),
                            A.Downscale(scale_min=0.75, scale_max=0.95),
                        ], p=0.2),
                        A.IAAPiecewiseAffine(p=0.2),
                        A.IAASharpen(p=0.2),
                        A.Cutout(max_h_size=int(self.conf.image_size * 0.1), max_w_size=int(self.conf.image_size * 0.1), num_holes=5, p=0.5),
                        A.Normalize()
                        ])

            valid_transform = A.Compose([
                        A.Resize(height=self.conf.image_size, width=self.conf.image_size, interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])

            self.train_dataset = RANZCRSegDataset(train_df, df_ann, annot_size = 20, transform=train_transform, mode = 'train')
            self.valid_dataset = RANZCRSegDataset(valid_df, df_ann, annot_size = 20, transform=train_transform, mode = 'train')
            
        elif stage == 'test':
            test_df = pd.read_csv(os.path.join(self.conf.data_dir, "sample_submission.csv"))
            test_transform = A.Compose([
                        A.Resize(height=self.conf.image_size, width=self.conf.image_size, interpolation=1, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])
            self.test_dataset = RANZCRDataset(test_df, os.path.join(self.conf.data_dir, 'test'), transform=test_transform, train=False)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
####################
# Lightning Module
####################
import segmentation_models_pytorch as smp
import torch.nn as nn
class SegNet(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=11, pretrained=False):
        super().__init__()
        aux_params=dict(
                        pooling='avg',
                        dropout=0.3,
                        classes=out_dim)
        self.model = smp.Unet(model_name, encoder_weights="imagenet", classes = 1, aux_params = aux_params)
    def forward(self, x):
        mask_logits, logits = self.model(x)
        #mask_logits = mask_logits.permute(0, 2, 3, 1)
        return mask_logits, logits
    
class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        #self.conf = conf
        self.save_hyperparameters(conf)
        #self.model = timm.create_model(model_name=self.hparams.model_name, num_classes=11, pretrained=True, in_chans=3)
        self.model = SegNet(model_name=self.hparams.model_name, out_dim=11, pretrained=False)
        self.criteria = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)
        
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        _, x, mask, y, no_anno = batch
        y_hat_mask, y_hat = self.model(x)
        cls_loss = self.criteria(y_hat, y).mean()
        seg_loss = (self.criteria(y_hat_mask, mask).mean(axis = -1).mean(axis = -1) * no_anno).mean()
        
        self.log('train_seg_loss', seg_loss, on_epoch=True)
        self.log('train_cls_loss', cls_loss, on_epoch=True)
        return cls_loss + seg_loss

    def validation_step(self, batch, batch_idx):
        _, x, mask, y, no_anno = batch
        y_hat_mask, y_hat = self.model(x)
        cls_loss = self.criteria(y_hat, y).mean()
        seg_loss = (self.criteria(y_hat_mask, mask).mean(axis = -1).mean(axis = -1) * no_anno).mean()
        
        return {
            "val_seg_loss": seg_loss,
            "val_cls_loss": cls_loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_seg_loss = torch.stack([x["val_seg_loss"] for x in outputs]).mean()
        avg_val_cls_loss = torch.stack([x["val_cls_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu()
        y_hat = torch.sigmoid(torch.cat([x["y_hat"] for x in outputs])).cpu()

        #preds = np.argmax(y_hat, axis=1)

        val_score, _ = get_score(y, y_hat)

        self.log('avg_val_seg_loss', avg_val_seg_loss)
        self.log('avg_val_cls_loss', avg_val_cls_loss)
        self.log('val_score', val_score)
        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='val_score', 
                                          save_last=True, save_top_k=5, mode='max', 
                                          save_weights_only=True, filename='{epoch}-{val_score:.2f}')

    data_module = RANZCRDataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        num_sanity_val_steps=0,
        val_check_interval=1.0
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()
