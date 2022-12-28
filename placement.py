import argparse
import os
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from compose import compose_images
from loader import dataset_dict, get_loader
from loader.utils import gen_composite_image


def sample(sample_dataset, model, iter, gen_dir):
    model.start_eval()
    id_list = [0, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 10000]
    for id in id_list:
        index, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, comp_crop_feat, label, trans_label, catnm = sample_dataset[id]
        pred_img, pred_msk, pred_trans = model.test_genorator(bg_feat.unsqueeze(0), fg_feat.unsqueeze(0), fg_msk_feat.unsqueeze(0), torch.Tensor(fg_bbox).unsqueeze(0))
        img = transforms.ToPILImage()(pred_img.cpu()[0]).convert('RGB')
        sample_dir = os.path.join(gen_dir, str(id))
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        img.save(os.path.join(sample_dir, '{}.jpg'.format(iter)))


def sample(sample_dataset, model, iter, gen_dir):
    model.start_eval()
    id_list = [0, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 10000]
    for id in id_list:
        index, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, comp_crop_feat, label, trans_label, catnm = sample_dataset[id]
        pred_img, pred_msk, pred_trans = model.test_genorator(bg_feat.unsqueeze(0), fg_feat.unsqueeze(0), fg_msk_feat.unsqueeze(0), torch.Tensor(fg_bbox).unsqueeze(0))
        img = transforms.ToPILImage()(pred_img.cpu()[0]).convert('RGB')
        sample_dir = os.path.join(gen_dir, str(id))
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        img.save(os.path.join(sample_dir, '{}.jpg'.format(iter)))


def infer(eval_loader, opt, model=None, repeat=1):
    def csv_title():
        return 'annID,scID,bbox,catnm,label,img_path,msk_path'
    def csv_str(annid, scid, gen_comp_bbox, catnm, gen_file_name):
        return '{},{},"{}",{},-1,images/{}.jpg,masks/{}.png'.format(annid, scid, gen_comp_bbox, catnm, gen_file_name, gen_file_name)

    assert (repeat >= 1)
    save_dir = os.path.join('result', opt.expid)
    eval_dir = os.path.join(save_dir, opt.eval_type, str(opt.epoch))
    assert (not os.path.exists(eval_dir))
    img_sav_dir = os.path.join(eval_dir, 'images')
    msk_sav_dir = os.path.join(eval_dir, 'masks')
    csv_sav_file = os.path.join(eval_dir, '{}.csv'.format(opt.eval_type))
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if not os.path.exists(img_sav_dir):
        os.mkdir(img_sav_dir)
    if not os.path.exists(msk_sav_dir):
        os.mkdir(msk_sav_dir)

    if model is None:
        from model import GAN
        model_dir = os.path.join(save_dir, 'models')
        model_path = os.path.join(model_dir, str(opt.epoch) + '.pth')
        assert(os.path.exists(model_path))
        model = GAN(opt)
        loaded = torch.load(model_path)
        assert(opt.epoch == loaded['epoch'])
        model.load_state_dict(loaded['model'], strict=True)
    model.start_eval()

    gen_res = []

    for i, (indices, annids, scids, bg_img_arrs, fg_img_arrs, fg_msk_arrs, comp_img_arrs, comp_msk_arrs, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, comp_img_feats, comp_msk_feats, comp_crop_feats, labels, trans_labels, catnms) in enumerate(tqdm(eval_loader)):
        index, annid, scid, bg_img_arr, fg_img_arr, fg_msk_arr, comp_img_arr, comp_msk_arr, label, trans_label, catnm = \
            indices[0], annids[0], scids[0], bg_img_arrs[0], fg_img_arrs[0], fg_msk_arrs[0], comp_img_arrs[0], comp_msk_arrs[0], labels[0], trans_labels[0], catnms[0]
        for repeat_id in range(repeat):
            pred_img_, pred_msk_, pred_trans_ = model.test_genorator(bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes)
            gen_comp_img, gen_comp_msk, gen_comp_bbox = gen_composite_image(
                bg_img=Image.fromarray(bg_img_arr.numpy().astype(np.uint8)).convert('RGB'), 
                fg_img=Image.fromarray(fg_img_arr.numpy().astype(np.uint8)).convert('RGB'), 
                fg_msk=Image.fromarray(fg_msk_arr.numpy().astype(np.uint8)).convert('L'), 
                trans=(pred_trans_.cpu().numpy().astype(np.float32)[0]).tolist(),
                fg_bbox=None
            )
            if repeat == 1:
                gen_file_name = "{}_{}_{}_{}_{}_{}_{}".format(index, annid, scid, gen_comp_bbox[0], gen_comp_bbox[1], gen_comp_bbox[2], gen_comp_bbox[3])
            else:
                gen_file_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(index, repeat_id, annid, scid, gen_comp_bbox[0], gen_comp_bbox[1], gen_comp_bbox[2], gen_comp_bbox[3])
            gen_comp_img.save(os.path.join(img_sav_dir, '{}.jpg'.format(gen_file_name)))
            gen_comp_msk.save(os.path.join(msk_sav_dir, '{}.png'.format(gen_file_name)))
            gen_res.append(csv_str(annid, scid, gen_comp_bbox, catnm, gen_file_name))

    with open(csv_sav_file, "w") as f:
        f.write(csv_title() + '\n')
        for line in gen_res:
            f.write(line + '\n')


class args:
    def __init__(self):
        self.lr = 0.00002
        self.d_noise = 1024
        self.d_model = 512
        self.d_k = 64
        self.d_v = 64
        self.n_heads = 8
        self.len_k = 84
        self.b1 = 0.5
        self.b2 = 0.999
        self.dst = "OPADst1"
        self.img_size = 256
        self.expid = 'graconet'
        self.data_root = "dataset"
        self.eval_type = "eval"
        self.epoch = 11
        self.repeat = 1
        self.category = 'person'

def information(comp, mask, bbox, scale):
    annID = '1'
    scID = '2'
    bbox = [605, 23, 116, 328]
    scale = scale
    label = '0'
    catnm = 'person'
    new_img_path = f'composite/{annID}_{scID}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{scale}_{label}.jpg'
    new_msk_path = f'composite/mask_{annID}_{scID}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{scale}_{label}.jpg'
    composite_path = os.path.join('dataset', new_img_path[:-4]+'.png')
    mask_path = os.path.join('dataset', new_msk_path[:-4]+'.png')
    comp.save(composite_path)
    os.rename(composite_path, 'dataset/'+new_img_path)
    mask.save(mask_path)
    os.rename(mask_path, 'dataset/'+new_msk_path)
    # annID = '3'
    # scID = '4'
    # bbox = '[216, 100, 363, 546]'
    # scale = 0.99
    # label = '1'
    # catnm = 'dog'
    # new_img_path = 'composite/test_set/3_4_71_338_101_154_0.99_1.jpg'
    # new_msk_path = 'composite/test_set/mask_3_4_71_338_101_154_0.99_1.jpg'
    # annID = '3'
    # scID = '4'
    # bbox = '[71, 338, 101, 154]'
    # scale = 0.0001
    # label = '1'
    # catnm = 'dog'
    # new_img_path = 'composite/test_set/3_4_71_338_101_154_0.1_0.jpg'
    # new_msk_path = 'composite/test_set/mask_3_4_71_338_101_154_0.1_0.jpg'
    info = [0, int(annID), int(scID),
            bbox, scale, int(label), catnm,
            new_img_path, new_msk_path]
    return [info]

def place(foreground='input/foreground.png',background='input/background.png',fg_mask='output_matting/fg_mask.png',scale=0.99):
    opt = args()
    scale = scale
    category = opt.category
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
        os.mkdir('dataset/background')
        os.mkdir('dataset/foreground')
        os.mkdir('dataset/transparent_mask')
        os.mkdir('dataset/background/person')
        os.mkdir('dataset/foreground/person')
        os.mkdir('dataset/transparent_mask/person')
    # path to transparent foreground
    # foreground = f'dataset/transparent_mask/{category}/1.png'
    # foreground = f'input/foreground.png'
    # path to background
    # background = f'dataset/background/{category}/2.jpg'
    # background = f'input/background.png'
    # generate composite images and receive mask
    comp, mask, bbox = compose_images(foreground, background)
    # path to mask for foreground
    foreground_mask = f'output_matting/fg_mask.png'
    foreground_dir = f'dataset/foreground/{category}/'
    # shutil.copy(background, f'dataset/background/{category}/2.jpg')
    shutil.copy(foreground, foreground_dir+'1.jpg')
    shutil.copy(background,'dataset/background/person/2.jpg')
    shutil.copy(foreground_mask, foreground_dir+'mask_1.jpg')
    if not os.path.exists('dataset/composite/'):
        os.mkdir('dataset/composite/')
    info = information(comp, mask, bbox, scale)
    eval_loader = get_loader(opt.dst, batch_size=1, num_workers=1, image_size=opt.img_size, shuffle=False, mode_type=opt.eval_type, data_root=opt.data_root, info=info)
    with torch.no_grad():
        infer(eval_loader, opt, model=None, repeat=opt.repeat)
    if not os.path.exists('output_object_placement'):
        os.mkdir('output_object_placement')
    image_name = os.listdir('result/graconet/eval/11/images')[0]
    mask_name = os.listdir('result/graconet/eval/11/masks')[0]
    shutil.copy('result/graconet/eval/11/images/'+image_name, '../output_object_placement/composite_image.png')
    shutil.copy('result/graconet/eval/11/masks/'+mask_name, '../output_object_placement/composite_mask.png')

if __name__ == '__main__':
  place()
