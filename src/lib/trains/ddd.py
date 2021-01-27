from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import torch.nn.functional as F

from models.losses import FocalLoss, L1Loss, BinRotLoss
from models.decode import ddd_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ddd_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
from models.utils import _transpose_and_gather_feat

class DddLoss(torch.nn.Module):
  def __init__(self, opt):
    super(DddLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = L1Loss()
    self.crit_rot = BinRotLoss()
    self.opt = opt
    self.emb_scale = 1
  
  def forward(self, outputs, batch):
    opt = self.opt

    hm_loss, dep_loss, rot_loss, dim_loss = 0, 0, 0, 0
    wh_loss, off_loss = 0, 0
    id_loss = 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
      
      if opt.eval_oracle_dep:
        output['dep'] = torch.from_numpy(gen_oracle_map(
          batch['dep'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          opt.output_w, opt.output_h)).to(opt.device)
      
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.dep_weight > 0:
        dep_loss += self.crit_reg(output['dep'], batch['reg_mask'],
                                  batch['ind'], batch['dep']) / opt.num_stacks
      if opt.dim_weight > 0:
        dim_loss += self.crit_reg(output['dim'], batch['reg_mask'],
                                  batch['ind'], batch['dim']) / opt.num_stacks
      if opt.rot_weight > 0:
        rot_loss += self.crit_rot(output['rot'], batch['rot_mask'],
                                  batch['ind'], batch['rotbin'],
                                  batch['rotres']) / opt.num_stacks
      if opt.reg_bbox and opt.wh_weight > 0:
        wh_loss += self.crit_reg(output['wh'], batch['rot_mask'],
                                 batch['ind'], batch['wh']) / opt.num_stacks
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['rot_mask'],
                                  batch['ind'], batch['reg']) / opt.num_stacks
      # if opt.reg_id and opt.id_weight > 0:
      #   pass
        # id_head = _transpose_and_gather_feat(output['id'],batch['ind'])
        # id_head = id_head[batch['reg_mask'] > 0].contiguous()
        # id_head = self.emb_scale * F.normalize(id_head)
        # id_target = batch['ids'][batch['reg_mask'] > 0]

        # id_output = 
    loss = opt.hm_weight * hm_loss + opt.dep_weight * dep_loss + \
           opt.dim_weight * dim_loss + opt.rot_weight * rot_loss + \
           opt.wh_weight * wh_loss + opt.off_weight * off_loss
    # loss = opt.hm_weight * hm_loss + opt.dep_weight * dep_loss + \
    #        opt.dim_weight * dim_loss + opt.rot_weight * rot_loss + \
    #        opt.off_weight * off_loss

    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'dep_loss': dep_loss, 
                  'dim_loss': dim_loss, 'rot_loss': rot_loss
                   }
    #需要记录以下，这会儿还挺佩服自己解决了这个bug。
    #当opt中加上了--not_reg_wh的时候，会报错
    #File "/home/ubuntu/Anaconda3/envs/CenterTrack/lib/python3.6/site-packages/torch/nn/parallel/scatter_gather.py", line 63, in gather_map
    #return type(out)(map(gather_map, zip(*outputs)))
    #TypeError: zip argument #1 must support iteration
    #简直神奇，经过一天多的定位，查到了原来使用dataparallel的时候，从多个gpu合并tensor，必须要每一个元素都是tensor，并且tensor处于gpu内
    #最终发现是loss_stats中，当没有reg_wh的时候，wh_loss是数字0，就会报这个错，把原本代码嵌入loss_statas的wh_loss取出来就好了
    #脑溢血的地方就是加上wh就没问题，只要--not_reg_...就出错，以为遇见鬼了，必须是特定个数的网络hed才行。想到这里可能有问题是我把head保持原本个数还是报错
    #才想到可能是硬编码问题。中间一度想放弃，但是想到后面还要魔改网络总要解决问题，最终坚持下来了，给记几点赞！2021.1.27

    if opt.reg_bbox and opt.wh_weight > 0:
      loss_stats.update({'wh_loss': wh_loss})
    if opt.reg_offset and opt.off_weight > 0:
      loss_stats.update({'off_loss': off_loss})
    return loss, loss_stats

class DddTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(DddTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'dep_loss', 'dim_loss', 'rot_loss'
                   ]
    if opt.reg_bbox and opt.wh_weight > 0:
      loss_states.append('wh_loss')
    if opt.reg_offset and opt.off_weight > 0:
      loss_states.append('off_loss')
    loss = DddLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
      opt = self.opt
      wh = output['wh'] if opt.reg_bbox else None
      reg = output['reg'] if opt.reg_offset else None
      dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], wh=wh, reg=reg, K=opt.K)

      # x, y, score, r1-r8, depth, dim1-dim3, cls
      dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
      calib = batch['meta']['calib'].detach().numpy()
      # x, y, score, rot, depth, dim1, dim2, dim3
      # if opt.dataset == 'gta':
      #   dets[:, 12:15] /= 3
      dets_pred = ddd_post_process(
        dets.copy(), batch['meta']['c'].detach().numpy(), 
        batch['meta']['s'].detach().numpy(), calib, opt)
      dets_gt = ddd_post_process(
        batch['meta']['gt_det'].detach().numpy().copy(),
        batch['meta']['c'].detach().numpy(), 
        batch['meta']['s'].detach().numpy(), calib, opt)
      #for i in range(input.size(0)):
      for i in range(1):
        debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),
                            theme=opt.debugger_theme)
        img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
        pred = debugger.gen_colormap(
          output['hm'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'hm_pred')
        debugger.add_blend_img(img, gt, 'hm_gt')
        # decode
        debugger.add_ct_detection(
          img, dets[i], show_box=opt.reg_bbox, center_thresh=opt.center_thresh, 
          img_id='det_pred')
        debugger.add_ct_detection(
          img, batch['meta']['gt_det'][i].cpu().numpy().copy(), 
          show_box=opt.reg_bbox, img_id='det_gt')
        debugger.add_3d_detection(
          batch['meta']['image_path'][i], dets_pred[i], calib[i],
          center_thresh=opt.center_thresh, img_id='add_pred')
        debugger.add_3d_detection(
          batch['meta']['image_path'][i], dets_gt[i], calib[i],
          center_thresh=opt.center_thresh, img_id='add_gt')
        # debugger.add_bird_view(
        #   dets_pred[i], center_thresh=opt.center_thresh, img_id='bird_pred')
        # debugger.add_bird_view(dets_gt[i], img_id='bird_gt')
        debugger.add_bird_views(
          dets_pred[i], dets_gt[i], 
          center_thresh=opt.center_thresh, img_id='bird_pred_gt')
        
        # debugger.add_blend_img(img, pred, 'out', white=True)
        debugger.compose_vis_add(
          batch['meta']['image_path'][i], dets_pred[i], calib[i],
          opt.center_thresh, pred, 'bird_pred_gt', img_id='out')
        # debugger.add_img(img, img_id='out')
        if opt.debug ==4:
          debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
        else:
          debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    opt = self.opt
    wh = output['wh'] if opt.reg_bbox else None
    reg = output['reg'] if opt.reg_offset else None
    dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                        output['dim'], wh=wh, reg=reg, K=opt.K)

    # x, y, score, r1-r8, depth, dim1-dim3, cls
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    calib = batch['meta']['calib'].detach().numpy()
    # x, y, score, rot, depth, dim1, dim2, dim3
    dets_pred = ddd_post_process(
      dets.copy(), batch['meta']['c'].detach().numpy(), 
      batch['meta']['s'].detach().numpy(), calib, opt)
    img_id = batch['meta']['img_id'].detach().numpy()[0]
    results[img_id] = dets_pred[0]
    for j in range(1, opt.num_classes + 1):
      keep_inds = (results[img_id][j][:, -1] > opt.center_thresh)
      results[img_id][j] = results[img_id][j][keep_inds]