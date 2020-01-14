#!/usr/bin/python
#
# Copyright 2018 Google LLC
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

import argparse, json, os

from imageio import imwrite
import torch

from sg2im.model import Sg2ImModel
from sg2im.data.utils import imagenet_deprocess_batch
import sg2im.vis as vis

import pickle
import pprint
from sklearn_crfsuite import CRF 
from sklearn_crfsuite import metrics

import gensim
from gensim.models import Word2Vec

import nltk
from nltk import word_tokenize
from nltk.tag.util import untag
from nltk.tree import ParentedTree, Tree
from nltk.corpus import brown, movie_reviews, treebank
from sg2im.drawer import *

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--crf_model_pretrained', default='./simple_crf/crf_model.pkl')
parser.add_argument('--coco_cls_txt', default='./datasets/coco_cls.txt')
parser.add_argument('--word2vec_sample', default='models/word2vec_sample/pruned.word2vec.txt')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])


def main(args):
  if not os.path.isfile(args.checkpoint):
    print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
    print('Maybe you forgot to download pretraind models? Try running:')
    print('bash scripts/download_models.sh')
    return

  if not os.path.isdir(args.output_dir):
    print('Output directory "%s" does not exist; creating it' % args.output_dir)
    os.makedirs(args.output_dir)

  if args.device == 'cpu':
    device = torch.device('cpu')
  elif args.device == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      print('WARNING: CUDA not available; falling back to CPU')
      device = torch.device('cpu')

  # Load the model, with a bit of care in case there are no GPUs
  map_location = 'cpu' if device == torch.device('cpu') else None
  checkpoint = torch.load(args.checkpoint, map_location=map_location)
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.to(device)

  # Load the scene graphs
#   with open(args.scene_graphs_json, 'r') as f:
#     scene_graphs = json.load(f)
  crf_model_path = args.crf_model_pretrained
  crf_model = pickle.load(open(crf_model_path, 'rb'))
  cate_list = load_cate(args.coco_cls_txt)
  pos_lists, feat_x, feat_y, pca, clf, wn_model = construct_pos_list(args.word2vec_sample)
  print("Start drawing something!")
  count = 0
  sg_list = [{'objects': [], 'relationships': []}]
  while 1:
    sentence = input("Please input a sentence: ")
    # Run the model forward
    # scene_graphs only with one graph
    token_sentence = word_tokenize(sentence)
    t = pos_tag(token_sentence, crf_model)
    print(t)
    so_list, p_list = spo_extractor(t, cate_list)
    if len(so_list) != 2: 
        print("please make sure that input sentence contain 2 objects in coco_list.")
        print("Only find ",so_list)
        continue
    so_list = so_extractor(so_list, cate_list)
    p_list = p_extractor(p_list, pos_lists, feat_x, feat_y, pca, clf, wn_model)
    scene_graphs = sg_constructor(so_list, p_list, sg_list)
    print(sg_list)
    with torch.no_grad():
      imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graphs)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated images
    for i in range(imgs.shape[0]):
      img_np = imgs[i].numpy().transpose(1, 2, 0)
      img_path = os.path.join(args.output_dir, 'img%06d.png' % count)
      imwrite(img_path, img_np)

    # Draw the scene graphs
    if args.draw_scene_graphs == 1:
      for i, sg in enumerate(scene_graphs):
        sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
        sg_img_path = os.path.join(args.output_dir, 'sg%06d.png' % count)
        imwrite(sg_img_path, sg_img)
    count += 1

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

