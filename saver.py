"""Tools to save/restore model from checkpoints."""

import argparse
import shutil
import sys
import os
import re
import json
import time

import torch

CHECKPOINT_PATTERN = re.compile('^model_checkpoint-(\d+)$')


class ArgsDict(dict):

    def __init__(self, **kwargs):
        super(ArgsDict, self).__init__()
        for key, value in kwargs.items():
            self[key] = value
        self.__dict__ = self


def load_checkpoint(item_dict, model_dir, best, map_location=None, step=None):
    """ item_dict: {"model": model, "opt1": opt1, ...}"""
    if best:
        path = os.path.join(model_dir, 'best_model_checkpoint')
    else:
        path = os.path.join(model_dir, 'model_checkpoint')


    if step is not None:
        path += '-{:08d}'.format(step)
    else:
        last_step = 0
        for filename in os.listdir(model_dir):
            if best:
                if 'best_model_checkpoint-' in filename:
                    step = int(filename[-8:])
                    if step > last_step:
                        last_step = step
            else:
                if 'model_checkpoint-' in filename:
                    step = int(filename[-8:])
                    if step > last_step:
                        last_step = step

            # with open(os.path.join(os.cwd(), filename), 'r') as f: # open in readonly mode
            # do your stuff
        path += '-{:08d}'.format(last_step)

    if os.path.exists(path):
        print("Loading model from %s" % path)
        checkpoint = torch.load(path, map_location=map_location)

        old_state_dict = item_dict["model"].state_dict()
        for key in old_state_dict.keys():
            if key not in checkpoint['model']:
                checkpoint['model'][key] = old_state_dict[key]
            
        for item_name in item_dict:
            # print('item_name', item_name)
            # todo: load grid model without the heads
            item_dict[item_name].load_state_dict(checkpoint[item_name])
            
        return checkpoint.get('step', 0)
    return 0


def load_and_map_checkpoint(model, model_dir, remap):
    path = os.path.join(model_dir, 'model_checkpoint')
    print("Loading parameters %s from %s" % (remap.keys(), model_dir))
    checkpoint = torch.load(path)
    new_state_dict = model.state_dict()
    for name, value in remap.items():
        # TODO: smarter mapping.
        new_state_dict[name] = checkpoint['model'][value]
    model.load_state_dict(new_state_dict)


def save_checkpoint(items, step, model_dir, best, ignore=[],
                    keep_every_n=10000000):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if best:
        path_without_step = os.path.join(model_dir, 'best_model_checkpoint')
    else:
        path_without_step = os.path.join(model_dir, 'model_checkpoint')
    step_padded = format(step, '08d')
    state_dict = items["model"].state_dict()
    if ignore:
        for key in state_dict.keys():
            for item in ignore:
                if key.startswith(item):
                    state_dict.pop(key)
    path_with_step = '{}-{}'.format(path_without_step, step_padded)
    

    saved_dic = {}
    for key in items:
        saved_dic[key] = items[key].state_dict()
    torch.save({**saved_dic, "step": step}, path_with_step)

    try:
        os.unlink(path_without_step)
    except FileNotFoundError:
        pass
    try:
        os.symlink(os.path.basename(path_with_step), path_without_step)
    except OSError:
        shutil.copy2(path_with_step, path_without_step)

    # Cull old checkpoints.
    if keep_every_n is not None:
        all_checkpoints = []
        for name in os.listdir(model_dir):
            m = CHECKPOINT_PATTERN.match(name)
            if m is None or name == os.path.basename(path_with_step):
                continue
            checkpoint_step = int(m.group(1))
            all_checkpoints.append((checkpoint_step, name))
        all_checkpoints.sort()

        last_step = float('-inf')
        for checkpoint_step, name in all_checkpoints:
            if checkpoint_step - last_step >= keep_every_n:
                last_step = checkpoint_step
                continue
            os.unlink(os.path.join(model_dir, name))


class Saver(object):
    """Class to manage save and restore for the model and optimizer."""

    def __init__(self, items, keep_every_n=None):
        assert type(items) == dict
        assert "model" in items
        self._items = items
        self._keep_every_n = keep_every_n

    def restore(self, model_dir, best=False, map_location=None, 
            step=None, item_keys=["model", "optimizer"]):
        """Restores model and optimizer from given directory.
            Specify what should be restored

        Returns:
           Last training step for the model restored.
        """
        items2restore = { k: self._items[k] for k in item_keys}
        last_step = load_checkpoint(
            items2restore, model_dir, best, map_location, step)
        return last_step

    def save(self, model_dir, step, best=False):
        """Saves model and optimizer to given directory.

        Args:
           model_dir: Model directory to save.
           step: Current training step.
        """
        save_checkpoint(self._items, step, model_dir, best,
                        keep_every_n=self._keep_every_n)

    def load_pretrained(self, model, pretrained_path, model_dict, map_location=None):
        # pretrained_dict = ...

        # model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        checkpoint = torch.load(pretrained_path, map_location=map_location)
        pretrained_dict = checkpoint['model']
        # print('checkpoint', pretrained_dict.keys())
        # print('checkpoint', checkpoint.model_dict().keys())
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        # print('pretrained_dict', pretrained_dict.keys())
        # print('-'*80)
        # print('model_dict', model_dict.keys())
        model.load_state_dict(model_dict)
        last_step = int(pretrained_path[-8:])
        return last_step, model


    def restore_part(self, other_model_dir, remap):
        """Restores part of the model from other directory.

        Useful to initialize part of the model with another pretrained model.

        Args:
            other_model_dir: Model directory to load from.
            remap: dict, remapping current parameters to the other model's.
        """
        load_and_map_checkpoint(self._items["model"], other_model_dir, remap)

    def check_if_checkpoint_exists(self, model_dir):
        m_dir = Path(model_dir)
        p1 = m_dir/'best_model_checkpoint'
        p2 = m_dir/'model_checkpoint'
        return p1.exists() or p2.exists()
        

# CHECKPOINT_PATTERN = re.compile('^model_checkpoint-(\d+)$')


# class ArgsDict(dict):

#     def __init__(self, **kwargs):
#         super(ArgsDict, self).__init__()
#         for key, value in kwargs.items():
#             self[key] = value
#         self.__dict__ = self


# def load_checkpoint(item_dict, model_dir, map_location=None, step=None):
#     """ item_dict: {"model": model, "opt1": opt1, ...}"""
#     path = os.path.join(model_dir, 'model_checkpoint')
#     if step is not None:
#         path += '-{:08d}'.format(step)
#     else:
#         last_step = 0
#         for filename in os.listdir(model_dir):
#             if 'model_checkpoint-' in filename:
#                 step = int(filename[-8:])
#                 if step > last_step:
#                     last_step = step

#             # with open(os.path.join(os.cwd(), filename), 'r') as f: # open in readonly mode
#             # do your stuff
#         path += '-{:08d}'.format(last_step)

#     if os.path.exists(path):
#         print("Loading model from %s" % path)
#         checkpoint = torch.load(path, map_location=map_location)

#         old_state_dict = item_dict["model"].state_dict()
#         for key in old_state_dict.keys():
#             if key not in checkpoint['model']:
#                 checkpoint['model'][key] = old_state_dict[key]
            
#         for item_name in item_dict:
#             item_dict[item_name].load_state_dict(checkpoint[item_name])
            
#             if item_name == 'optimizer':
#                 for state in item_dict[item_name].state.values():
#                     for k, v in state.items():
#                         if torch.is_tensor(v):
#                             state[k] = v.to(map_location)
#         return checkpoint.get('step', 0)
#     return 0


# def load_and_map_checkpoint(model, model_dir, remap):
#     path = os.path.join(model_dir, 'model_checkpoint')
#     print("Loading parameters %s from %s" % (remap.keys(), model_dir))
#     checkpoint = torch.load(path)
#     new_state_dict = model.state_dict()
#     for name, value in remap.items():
#         # TODO: smarter mapping.
#         new_state_dict[name] = checkpoint['model'][value]
#     model.load_state_dict(new_state_dict)


# def save_checkpoint(items, step, model_dir, ignore=[],
#                     keep_every_n=10000000):
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     path_without_step = os.path.join(model_dir, 'model_checkpoint')
#     step_padded = format(step, '08d')
#     state_dict = items["model"].state_dict()
#     if ignore:
#         for key in state_dict.keys():
#             for item in ignore:
#                 if key.startswith(item):
#                     state_dict.pop(key)
#     path_with_step = '{}-{}'.format(path_without_step, step_padded)

#     saved_dic = {}
#     for key in items:
#         saved_dic[key] = items[key].state_dict()
#     torch.save({**saved_dic, "step": step}, path_with_step)

#     try:
#         os.unlink(path_without_step)
#     except FileNotFoundError:
#         pass
#     try:
#         os.symlink(os.path.basename(path_with_step), path_without_step)
#     except OSError:
#         shutil.copy2(path_with_step, path_without_step)

#     # Cull old checkpoints.
#     if keep_every_n is not None:
#         all_checkpoints = []
#         for name in os.listdir(model_dir):
#             m = CHECKPOINT_PATTERN.match(name)
#             if m is None or name == os.path.basename(path_with_step):
#                 continue
#             checkpoint_step = int(m.group(1))
#             all_checkpoints.append((checkpoint_step, name))
#         all_checkpoints.sort()

#         last_step = float('-inf')
#         for checkpoint_step, name in all_checkpoints:
#             if checkpoint_step - last_step >= keep_every_n:
#                 last_step = checkpoint_step
#                 continue
#             os.unlink(os.path.join(model_dir, name))


# class Saver(object):
#     """Class to manage save and restore for the model and optimizer."""

#     def __init__(self, items, keep_every_n=None):
#         assert type(items) == dict
#         assert "model" in items
#         self._items = items
#         self._keep_every_n = keep_every_n

#     def restore(self, model_dir, map_location=None, 
#             step=None, item_keys=["model", "optimizer"]):
#         """Restores model and optimizer from given directory.
#             Specify what shoud be restored

#         Returns:
#            Last training step for the model restored.
#         """
#         items2restore = { k: self._items[k] for k in item_keys}
#         last_step = load_checkpoint(
#             items2restore, model_dir, map_location, step)
#         return last_step

#     def save(self, model_dir, step):
#         """Saves model and optimizer to given directory.

#         Args:
#            model_dir: Model directory to save.
#            step: Current training step.
#         """
#         save_checkpoint(self._items, step, model_dir,
#                         keep_every_n=self._keep_every_n)

#     def restore_part(self, other_model_dir, remap):
#         """Restores part of the model from other directory.

#         Useful to initialize part of the model with another pretrained model.

#         Args:
#             other_model_dir: Model directory to load from.
#             remap: dict, remapping current parameters to the other model's.
#         """
#         load_and_map_checkpoint(self._items["model"], other_model_dir, remap)

