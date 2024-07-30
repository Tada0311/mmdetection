# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import logging
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from .amp import autocast
from .base_loop import BaseLoop
from .utils import calc_dynamic_intervals


@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


class _InfiniteDataloaderIterator:
    """An infinite dataloader iterator wrapper for IterBasedTrainLoop.

    It resets the dataloader to continue iterating when the iterator has
    iterated over all the data. However, this approach is not efficient, as the
    workers need to be restarted every time the dataloader is reset. It is
    recommended to use `mmengine.dataset.InfiniteSampler` to enable the
    dataloader to iterate infinitely.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        self._iterator = iter(self._dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Sequence[dict]:
        try:
            data = next(self._iterator)
        except StopIteration:
            print_log(
                'Reach the end of the dataloader, it will be '
                'restarted and continue to iterate. It is '
                'recommended to use '
                '`mmengine.dataset.InfiniteSampler` to enable the '
                'dataloader to iterate infinitely.',
                logger='current',
                level=logging.WARNING)
            self._epoch += 1
            if hasattr(self._dataloader, 'sampler') and hasattr(
                    self._dataloader.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, 'batch_sampler') and hasattr(
                    self._dataloader.batch_sampler.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
        return data


@LOOPS.register_module()
class IterBasedTrainLoop(BaseLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_iters: int,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_iters = int(max_iters)
        assert self._max_iters == max_iters, \
            f'`max_iters` should be a integer number, but get {max_iters}'
        self._max_epochs = 1  # for compatibility with EpochBasedTrainLoop
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)
        # get the iterator of the dataloader
        self.dataloader_iterator = _InfiniteDataloaderIterator(self.dataloader)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            for _ in range(self._iter):
                next(self.dataloader_iterator)
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self._iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        super().__init__(runner, dataloader)

        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self,modelname) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(tqdm(self.dataloader)):
            self.run_iter(idx, data_batch,modelname)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict],modelname) -> None:
        
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        
        # zht
        import torch
        from torchvision.transforms import ToPILImage
        from PIL import Image
        import copy


        import sys
        import numpy as np
        import os
        from datetime import datetime
        from torchvision.transforms import ToPILImage
        

        sys.path.append("/users/r/z/rzh/mmdetect/mmdetection")
        sys.path.append("/users/r/z/rzh/mmdetect/mmdetection/BlackboxBench-main/query")
    
        # from attacker.random_attacker import RandomAttacker 
        # attacker = RandomAttacker(data_batch)
        from mmdet.evaluation.functional import eval_map
        from tools.analysis_tools.analyze_results import bbox_map_eval
        from attacker.square_attack import SquareAttack 
        attacker = SquareAttack(epsilon=8,p="inf",name = "Square"
                             , max_loss_queries=10000,batch_size =1,lb = 0,ub=255,p_init= 0.05)


        outputs_worst = None
        data_batch_worst = None
        mean_ap_worst = 1.0
        loss_fct_lambda = lambda db,q: self.loss_fct(db,q)
        
        
        
        
        origin_batch = copy.deepcopy(data_batch)
            
        inputs = copy.deepcopy(origin_batch['inputs'][0].unsqueeze(0))
        # inputs = torch.from_numpy(inputs)
        inputs = inputs.to(torch.float32)
        attacker.new_batch(is_new=1)

        for i in range(1, 151):


            data_back,_ = attacker._perturb(xs_t = inputs, loss_fct = loss_fct_lambda,data_batch=origin_batch)
          
            noise = torch.clamp(data_back- origin_batch['inputs'][0], min=-8, max=8) 
            data_back = noise + origin_batch['inputs'][0]
            # print(torch.max(data_back - origin_batch['inputs'][0])) 
            
            data_back = torch.clamp(data_back, min=0, max=255)
            inputs = copy.deepcopy(data_back)
        
            
            
            outputs_worst,data_batch_worst,mean_ap_worst =self.loss_fct2(data_back,outputs_worst,data_batch_worst,mean_ap_worst,i,origin_batch) 
            attacker.new_batch(is_new=2)
                               


        self.evaluator.process(data_samples=outputs_worst, data_batch=data_batch_worst)
   

        bgr_tensor = data_batch_worst['inputs'][0]
        rgb_tensor = bgr_tensor[[2, 1, 0], :, :].to(torch.uint8)
        to_pil = ToPILImage()
        image = to_pil(rgb_tensor)
        save_dir = '/users/r/z/rzh/mmdetect/mmdetection/img'+str(modelname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"noise{timestamp}.jpg"
        file_path = os.path.join(save_dir, file_name)
        image.save(file_path)
        print(mean_ap_worst)

        # zht over

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch_worst,
            outputs=outputs_worst)
        
    
        
    @torch.no_grad()
    def loss_fct(self,xs_t2,data_batch2):
        with autocast(enabled=self.fp16):
            import numpy as np
            import copy
            from mmdet.evaluation.functional import eval_map
            from tools.analysis_tools.analyze_results import bbox_map_eval
            data_batch = copy.deepcopy(data_batch2)              
            xs_t =  copy.deepcopy(xs_t2)            
            data_batch['inputs'][0] = xs_t.squeeze(0)
            outputs = self.runner.model.test_step(data_batch)

            # print(outputs)

            
            det_results = []
            annotations = []

            for output in outputs:
                # 提取预测实例
                pred_instances = output.pred_instances
                pred_bboxes = pred_instances.bboxes.cpu().numpy()
                pred_scores = pred_instances.scores.cpu().numpy()
                pred_labels = pred_instances.labels.cpu().numpy()

                # 将预测的边界框和得分组合，确保格式正确
                det_result = np.hstack((pred_bboxes, pred_scores[:, np.newaxis]))
                # 分别存储每个类别的检测结果
                det_results.append([det_result[pred_labels == i] for i in range(max(pred_labels) + 1)])

                # 提取真实实例
                gt_instances = output.gt_instances
                gt_bboxes = gt_instances.bboxes.cpu().numpy()
                gt_labels = gt_instances.labels.cpu().numpy()

                annotation = {
                    'bboxes': gt_bboxes,
                    'labels': gt_labels
                }
                annotations.append(annotation)

            # 计算 mAP
            # mean_ap, eval_results = eval_map(det_results, annotations, iou_thr=0.5, dataset='coco', logger='silent')
            mean_ap = bbox_map_eval(det_results[0], annotations[0])
            loss = 1 - mean_ap
            # print("mean_ap" +str(mean_ap))
            loss = torch.tensor(loss)

            return loss

        
    @torch.no_grad()
    def loss_fct2(self,xs_t2,outputs_worst= None,data_batch_worst= None,mean_ap_worst= 1.0,i=1,origin_batch2 = None):
        with autocast(enabled=self.fp16):
            import numpy as np
            import copy
            from mmdet.evaluation.functional import eval_map
            from tools.analysis_tools.analyze_results import bbox_map_eval
                               
            origin_batch =  copy.deepcopy(origin_batch2)              
            xs_t =  copy.deepcopy(xs_t2).squeeze(0)  
                               
            origin_batch['inputs'][0] = xs_t
            outputs = self.runner.model.test_step(origin_batch)

            # print(outputs)

            
            det_results = []
            annotations = []

            for output in outputs:
                # 提取预测实例
                pred_instances = output.pred_instances
                pred_bboxes = pred_instances.bboxes.cpu().numpy()
                pred_scores = pred_instances.scores.cpu().numpy()
                pred_labels = pred_instances.labels.cpu().numpy()

                # 将预测的边界框和得分组合，确保格式正确
                det_result = np.hstack((pred_bboxes, pred_scores[:, np.newaxis]))
                # 分别存储每个类别的检测结果
                det_results.append([det_result[pred_labels == i] for i in range(max(pred_labels) + 1)])

                # 提取真实实例
                gt_instances = output.gt_instances
                gt_bboxes = gt_instances.bboxes.cpu().numpy()
                gt_labels = gt_instances.labels.cpu().numpy()

                annotation = {
                    'bboxes': gt_bboxes,
                    'labels': gt_labels
                }
                annotations.append(annotation)

            # 计算 mAP
            # mean_ap, eval_results = eval_map(det_results, annotations, iou_thr=0.5, dataset='coco', logger='silent')
            mean_ap = bbox_map_eval(det_results[0], annotations[0])
            
            # print(f"Mean AP: {mean_ap}")

            # 记录攻击效果最好的结果
            if i == 1 or mean_ap < mean_ap_worst:
                outputs_worst = copy.deepcopy(outputs)
                data_batch_worst = copy.deepcopy(origin_batch)
                mean_ap_worst = mean_ap
            return outputs_worst,data_batch_worst,mean_ap_worst
        