import torch
from torch.utils.tensorboard import SummaryWriter
from .utils import *
from tqdm import tqdm
import os
from dataclasses import dataclass
import argparse
from transformers import Trainer, TrainingArguments, HfArgumentParser
import collections
import numpy as np
from datetime import datetime

def train_step(
    model, batch, optimizer, lr_schedule=None, clip_grad=1
):
    """
    进行一次forward + 一次backward + optim.step.

    Args:
        model: the outputs of the model are required to include loss.
        batch: Inputs 需要提前配置好device. 输入形式为dict
        optimizer
        lr_schedule: optional
        clip_grad: optional 切割梯度
    """

    model.train()
    optimizer.zero_grad()
    outputs=model(**batch)
    if isinstance(model, torch.nn.DataParallel):
        loss=outputs.loss.mean()
    else:
        loss=outputs.loss
    
    loss.backward()

    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    optimizer.step()
    if lr_schedule is not None:
        lr_schedule.step()

    return loss, model, optimizer, lr_schedule

def train_loop(
    epoches, dl, model, optimizer, writer=None, lr_schedule=None, 
    eval_func=None, save_epoch=None, save_prefix='', clip_grad=None, 
    save_best=None, small_is_better=None, **eval_args
):
    """[summary]

    Args:
        epoches (int): [description]
        dl ([type]): [description]
        model ([type]): [description]
        optim ([type]): [description]
        writer ([type], optional): TensorBoard SummaryWriter 对象. Defaults to None.
        lr_schedule ([type], optional): [description]. Defaults to None.
    """
    num_steps_per_epoches = len(dl)
    loss = 0
    eval_loss = 0
    if save_best is not None:
        best_criteria = -1000000
    if save_epoch is not None:
        run_sufix = datetime.now().strftime('%b%d_%H-%M-%S')

    for epoch in range(epoches):
        losses = []
        for idx, batch in enumerate(tqdm(dl, desc=f'Training {epoch}th epoch, train loss {loss:.4f}, eval loss {eval_loss:.4f}: ')):
            loss, model, optimizer, lr_schedule = train_step(model, batch, optimizer, clip_grad=clip_grad, lr_schedule=lr_schedule)
            losses.append(loss.item())
        loss = np.mean(losses)
        if writer is not None:
            writer.add_scalar('loss', loss, epoch)
        if eval_func is not None:
            results = eval_func(model, writer = writer, global_step=epoch, **eval_args)
            eval_loss = results['eval_loss']
            if save_best is not None:
                moving_criteria = results[save_best]
                if small_is_better is not None:
                    moving_criteria = - moving_criteria
                if moving_criteria>best_criteria:
                    best_criteria = moving_criteria
        if save_epoch is not None and epoch % save_epoch == 0 and epoch!=0:
            if save_best is not None:
                if moving_criteria < best_criteria:
                    continue
            try:
                os.remove(save_path)
            except:
                pass
            save_path = f'checkpoints/model_{save_prefix}_{save_best}-{best_criteria:.2f}_{run_sufix}.bin' if save_best is not None else \
                f'checkpoints/model_{save_prefix}_{run_sufix}.bin'
            torch.save(model.state_dict(), save_path)
    return model          

class LycTrainer:
    def __init__(
        self,
        model,
        train_dl,
        optimizer,
        num_epoch,
        lr_schedule = None,
        require_logger = None,
        log_path = None
   ):
        self.model=model
        self.train_dl=train_dl
        self.optimizer=optimizer
        self.num_epoch=num_epoch
        self.lr_schedule=lr_schedule

        self.num_train_step_per_epoch=len(self.train_dl)

        if require_logger and log_path:
            self.writer=SummaryWriter(log_path)
    
    def save_setting(self, save_type, save_steps = 500, save_path = 'checkpoints/'):
        assert save_type in ['torch', 'hf']
        self.save_method = save_type
        self.save_steps=save_steps
        self.save_path=save_path
        
    def eval_setting(self, eval_func, eval_dl, eval_steps=500):
        self.eval_func=eval_func
        self.eval_dl=eval_dl
        self.eval_steps=eval_steps
    
    def save(self, model, global_step):
        if self.save_method == 'torch':
            file_name = f'Step-{global_step}.bin'
            torch.save(model.state_dict(), os.path.join(self.save_path, file_name))
        elif self.save_method == 'hf':
            dir_name=f'Step-{global_step}/'
            model.save_pretrained(os.path.join(self.save_path, dir_name))

    def train(self):
        
        for epoch in range(self.num_epoch):
            for index, batch in enumerate(tqdm(self.train_dl, desc=f'Traing for epoch: {epoch}')):

                global_current_step = self.num_train_step_per_epoch * epoch + index
                loss=train_step(self.model, batch, self.optimizer, self.lr_schedule)
                if self.writer is not None:
                    self.writer.add_scalar('train_loss', loss, global_current_step)

                # eval
                if self.eval_func is not None and global_current_step % self.eval_steps == 0 and global_current_step !=0:
                    self.eval_func(self.model, self.eval_dl, self.writer, global_current_step)

                # save
                if self.save_method is not None and global_current_step % self.save_steps ==0 and global_current_step !=0:
                    if isinstance(self.model, torch.nn.DataParallel):
                        self.save(model.module, global_current_step)
                    else:
                        self.save(self.model, global_current_step)

@dataclass
class TrainingArgs:
    # must claim
    train_data : str = None
    save_path : str = None
    dataset_scripts : str = None
    # data
    eval_data_path : str = None
    min_sent_len : int = 10
    max_sent_len : int = 256
    data_workers : int = 8
    tokenizer_name_or_path : str = None
    # train
    lr: float = 1e-3
    sm_lr : float = 5e-5
    weight_decay : float = 1e-5
    early_stop : bool = None
    accum_grad : int = 1
    log_path : str = '/logs'
    root_dir : str = '/checkpoints'
    epoches : int = None
    log_steps : int = 500
    version : str = 'v0.1'
    share_emb_prj_weight : bool = True
    batch_size : int = 32
    # model
    model_name_or_path : str = None
    cache_dir : str = None
    #eval
    eval_batch_size : int = 16
    # optimizer

def get_args(Args = TrainingArgs):
    """
    解析命令行参数，并返回一个所有命中的args。
    Args：
        Args: (optional) dataclass, 默认为TrainingArgs， 可以集成TrainingArgs加入个性化参数

    Return：
        train_args: datacalss

    """

    parser = argparse.ArgumentParser()

    for arg, content in Args.__dataclass_fields__.items():
        if content.type == bool:
            parser.add_argument('--'+arg, action='store_true')
            continue
        parser.add_argument('--'+arg, type=content.type, default=content.default)
    
    args = parser.parse_args()
    args=vars(args)
    train_args = Args(**args)

    return train_args

def parse_hf_args(Args, json=None):
    """从命令行解析参数，并返回Hf的TrainingArguments对象。

    """
    parser = HfArgumentParser(Args)
    if json is not None:
        args = parser.parse_json_file(json_file=json)
    else:
        args=parser.parse_args_into_dataclasses()
    return args

def convert_lyc_arguments_to_hf(Args: TrainingArgs, **kwargs):
    return TrainingArguments(
        
    )

def get_base_hf_args(
    output_dir,
    **kwargs
):
    """[summary]

    Args:
        output_dir (str): [description]
        evaluation_strategy (str, optional): IntervalStrategy:['no', 'steps', 'epoch]. Defaults to 'no'.
        train_batch_size (int, optional): [description]. Defaults to 8.
        eval_batch_size (int, optional): [description]. Defaults to 8.
        gradient_accumulation_steps (int, optional): [description]. Defaults to 1.
        eval_accumulation_steps (int, optional): [description]. Defaults to None.
        lr (float, optional): [description]. Defaults to 5e-05.
        weight_decay (float, optional): [description]. Defaults to 1e-5.
        max_grad_norm (float, optional): 梯度剪裁，防止爆炸. Defaults to 1.0.
        epochs (float, optional): [description]. Defaults to 3.0.
        max_steps (int, optional): [description]. Defaults to -1.
        lr_scheduler_type (str, optional): 
                可选的有['linear', 'cosine', 'polynomial', 'constant', 'constant_whti_warmup', 'consine_with_restart'].
                Defaults to 'linear'.
        warmup_ratio (float, optional): 花费总步数中的多少用于warm_up，小数. Defaults to 0.0.
        warmup_steps (int, optional): 花费多少步数来warm_up. Defaults to 0.
        logging_dir (str, optional): [description]. Defaults to './logs'.
        logging_strategy ([type], optional): 可选的有['no','epoch','steps'].
                其中 'steps' 选项与logging_steps配合使用。 Defaults to None.
        logging_steps (int, optional): [description]. Defaults to 500.
        save_strategy ([type], optional): 可选的有['no','epoch','steps'].
                其中 'steps' 选项与save_steps配合使用。. Defaults to None.
        save_steps (int, optional): [description]. Defaults to 500.
        save_total_limit (int, optional): 最多保存多少个，删除更旧的checkpoints. Defaults to 3.
        seed (int, optional): [description]. Defaults to 1111.
        eval_steps (int, optional): [description]. Defaults to 500.
        dataloader_num_workers (int, optional): [description]. Defaults to 0.
        run_name (str, optional): 好像没用，待测试. Defaults to '.'.

        --- 以下，除特殊情况外无用 ---
        ignore_data_skip (bool, optional): 重启训练的时候要不要跳过已训练过的数据. Defaults to False.
        skip_memory_metrics (bool, optional): 要不要log内存占用情况. Defaults to False.
        report_to ([type], optional): 使用哪些logging模块. Defaults to None.
        length_column_name ([type], optional): [description]. Defaults to length.
        group_by_length (bool, optional): [description]. Defaults to True.
        past_index (int, optional): [description]. Defaults to -1.
        load_best_model_at_end (bool, optional): 与metric_for_best_model配合使用. Defaults to False.
        metric_for_best_model ([type], optional): 用哪个指标比较model checkpoints，默认是evaluation loss. Defaults to None.
        greater_is_better ([type], optional): 指标是否越大越好. Defaults to None.
    """
    defaults_args={
        "evaluation_strategy":'no',
        "train_batch_size":8,
        "eval_batch_size":8,
        "gradient_accumulation_steps":1,
        "eval_accumulation_steps":1,
        "lr": 5e-05,
        "weight_decay":1e-5,
        "max_grad_norm":1.0,
        "epochs":3.0,
        "max_steps":-1,
        "lr_scheduler_type":'linear',
        "warmup_ratio":0.0,
        "warmup_steps":0,
        "logging_dir":'./logs',
        "logging_strategy":'steps',
        "logging_steps":500,
        "save_strategy":'epoch',
        "save_steps":500,
        "save_total_limit":3,
        "seed":1111,
        "eval_steps":500,
        "dataloader_num_workers":0,
        "run_name":None,
        "ignore_data_skip":False,
        "skip_memory_metrics":False,
        "report_to":None,
        "length_column_name":'length',
        "group_by_length":True,
        "past_index":-1,
        "load_best_model_at_end":False,
        "metric_for_best_model":None,
        "greater_is_better":None
    }
    defaults_args.update(kwargs)
    defaults_args['learning_rate']=defaults_args['lr']
    defaults_args['per_device_train_batch_size']=defaults_args['train_batch_size']
    defaults_args['per_device_eval_batch_size']=defaults_args['eval_batch_size']
    defaults_args['num_train_epochs']=defaults_args['epochs']
    del defaults_args['lr'], defaults_args['train_batch_size'], defaults_args['eval_batch_size'], defaults_args['epochs']

    return TrainingArguments(output_dir, **defaults_args)

class HfTrainer(Trainer):
    def _get_eval_sampler(self, eval_dataset):
        if isinstance(self.eval_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.eval_dataset, collections.abc.Sized
        ):
            return None
        if is_torch_tpu_available():
            return SequentialDistributedSampler(eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        elif is_sagemaker_mp_enabled():
            return SequentialDistributedSampler(
                eval_dataset,
                num_replicas=smp.dp_size(),
                rank=smp.dp_rank(),
                batch_size=self.args.per_device_eval_batch_size,
            )
        elif self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)