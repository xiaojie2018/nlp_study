# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/9/18 12:38
# software: PyCharm
import logging
import os
from itertools import product
import torch
from helper import prepare_doc_batch_dict, init_logger, set_seed
from model import Doc2EDAGModel
import torch.nn.parallel as para
import torch.distributed as dist
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange, tqdm
import collections.abc as container_abcs
import random
from tensorboardX import SummaryWriter
from tools import measure_dee_prediction
import json
init_logger()
logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None,
                 train_examples=None, test_examples=None, dev_examples=None,
                 train_features=None, test_features=None, dev_features=None,
                 parallel_decorate=True):

        self.args = args

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.dev_examples = dev_examples
        self.train_features = train_features
        self.test_features = test_features
        self.dev_features = dev_features

        self._check_setting_validity()

        self.custom_collate_fn = prepare_doc_batch_dict
        # self.custom_collate_fn = None
        set_seed(self.args)

        if not self.args.use_token_role:
            assert self.args.model_type == "Doc2EDAG"
            assert self.args.add_greedy_dec is False
            self.args.num_entity_labels = 3
        else:
            self.args.num_entity_labels = len(self.args.entity_label_list)

        self.model = Doc2EDAGModel(self.args)
        self.model_class = Doc2EDAGModel

        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.args.n_gpu = torch.cuda.device_count()
        self.args.n_gpu = 1
        self.n_gpu = self.args.n_gpu

        self._decorate_model(parallel_decorate=parallel_decorate)
        self.model.to(self.device)

        self.min_teacher_prob = None
        self.teacher_norm = None
        self.teacher_cnt = None
        self.teacher_base = None
        self.reset_teacher_prob()
        self.event_type_fields_pairs = args.event_type_fields_pairs
        self.entity_label_list = args.entity_label_list

    def _init_device(self):
        logger.info('='*20 + 'Init Device' + '='*20)

        # set device
        if self.args.local_rank == -1 or self.args.no_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            self.device = torch.device("cuda", self.args.local_rank)
            self.n_gpu = 1
            if self.args.fp16:
                logger.info("16-bits training currently not supported in distributed training")
                self.args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
        logger.info("device {} n_gpu {} distributed training {}".format(
            self.device, self.n_gpu, self.in_distributed_mode()
        ))

    def _check_setting_validity(self):
        logging.info('='*20 + 'Check Setting Validity' + '='*20)
        logging.info('Setting: {}'.format(
            json.dumps(self.args.__dict__, ensure_ascii=False, indent=2)
        ))

        # check valid grad accumulate step
        if self.args.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.args.gradient_accumulation_steps))
        # reset train batch size
        self.args.train_batch_size = int(self.args.train_batch_size / self.args.gradient_accumulation_steps)

        # # check output dir
        # if os.path.exists(self.args.output_dir) and os.listdir(self.args.output_dir):
        #     logging.info("Output directory ({}) already exists and is not empty.".format(self.args.output_dir),
        #                  level=logging.WARNING)
        # os.makedirs(self.args.output_dir, exist_ok=True)
        #
        # # check model dir
        # if os.path.exists(self.args.model_dir) and os.listdir(self.args.model_dir):
        #     logging.info("Model directory ({}) already exists and is not empty.".format(self.args.model_dir),
        #                  level=logging.WARNING)
        # os.makedirs(self.args.model_dir, exist_ok=True)

    def in_distributed_mode(self):
        return self.args.local_rank >= 0

    def reset_teacher_prob(self):
        self.min_teacher_prob = self.args.min_teacher_prob
        if self.train_dataset is None:
            num_step_per_epoch = 500
        else:
            num_step_per_epoch = int(len(self.train_dataset)/self.args.train_batch_size)

        self.teacher_norm = num_step_per_epoch*self.args.schedule_epoch_length
        self.teacher_base = num_step_per_epoch*self.args.schedule_epoch_start
        self.teacher_cnt = 0

    def get_teacher_prob(self, batch_inc_flag=True):
        if self.teacher_cnt < self.teacher_base:
            prob = 1
        else:
            prob = max(
                self.min_teacher_prob, (self.teacher_norm - self.teacher_cnt + self.teacher_base) / self.teacher_norm
            )

        if batch_inc_flag:
            self.teacher_cnt += 1

        return prob

    def _decorate_model(self, parallel_decorate=True):
        logger.info('='*20 + 'Decorate Model' + '='*20)

        if self.args.fp16:
            self.model.half()

        self.model.to(self.device)
        logger.info('Set model device to {}'.format(str(self.device)))

        if parallel_decorate:
            if self.in_distributed_mode():
                self.model = para.DistributedDataParallel(self.model,
                                                          device_ids=[self.args.local_rank],
                                                          output_device=self.args.local_rank)
                logger.info('Wrap distributed data parallel')
                # logger.info('In Distributed Mode, but do not use DistributedDataParallel Wrapper')
            elif self.n_gpu > 1:
                self.model = para.DataParallel(self.model)
                logger.info('Wrap data parallel')
        else:
            logger.info('Do not wrap parallel layers')

    def get_event_idx2entity_idx2field_idx(self):
        entity_idx2entity_type = {}
        for entity_idx, entity_label in enumerate(self.entity_label_list):
            if entity_label == 'O':
                entity_type = entity_label
            else:
                entity_type = entity_label[2:]

            entity_idx2entity_type[entity_idx] = entity_type

        event_idx2entity_idx2field_idx = {}
        for event_idx, (event_name, field_types) in enumerate(self.event_type_fields_pairs):
            field_type2field_idx = {}
            for field_idx, field_type in enumerate(field_types):
                field_type2field_idx[field_type] = field_idx

            entity_idx2field_idx = {}
            for entity_idx, entity_type in entity_idx2entity_type.items():
                if entity_type in field_type2field_idx:
                    entity_idx2field_idx[entity_idx] = field_type2field_idx[entity_type]
                else:
                    entity_idx2field_idx[entity_idx] = None

            event_idx2entity_idx2field_idx[event_idx] = entity_idx2field_idx

        return event_idx2entity_idx2field_idx

    def get_current_train_batch_size(self):
        if self.in_distributed_mode():
            train_batch_size = max(self.args.train_batch_size // dist.get_world_size(), 1)
        else:
            train_batch_size = self.args.train_batch_size
        return train_batch_size

    def prepare_data_loader(self, dataset, batch_size, rand_flag=True):
        if rand_flag:
            data_sampler = RandomSampler(dataset)
        else:
            data_sampler = SequentialSampler(dataset)

        if self.custom_collate_fn is None:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler,
                                    collate_fn=self.custom_collate_fn)

        return dataloader

    def prepare_dist_data_loader(self, dataset, batch_size, epoch=0):
        # prepare distributed data loader
        data_sampler = DistributedSampler(dataset)
        data_sampler.set_epoch(epoch)

        if self.custom_collate_fn is None:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=data_sampler,
                                    collate_fn=self.custom_collate_fn)
        return dataloader

    def set_batch_to_device(self, batch):
        # move mini-batch data to the proper device
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)

            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                elif isinstance(value, dict) or isinstance(value, container_abcs.Sequence):
                    batch[key] = self.set_batch_to_device(value)

            return batch
        elif isinstance(batch, container_abcs.Sequence):
            # batch = [
            #     t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch
            # ]
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(self.device))
                elif isinstance(value, dict) or isinstance(value, container_abcs.Sequence):
                    new_batch.append(self.set_batch_to_device(value))
                else:
                    new_batch.append(value)

            return new_batch
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))

    def train(self):

        train_batch_size = self.get_current_train_batch_size()
        train_dataloader = self.prepare_data_loader(self.train_dataset, self.args.train_batch_size, rand_flag=True)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        witer = SummaryWriter(log_dir=self.args.model_dir, comment="Event_extraction")

        global_step = 0
        tr_loss = 0.0
        best_mean_precision = 0.0
        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for epoch_idx in train_iterator:

            iter_desc = "Iteration"

            if self.in_distributed_mode():
                train_dataloader = self.prepare_dist_data_loader(
                    self.train_dataset, train_batch_size, epoch=epoch_idx
                )

                iter_desc = "Rank {} {}".format(dist.get_rank(), iter_desc)

            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0

            step_batch_iter = enumerate(tqdm(train_dataloader, desc=iter_desc))
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in step_batch_iter:

                batch = self.set_batch_to_device(batch)

                if self.args.use_scheduled_sampling:
                    use_gold_span = False
                    teacher_prob = self.get_teacher_prob()
                    if random.random() < teacher_prob:
                        use_gold_span = True
                else:
                    use_gold_span = True
                    teacher_prob = 1

                # inputs = {
                #     "ex_idx": batch['ex_idx'],
                #     "doc_token_ids": batch['doc_token_ids'],
                #     "doc_token_masks": batch['doc_token_masks'],
                #     "doc_token_labels": batch['doc_token_labels'],
                #     "valid_sent_num": batch['valid_sent_num'],
                #     "features": self.train_features,
                #     "use_gold_span": use_gold_span,
                #     "teacher_prob": teacher_prob
                # }

                loss = self.model(batch, self.train_features, use_gold_span=use_gold_span, teacher_prob=teacher_prob)

                if self.n_gpu > 1:
                    loss = loss.mean()

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                loss_scalar = loss.item()

                tr_loss += loss_scalar

                nb_tr_examples += self.args.train_batch_size
                nb_tr_steps += 1
                global_step += 1

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    # global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        _, eval_results = self.evaluate('dev')

                        eval_info = eval_results[-1]
                        witer.add_scalar("Test/MacroPrecision", eval_info.get("MacroPrecision", 0), global_step)
                        witer.add_scalar("Test/MacroRecall", eval_info.get("MacroRecall", 0), global_step)
                        witer.add_scalar("Test/MacroF1", eval_info.get("MacroF1", 0), global_step)
                        witer.add_scalar("Test/MicroPrecision", eval_info.get("MicroPrecision", 0), global_step)
                        witer.add_scalar("Test/MicroRecall", eval_info.get("MicroRecall", 0), global_step)
                        witer.add_scalar("Test/MicroF1", eval_info.get("MicroF1", 0), global_step)

                        for (event_x, entity_x) in eval_results[:-1]:
                            witer.add_scalar("Test/{}/MacroPrecision".format(event_x.get("EventType")),
                                             event_x.get("MacroPrecision", 0), global_step)
                            witer.add_scalar("Test/{}/MacroRecall".format(event_x.get("EventType")),
                                             event_x.get("MacroRecall", 0), global_step)
                            witer.add_scalar("Test/{}/MacroF1".format(event_x.get("EventType")),
                                             event_x.get("MacroF1", 0), global_step)
                            witer.add_scalar("Test/{}/MicroPrecision".format(event_x.get("EventType")),
                                             event_x.get("MicroPrecision", 0), global_step)
                            witer.add_scalar("Test/{}/MicroRecall".format(event_x.get("EventType")),
                                             event_x.get("MicroRecall", 0), global_step)
                            witer.add_scalar("Test/{}/MicroF1".format(event_x.get("EventType")),
                                             event_x.get("MicroF1", 0), global_step)

                            for x1 in entity_x:
                                witer.add_scalar(
                                    "Test/{}/{}/Precision".format(event_x.get("EventType"), x1.get("RoleType")),
                                    x1.get("Precision", 0), global_step)
                                witer.add_scalar(
                                    "Test/{}/{}/Recall".format(event_x.get("EventType"), x1.get("RoleType")),
                                    x1.get("Recall", 0), global_step)
                                witer.add_scalar(
                                    "Test/{}/{}/F1".format(event_x.get("EventType"), x1.get("RoleType")),
                                    x1.get("F1", 0), global_step)

                        if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                            if eval_results[-1].get("MicroF1", 0) > best_mean_precision:
                                best_mean_precision = eval_results[-1].get("MicroF1", 0)
                                self.save_model()

                if 0 < self.args.max_steps < global_step:
                    # epoch_iterator.close()
                    break

                witer.add_scalar("Train/loss", tr_loss / global_step, global_step)
                lr = scheduler.get_lr()[-1]
                witer.add_scalar("Train/lr", lr, global_step)

            eval_results = self.evaluate('dev')

            eval_info = eval_results[-1]
            witer.add_scalar("Test/MacroPrecision", eval_info.get("MacroPrecision", 0), global_step)
            witer.add_scalar("Test/MacroRecall", eval_info.get("MacroRecall", 0), global_step)
            witer.add_scalar("Test/MacroF1", eval_info.get("MacroF1", 0), global_step)
            witer.add_scalar("Test/MicroPrecision", eval_info.get("MicroPrecision", 0), global_step)
            witer.add_scalar("Test/MicroRecall", eval_info.get("MicroRecall", 0), global_step)
            witer.add_scalar("Test/MicroF1", eval_info.get("MicroF1", 0), global_step)

            for (event_x, entity_x) in eval_results[:-1]:
                witer.add_scalar("Test/{}/MacroPrecision".format(event_x.get("EventType")),
                                 event_x.get("MacroPrecision", 0), global_step)
                witer.add_scalar("Test/{}/MacroRecall".format(event_x.get("EventType")),
                                 event_x.get("MacroRecall", 0), global_step)
                witer.add_scalar("Test/{}/MacroF1".format(event_x.get("EventType")),
                                 event_x.get("MacroF1", 0), global_step)
                witer.add_scalar("Test/{}/MicroPrecision".format(event_x.get("EventType")),
                                 event_x.get("MicroPrecision", 0), global_step)
                witer.add_scalar("Test/{}/MicroRecall".format(event_x.get("EventType")),
                                 event_x.get("MicroRecall", 0), global_step)
                witer.add_scalar("Test/{}/MicroF1".format(event_x.get("EventType")),
                                 event_x.get("MicroF1", 0), global_step)

                for x1 in entity_x:
                    witer.add_scalar(
                        "Test/{}/{}/Precision".format(event_x.get("EventType"), x1.get("RoleType")),
                        x1.get("Precision", 0), global_step)
                    witer.add_scalar(
                        "Test/{}/{}/Recall".format(event_x.get("EventType"), x1.get("RoleType")),
                        x1.get("Recall", 0), global_step)
                    witer.add_scalar(
                        "Test/{}/{}/F1".format(event_x.get("EventType"), x1.get("RoleType")),
                        x1.get("F1", 0), global_step)

            if eval_results[-1].get("MicroF1", 0) > best_mean_precision:
                best_mean_precision = eval_results[-1].get("MicroF1", 0)
                self.save_model()

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss/global_step

    def evaluate(self, data_type):

        if self.args.model_type == 'DCFEE':
            eval_task = product(['dev', 'test'], [False, True], ['DCFEE-0', 'DCFEE-M'])
        else:
            if self.args.add_greedy_dec:
                eval_task = product(['dev', 'test'], [False, True], ['GreedyDec', None])
            else:
                eval_task = product(['dev', 'test'], [False, True], [None])

        for task_idx, (data_type, gold_span_flag, heuristic_type) in enumerate(eval_task):
            if self.in_distributed_mode() and task_idx % dist.get_world_size() != dist.get_rank():
                continue

            if data_type == 'dev':
                features = self.dev_features
                dataset = self.dev_dataset
            elif data_type == 'test':
                features = self.test_features
                dataset = self.test_dataset
            else:
                raise Exception("Unsupported data type {}".format(data_type))

            if gold_span_flag:
                span_str = 'gold_span'
            else:
                span_str = 'pred_span'

            if heuristic_type is None:
                model_str = self.args.cpt_file_name.replace('.', '~')
            else:
                model_str = heuristic_type

            # eval
            # 重要参数： dataset feature reduce_info_type reduce_info_type=[none, mean], heuristic_type, use_gold_span
            reduce_info_type = 'none'
            eval_dataloader = self.prepare_data_loader(dataset, self.args.eval_batch_size, rand_flag=False)

            self.model.eval()
            iter_desc = "Iteration"

            if self.in_distributed_mode():
                iter_desc = "Rank {} {}".format(dist.get_rank(), iter_desc)

            total_info = []
            for step, batch in enumerate(tqdm(eval_dataloader, desc=iter_desc)):
                batch = self.set_batch_to_device(batch)

                # use_gold_span = False
                if heuristic_type is None:
                    event_idx2entity_idx2field_idx = None
                else:
                    event_idx2entity_idx2field_idx = self.get_event_idx2entity_idx2field_idx()

                with torch.no_grad():
                    batch_info = self.model(
                        batch, features, use_gold_span=False, train_flag=False,
                        event_idx2entity_idx2field_idx=event_idx2entity_idx2field_idx, heuristic_type=heuristic_type
                    )

                if isinstance(batch_info, torch.Tensor):
                    total_info.append(batch_info.to(torch.device('cpu')))
                else:
                    total_info.extend(batch_info)

            if isinstance(total_info[0], torch.Tensor):
                # transform event_info to torch.Tensor
                total_info = torch.cat(total_info, dim=0)

            # [batch_size, ...] -> [...]
            if reduce_info_type.lower() == 'sum':
                reduced_info = total_info.sum(dim=0)
            elif reduce_info_type.lower() == 'mean':
                reduced_info = total_info.mean(dim=0)
            elif reduce_info_type.lower() == 'none':
                reduced_info = total_info
            else:
                raise Exception('Unsupported reduce metric type {}'.format(reduce_info_type))

            total_event_decode_results = reduced_info
            dump_eval_json_path = None

            total_eval_res = measure_dee_prediction(
                self.event_type_fields_pairs, features, total_event_decode_results,
                dump_json_path=dump_eval_json_path
            )

            # 打印
            for s in total_eval_res[:-1]:
                logger.info(s[0])
                for s1 in s:
                    logger.info(s1)
            logger.info(total_eval_res[-1])

            return total_event_decode_results, total_eval_res

    def evaluate_test(self, dataset, features):
        pass

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
