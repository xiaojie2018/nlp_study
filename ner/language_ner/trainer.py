# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 18:06
# software: PyCharm

import os
import logging
from tqdm import tqdm, trange
import json
import codecs
from adversarial import FGM
from mertics import name_entity_recognition_metric
from utils import set_seed, collate_fn, jiexi, bert_extract_item, get_extract_item
from config import MODEL_CLASSES, MODEL_TASK
from model import LanguageSoftmaxForNer, LanguageCrfForNer, LanguageSpanForNer
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
import copy
logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, train_examples=None, test_examples=None, dev_examples=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.train_examples = train_examples
        self.test_examples = test_examples
        self.dev_examples = dev_examples

        self.ten_labels = set()

        for x in self.args.labels:
            if x.startswith("B-") or x.startswith("I-"):
                self.ten_labels.add(x[2:])

        self.ten_labels = sorted(list(self.ten_labels))

        # self.config_class, _, _ = MODEL_CLASSES[args.model_type]
        self.model_class = MODEL_TASK[args.model_decode_fc]
        # self.config = self.config_class.from_pretrained(args.model_name_or_path)
        self.model = self.model_class(args.model_dir, args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)
        self.args.n_gpu = torch.cuda.device_count()
        self.args.n_gpu = 1

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size,)
                                      # collate_fn=collate_fn)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        if self.args.model_decode_fc == 'softmax':
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            self.args.warmup_steps = int(t_total * self.args.warmup_proportion)
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        elif self.args.model_decode_fc == 'crf':
            no_decay = ["bias", "LayerNorm.weight"]
            bert_param_optimizer = list(self.model.bert.named_parameters())
            crf_param_optimizer = list(self.model.crf.named_parameters())
            linear_param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [
                {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate},
                {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': self.args.learning_rate},

                {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay, 'lr': self.args.crf_learning_rate},
                {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': self.args.crf_learning_rate},

                {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay, 'lr': self.args.crf_learning_rate},
                {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 'lr': self.args.crf_learning_rate}
            ]
            self.args.warmup_steps = int(t_total * self.args.warmup_proportion)
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.args.warmup_steps,
                                                        num_training_steps=t_total)

        elif self.args.model_decode_fc == 'span':
            no_decay = ["bias", "LayerNorm.weight"]
            bert_parameters = self.model.bert.named_parameters()
            start_parameters = self.model.start_fc.named_parameters()
            end_parameters = self.model.end_fc.named_parameters()
            optimizer_grouped_parameters = [
                {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.args.weight_decay, 'lr': self.args.learning_rate},
                {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                    , 'lr': self.args.learning_rate},

                {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.args.weight_decay, 'lr': self.args.start_learning_rate},
                {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                    , 'lr': self.args.start_learning_rate},

                {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
                 "weight_decay": self.args.weight_decay, 'lr': self.args.end_learning_rate},
                {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
                    , 'lr': self.args.end_learning_rate},
            ]
            # optimizer_grouped_parameters = [
            #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            #      "weight_decay": args.weight_decay, },
            #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            # ]
            self.args.warmup_steps = int(t_total * self.args.warmup_proportion)
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.args.warmup_steps,
                                                        num_training_steps=t_total)
        else:
            raise Exception("Some model files might be missing ... ")

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        witer = SummaryWriter(logdir=self.args.model_dir, comment="BERT_intent")

        global_step = 0
        tr_loss = 0.0
        best_mean_precision = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)

        # if self.args.do_adv:
        #     fgm = FGM(self.model, emb_name=self.args.adv_name, epsilon=self.args.adv_epsilon)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                if self.args.model_decode_fc in ['softmax', 'crf']:

                    inputs = {"input_ids": batch[0],
                              "attention_mask": batch[1],
                              "token_type_ids": batch[2],
                              "label": batch[3]}

                    outputs = self.model(**inputs)
                    loss = outputs[0]

                elif self.args.model_decode_fc == 'span':
                    inputs = {"input_ids": batch[0],
                              "attention_mask": batch[1],
                              "token_type_ids": batch[2],
                              "start_positions": batch[3],
                              "end_positions": batch[4]}

                    outputs = self.model(**inputs)
                    loss = outputs[0]

                # eval_results = self.evaluate('dev')
                # print(1)

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                # if self.args.do_adv:
                #     fgm.attack()
                #     loss_adv = self.model(**inputs)[0]
                #     if self.args.n_gpu > 1:
                #         loss_adv = loss_adv.mean()
                #     loss_adv.backward()
                #     fgm.restore()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        eval_results = self.evaluate('dev')

                        witer.add_scalar("Test/loss", eval_results.get("loss", 0), global_step)
                        witer.add_scalar("Test/total/precision", eval_results.get('total', {}).get("precision", 0), global_step)
                        witer.add_scalar("Test/total/recall", eval_results.get('total', {}).get("recall", 0), global_step)
                        witer.add_scalar("Test/total/f1", eval_results.get('total', {}).get("f1", 0), global_step)

                        for x in self.ten_labels:
                            witer.add_scalar(f"Test/{x}/precision", eval_results.get(x, {}).get("precision", 0), global_step)
                            witer.add_scalar(f"Test/{x}/recall", eval_results.get(x, {}).get("recall", 0), global_step)
                            witer.add_scalar(f"Test/{x}/f1", eval_results.get(x, {}).get("f1", 0), global_step)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        if eval_results.get("total", {}).get("f1", 0) > best_mean_precision:
                            best_mean_precision = eval_results.get("total", {}).get("f1", 0)
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                witer.add_scalar("Train/loss", tr_loss/global_step, global_step)
                lr = scheduler.get_lr()[-1]
                witer.add_scalar("Train/lr", lr, global_step)

            eval_results = self.evaluate('dev')
            witer.add_scalar("Test/loss", eval_results.get("loss", 0), global_step)
            witer.add_scalar("Test/total/precision", eval_results.get('total', {}).get("precision", 0), global_step)
            witer.add_scalar("Test/total/recall", eval_results.get('total', {}).get("recall", 0), global_step)
            witer.add_scalar("Test/total/f1", eval_results.get('total', {}).get("f1", 0), global_step)

            for x in self.ten_labels:
                witer.add_scalar(f"Test/{x}/precision", eval_results.get(x, {}).get("precision", 0), global_step)
                witer.add_scalar(f"Test/{x}/recall", eval_results.get(x, {}).get("recall", 0), global_step)
                witer.add_scalar(f"Test/{x}/f1", eval_results.get(x, {}).get("f1", 0), global_step)

            if eval_results.get("total", {}).get("f1", 0) > best_mean_precision:
                best_mean_precision = eval_results.get("total", {}).get("f1", 0)
            self.save_model()

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode, prefix=""):
        # metric = SeqEntityScore(args.id2label, markup=args.markup)

        if mode == 'test':
            eval_dataset = self.test_dataset
        elif mode == 'dev':
            eval_dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        if self.args.model_decode_fc in ['softmax', 'crf']:
            eval_dataloader = DataLoader(eval_dataset, batch_size=self.args.train_batch_size)
        # Eval!
        logger.info("***** Running evaluation %s *****", prefix)
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.args.train_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        preds = None
        out_label_ids = None
        out_lens = None

        start_preds = None
        end_preds = None
        out_start_ids = None
        out_end_ids = None

        if self.args.model_decode_fc in ['softmax', 'crf']:

            epoch_iterator = tqdm(eval_dataloader, desc="eval_Iteration")

            for step, batch in enumerate(epoch_iterator):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                with torch.no_grad():
                    inputs = {"input_ids": batch[0],
                              "attention_mask": batch[1],
                              "token_type_ids": batch[2],
                              "label": batch[3],
                              "is_test": True}

                    outputs = self.model(**inputs)

                if self.args.model_decode_fc == 'softmax':
                    tmp_eval_loss, logits = outputs[:2]
                elif self.args.model_decode_fc == 'crf':
                    tmp_eval_loss, logits = outputs[:2]
                    logits = logits.squeeze(0)

                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
                eval_loss += tmp_eval_loss.item()
                nb_eval_steps += 1

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = inputs['label'].detach().cpu().numpy()
                    out_lens = batch[4].detach().cpu().numpy()

                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, inputs['label'].detach().cpu().numpy(), axis=0)
                    out_lens = np.append(out_lens, batch[4].detach().cpu().numpy())

        elif self.args.model_decode_fc == 'span':

            features = eval_dataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
            all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
            all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids,
                                    all_input_lens)

            subjects = [f.subjects for f in features]

            eval_dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size)

            epoch_iterator = tqdm(eval_dataloader, desc='eval_Iteration')

            for step, batch in enumerate(epoch_iterator):

                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                with torch.no_grad():
                    inputs = {"input_ids": batch[0],
                              "attention_mask": batch[1],
                              "token_type_ids": batch[2],
                              "start_positions": batch[3],
                              "end_positions": batch[4]}

                    outputs = self.model(**inputs)

            tmp_eval_loss, start_logits, end_logits = outputs[:3]

            if self.args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

            tmp_eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            if start_preds is None:
                start_preds = start_logits.detach().cpu().numpy()
                end_preds = end_logits.detach().cpu().numpy()

                out_start_ids = inputs['start_positions'].detach().cpu().numpy()
                out_end_ids = inputs['end_positions'].detach().cpu().numpy()

                out_lens = batch[5].detach().cpu().numpy()

            else:
                start_preds = np.append(start_preds, start_logits.detach().cpu().numpy(), axis=0)
                end_preds = np.append(end_preds, end_logits.detach().cpu().numpy(), axis=0)

                out_start_ids = np.append(out_start_ids, inputs['start_positions'].detach().cpu().numpy(), axis=0)
                out_end_ids = np.append(out_end_ids, inputs['end_positions'].detach().cpu().numpy(), axis=0)

                out_lens = np.append(out_lens, batch[5].detach().cpu().numpy())

            # preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()

            # out_label_ids = inputs['label'].cpu().numpy().tolist()
            # input_lens = batch[4].cpu().numpy().tolist()
            #
            # for i, label in enumerate(out_label_ids):
            #     temp_1 = []
            #     temp_2 = []
            #     for j, m in enumerate(label):
            #         if j == 0:
            #             continue
            #         elif j == input_lens[i] - 1:
            #             metric.update(pred_paths=[temp_2], label_paths=[temp_1])
            #             break
            #         else:
            #             temp_1.append(self.args.id_label[out_label_ids[i][j]])
            #             temp_2.append(preds[i][j])

        if self.args.model_decode_fc in ['softmax', 'crf']:
            if self.args.model_decode_fc == "softmax":
                preds = np.argmax(preds, axis=2).tolist()
            elif self.args.model_decode_fc == 'crf':
                preds = preds.tolist()
            out_label_ids = out_label_ids.tolist()
            out_lens = out_lens.tolist()

            r_ps = []
            r_os = []
            for x, y, z, t in zip(preds, out_label_ids, out_lens, self.dev_examples):
                x1 = [self.args.id_label[x[i]] for i in range(1, z-1)]
                y1 = [self.args.id_label[y[i]] for i in range(1, z-1)]
                words0 = t.text[:len(x1)]
                r_p = jiexi(words0, x1)
                r_o = jiexi(words0, y1)
                r_ps.append({'text': "".join(t.text), "entities": r_p})
                r_os.append({'text': "".join(t.text), "entities": r_o})

        elif self.args.model_decode_fc == 'span':
            start_preds = start_preds.tolist()
            end_preds = end_preds.tolist()

            r_ps = []
            r_os = []

            for ind, (y, z, t) in enumerate(zip(subjects, out_lens, self.dev_examples)):

                start_preds_i = [start_preds[ind][i] for i in range(1, z-1)]
                end_preds_i = [end_preds[ind][i] for i in range(1, z - 1)]

                S = get_extract_item(start_preds_i, end_preds_i)

                r_p = []
                for y1 in S:
                    r_p.append({
                        "entity_type": self.args.id_label[y1[0]],
                        "start_pos": y1[1],
                        "end_pos": y1[2] + 1,
                        "word": "".join(t.text[y1[1]: y1[2]+1])
                    })

                r_o = []
                for y1 in y:
                    r_o.append({
                        "entity_type": self.args.id_label[y1[0]],
                        "start_pos": y1[1],
                        "end_pos": y1[2] + 1,
                        "word": "".join(t.text[y1[1]: y1[2] + 1])
                    })

                r_ps.append({"text": "".join(t.text), "entities": r_p})
                r_os.append({"text": "".join(t.text), "entities": r_o})

        rpt_info, cf_mat_df, pred_std_cmp_ner_infos, entity_type_list = name_entity_recognition_metric(r_ps, r_os)

        entity_info = copy.deepcopy(rpt_info)
        logger.info("\n")
        eval_loss = eval_loss / nb_eval_steps
        # eval_info, entity_info = metric.result()
        # results = {f'{key}': value for key, value in eval_info.items()}
        results = rpt_info
        results['loss'] = eval_loss
        # logger.info("***** Eval results %s *****", prefix)
        # info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        # logger.info(info)
        info = f'----loss----: {eval_loss:.4f} -----'
        logger.info(info)
        logger.info("***** Entity results %s *****", prefix)
        for key in sorted(entity_info.keys()):
            logger.info("******* %s results ********" % key)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            logger.info(info)
        return results

    def evaluate_test(self, dataset):

        if self.args.model_decode_fc in ['softmax', 'crf']:
            eval_dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size)

        # Eval!
        preds = None
        out_label_ids = None
        out_lens = None

        start_preds = None
        end_preds = None

        if self.args.model_decode_fc in ['softmax', 'crf']:

            for step, batch in enumerate(eval_dataloader):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                with torch.no_grad():
                    inputs = {"input_ids": batch[0],
                              "attention_mask": batch[1],
                              "token_type_ids": batch[2],}
                              # "label": batch[3],
                              # "is_test": True}

                    outputs = self.model(**inputs)

                if self.args.model_decode_fc == 'softmax':
                    tmp_eval_loss, logits = outputs[:2]
                elif self.args.model_decode_fc == 'crf':
                    tmp_eval_loss, logits = outputs[:2]
                    logits = logits.squeeze(0)

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_lens = batch[4].detach().cpu().numpy()

                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_lens = np.append(out_lens, batch[4].detach().cpu().numpy())

            if self.args.model_decode_fc == "softmax":
                preds = np.argmax(preds, axis=2).tolist()
            elif self.args.model_decode_fc == 'crf':
                preds = preds.tolist()

            out_lens = out_lens.tolist()

            r_ps = []
            for x, z, t in zip(preds, out_lens, self.dev_examples):
                x1 = [self.args.id_label[x[i]] for i in range(1, z-1)]
                words0 = t.text[:len(x1)]
                r_p = jiexi(words0, x1)
                r_ps.append({'text': "".join(t.text), "entities": r_p})

            return r_ps

        elif self.args.model_decode_fc == 'span':

            features = dataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
            all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
            all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids,
                                    all_input_lens)

            eval_dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size)

            for step, batch in enumerate(eval_dataloader):
                self.model.eval()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                with torch.no_grad():
                    inputs = {"input_ids": batch[0],
                              "attention_mask": batch[1],
                              "token_type_ids": batch[2]}

                    outputs = self.model(**inputs)

            start_logits, end_logits = outputs[:2]

            if start_preds is None:
                start_preds = start_logits.detach().cpu().numpy()
                end_preds = end_logits.detach().cpu().numpy()

                out_lens = batch[5].detach().cpu().numpy()

            else:
                start_preds = np.append(start_preds, start_logits.detach().cpu().numpy(), axis=0)
                end_preds = np.append(end_preds, end_logits.detach().cpu().numpy(), axis=0)

                out_lens = np.append(out_lens, batch[5].detach().cpu().numpy())

            start_preds = start_preds.tolist()
            end_preds = end_preds.tolist()

            r_ps = []

            for ind, (z, t) in enumerate(zip(out_lens, self.dev_examples)):

                start_preds_i = [start_preds[ind][i] for i in range(1, z - 1)]
                end_preds_i = [end_preds[ind][i] for i in range(1, z - 1)]

                S = get_extract_item(start_preds_i, end_preds_i)

                r_p = []
                for y1 in S:
                    r_p.append({
                        "entity_type": self.args.id_label[y1[0]],
                        "start_pos": y1[1],
                        "end_pos": y1[2] + 1,
                        "word": "".join(t.text[y1[1]: y1[2] + 1])
                    })

                r_ps.append({"text": "".join(t.text), "entities": r_p})

            return r_ps

        else:
            return []

    def save_model(self, step=0):
        # Save model checkpoint (Overwrite)
        model_file_path = self.args.model_dir

        if not os.path.exists(model_file_path):
            os.makedirs(model_file_path)

        # with codecs.open(os.path.join(model_file_path, '{}_config.json'.format(self.args.task_type)), 'w', encoding='utf-8') as fd:
        #     json.dump(vars(self.args), fd, indent=4, ensure_ascii=False)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(model_file_path)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(model_file_path, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", model_file_path)

    def load_model(self, model_file_path):
        # Check whether model exists
        if not os.path.exists(model_file_path):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(model_file_path, args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

