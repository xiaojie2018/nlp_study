# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/6/22 18:06
# software: PyCharm

import os
import logging
from tqdm import tqdm, trange
from utils import set_seed, compute_metrics
# from mertics import text_multi_label_classification_evaluate
# from config import MODEL_CLASSES, MODEL_TASK
from model import ClassificationModel
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        # self.config_class, _, _ = MODEL_CLASSES[args.model_type]
        # self.model_class = MODEL_TASK[args.task_type]
        self.model_class = ClassificationModel
        # self.config = self.config_class.from_pretrained(args.model_name_or_path)
        self.model = self.model_class(args.model_dir, args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)

        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

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
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

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

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'attention_mask_a': batch[3],
                          'attention_mask_b': batch[4],
                          'label': batch[5]}
                outputs = self.model(**inputs)
                loss = outputs[0]
                # eval_results = self.evaluate('dev')
                # print(1)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

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
                        witer.add_scalar("Test/mean", eval_results.get('mean', {}).get("mean_precision", 0),
                                         global_step)
                        witer.add_scalar("Test/mean", eval_results.get('mean', {}).get("mean_recall", 0), global_step)
                        witer.add_scalar("Test/mean", eval_results.get('mean', {}).get("mean_f1-score", 0), global_step)
                        witer.add_scalar("Test/sum", eval_results.get('sum', {}).get("sum_precision", 0), global_step)
                        witer.add_scalar("Test/sum", eval_results.get('sum', {}).get("sum_recall", 0), global_step)
                        witer.add_scalar("Test/sum", eval_results.get('sum', {}).get("sum_f1-score", 0), global_step)
                    # labels = ["0", "10001", "10002"]
                    # for k in labels:
                    #     witer.add_scalar("Test/{}".format(k), eval_results.get(k, {}).get("precision", 0), global_step)
                    #     witer.add_scalar("Test/{}".format(k), eval_results.get(k, {}).get("recall", 0), global_step)
                    #     witer.add_scalar("Test/{}".format(k), eval_results.get(k, {}).get("f1", 0), global_step)

                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        if eval_results.get("mean", {}).get("mean_f1-score", 0) > best_mean_precision:
                            best_mean_precision = eval_results.get("mean", {}).get("mean_f1-score", 0)
                            self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                witer.add_scalar("Train/loss", tr_loss/global_step, global_step)
                lr = scheduler.get_lr()[-1]
                witer.add_scalar("Train/lr", lr, global_step)

            eval_results = self.evaluate('dev')
            witer.add_scalar("Test/loss", eval_results.get("loss", 0), global_step)
            witer.add_scalar("Test/mean", eval_results.get('mean', {}).get("mean_precision", 0), global_step)
            witer.add_scalar("Test/mean", eval_results.get('mean', {}).get("mean_recall", 0), global_step)
            witer.add_scalar("Test/mean", eval_results.get('mean', {}).get("mean_f1-score", 0), global_step)
            witer.add_scalar("Test/sum", eval_results.get('sum', {}).get("sum_precision", 0), global_step)
            witer.add_scalar("Test/sum", eval_results.get('sum', {}).get("sum_recall", 0), global_step)
            witer.add_scalar("Test/sum", eval_results.get('sum', {}).get("sum_f1-score", 0), global_step)

            if eval_results.get("mean", {}).get("mean_f1-score", 0) > best_mean_precision:
                best_mean_precision = eval_results.get("mean", {}).get("mean_f1-score", 0)
                self.save_model()

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        out_intent_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'attention_mask_a': batch[3],
                          'attention_mask_b': batch[4],
                          'label': batch[5]}
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['label'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['label'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        intent_preds_list = []
        out_intent_label_list = []
        # intent_label_map = {i: label for i, label in enumerate(self.intent_label_lst)}
        intent_label_map = self.args.id_label

        for i in range(intent_preds.shape[0]):
            p1 = intent_preds[i].tolist()
            o1 = out_intent_label_ids[i].tolist()

            intent_preds_list.append(intent_label_map[p1.index(max(p1))])
            out_intent_label_list.append(intent_label_map[o1.index(max(o1))])

            # p11 = 1 if p1[0] > 0.5 else 0
            # intent_preds_list.append(p11)
            # out_intent_label_list.append(int(o1))

        total_result, report = compute_metrics(intent_preds_list, out_intent_label_list)

        logger.info(report)

        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in results.keys():
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def evaluate_test(self, dataset):
        eval_dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", "predict")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        intent_preds = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'attention_mask_a': batch[3],
                          'attention_mask_b': batch[4],
                          'label': None}
                outputs = self.model(**inputs)
                # tmp_eval_loss, (intent_logits) = outputs[:2]
                intent_logits = outputs

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

        intent_preds_list = []
        intent_preds_list_all = []
        intent_preds_list_pr = []
        intent_label_map = {int(k): v for k, v in self.args.id_label.items()}

        for i in range(intent_preds.shape[0]):
            p1 = intent_preds[i].tolist()
            intent_preds_list.append(intent_label_map[p1.index(max(p1))])
            intent_preds_list_pr.append(max(p1))
            p2 = {}
            for j, o in enumerate(p1):
                p2[intent_label_map[j]] = o
            intent_preds_list_all.append(p2)

        return intent_preds_list, intent_preds_list_pr, intent_preds_list_all

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

