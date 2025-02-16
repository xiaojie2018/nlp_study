# -*- coding:utf-8 -*-
# author: xiaojie
# datetime: 2020/5/13 14:27
# software: PyCharm


import os
import logging
from tqdm import tqdm, trange
from utils import MODEL_CLASSES, set_seed, compute_metrics, compute_metrics_link
from model import BertTwoSentenceSimilarityToEntityLink
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.config_class, _ = MODEL_CLASSES[args.model_type]
        self.model_class = BertTwoSentenceSimilarityToEntityLink
        self.config = self.config_class.from_pretrained(args.model_name_or_path)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args)

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
        best_f1 = 0.0
        eval_results = {}
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids1': batch[0],
                          'attention_mask1': batch[1],
                          'token_type_ids1': batch[2],
                          'mention_masks': batch[3],
                          'input_ids2': batch[4],
                          'attention_mask2': batch[5],
                          'token_type_ids2': batch[6],
                          'sep_masks2': batch[7],
                          'labels': batch[8]}

                outputs = self.model(**inputs)
                loss = outputs[0]
                # eval_results = self.evaluate('dev')

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

                    # witer.add_scalar("Test/loss", eval_results.get("loss", 0), global_step)
                    # labels = ["0", "10001", "10002"]
                    # for k in labels:
                    #     witer.add_scalar("Test/{}".format(k), eval_results.get(k, {}).get("precision", 0), global_step)
                    #     witer.add_scalar("Test/{}".format(k), eval_results.get(k, {}).get("recall", 0), global_step)
                    #     witer.add_scalar("Test/{}".format(k), eval_results.get(k, {}).get("f1", 0), global_step)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        if eval_results.get("f1", 0.0) > best_f1:
                            best_f1 = eval_results.get("f1", 0)
                            self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                witer.add_scalar("Train/loss", tr_loss/global_step, global_step)
                lr = scheduler.get_lr()[-1]
                witer.add_scalar("Train/lr", lr, global_step)

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            test_data = self.test_dataset
        elif mode == 'dev':
            test_data = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        test_result = []
        for td in test_data:
            o_d = td['o_data']
            kb_d = td['kb_data']
            o_kb_d = td['o_kb_data']
            pre_kb_id = "NIL"
            if kb_d:
                link_preds = self.evaluate_predict(kb_d)
                link_preds = link_preds.tolist()
                ind = link_preds.index(max(link_preds))
                pre_kb_id = o_kb_d[ind][3]
            o_d["pre_kb_id"] = pre_kb_id
            test_result.append(o_d)

        # 评估
        results = compute_metrics_link(test_result)

        logger.info("***** {} results *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def evaluate1(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

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
                inputs = {'input_ids1': batch[0],
                          'attention_mask1': batch[1],
                          'token_type_ids1': batch[2],
                          'mention_masks': batch[3],
                          'input_ids2': batch[4],
                          'attention_mask2': batch[5],
                          'token_type_ids2': batch[6],
                          'sep_masks2': batch[7],
                          'labels': batch[8]}

                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        intent_preds_list = []
        # intent_label_map = {i: label for i, label in enumerate(self.intent_label_lst)}
        intent_label_map = self.args.id_labels

        for i in range(intent_preds.shape[0]):
            intent_preds_list.append(intent_label_map[intent_preds[i]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def evaluate_predict(self, dataset):
        # if mode == 'test':
        #     dataset = self.test_dataset
        # elif mode == 'dev':
        #     dataset = self.dev_dataset
        # else:
        #     raise Exception("Only dev and test dataset available")

        # eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size)

        # Eval!
        # logger.info("***** Running evaluation on %s dataset *****", "predict")
        # logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", self.args.eval_batch_size)
        # eval_loss = 0.0
        # nb_eval_steps = 0
        link_preds = None
        # out_intent_label_ids = None

        self.model.eval()

        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids1': batch[0],
                          'attention_mask1': batch[1],
                          'token_type_ids1': batch[2],
                          'mention_masks': batch[3],
                          'input_ids2': batch[4],
                          'attention_mask2': batch[5],
                          'token_type_ids2': batch[6],
                          'sep_masks2': batch[7],
                          'labels': batch[8]}

                outputs = self.model(**inputs)

                tmp_eval_loss, link_logits = outputs[0], outputs[1]

                # eval_loss += tmp_eval_loss.mean().item()
            # nb_eval_steps += 1

            # Intent prediction
            if link_preds is None:
                link_preds = link_logits.detach().cpu().numpy()
                # out_intent_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                link_preds = np.append(link_preds, link_logits.detach().cpu().numpy(), axis=0)
                # out_intent_label_ids = np.append(
                #     out_intent_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        # eval_loss = eval_loss / nb_eval_steps
        # results = {
        #     "loss": eval_loss
        # }

        # Intent result
        # print(1)
        # link_preds = np.argmax(link_preds, axis=1)

        # intent_preds_list = []
        # # intent_label_map = {i: label for i, label in enumerate(self.intent_label_lst)}
        # intent_label_map = self.args.id_labels
        #
        # for i in range(intent_preds.shape[0]):
        #     intent_preds_list.append(intent_label_map[intent_preds[i]])
        #
        # total_result = compute_metrics(intent_preds, out_intent_label_ids)
        # results.update(total_result)
        #
        # logger.info("***** Eval results *****")
        # for key in sorted(results.keys()):
        #     logger.info("  %s = %s", key, str(results[key]))

        return link_preds

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
