

# 训练步骤

## load_data

- sqlent/utils.py ---> load_data(sql_paths, table_paths, use_small=False)
    - sql_paths: 训练数据文件
    - tabel_paths: 存表数据文件

   output: ret_sql_data, table_data

        dev_sql, dev_table = load_data(data_dirs['val']['data'], data_dirs['val']['tables'], use_small=use_small)
        dev_db = data_dirs['val']['db']
        if mode == 'train':
            train_sql, train_table = load_data(data_dirs['train']['data'], data_dirs['train']['tables'], use_small=use_small)
            train_db = data_dirs['train']['db']
            return train_sql, train_table, train_db, dev_sql, dev_table, dev_db
        elif mode == 'test':
            test_sql, test_table = load_data(data_dirs['test']['data'], data_dirs['test']['tables'], use_small=use_small)
            test_db = data_dirs['test']['db']
                return dev_sql, dev_table, dev_db, test_sql, test_table, test_db


- train_bert.py 

- sqlent/utils.py  --->  epoch_train(model, optimizer, batch_size, sql_data, table_data, tokenizer=None)

    - train_loss = epoch_train(model, optimizer, batch_size, train_sql, train_table, tokenizer=tokenizer)


        # 进行编码  tokenizer
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, header_type = to_batch_seq(sql_data,
                                                                                              table_data,
                                                                                              perm,
                                                                                              st,
                                                                                              ed,
																							  tokenizer=tokenizer)










