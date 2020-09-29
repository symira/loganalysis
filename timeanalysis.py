# -*- coding:utf-8 -*-
import importlib
from datetime import datetime
import pymysql
import json
import matplotlib.pyplot as plt
from graphviz import Digraph
import pandas as pd
from dataprocess.turndata import Turn
import numpy as np
import os
import sys
import re
import seaborn as sns
import math




def get_botact_data():
    # botact字典
    db = pymysql.connect("10.110.147.194", "nlu", "nlu2019#421", "dm")
    cursor = db.cursor()
    sql = cursor.execute("SELECT botact_subname,botactid from botact_info;")
    botactname_info = cursor.fetchmany(sql)
    cursor.close()
    db.close()
    for tup in botactname_info:
        botactid = tup[1]
        botactname = tup[0]
        if botactid not in botactname_dic:
            botactname_dic[botactid] = botactname
        else:
            continue
    return botactname_dic


def readdir(datadir):
    """从文件目录读取数据源"""
    # datadir = r'D:\logdata\\'
    filenames = os.listdir(datadir)

    dfs = []
    for filename in filenames:
        df = pd.read_excel(os.path.join(datadir, filename),
                           names=['id', 'turn_id', 'service_id', 'user_id', 'generator_time', 'turn_json',
                                  'dialog_type'])
        dfs.append(df)
    data_df = pd.concat(dfs, axis=0, ignore_index=True)
    print(data_df.head(5))
    return data_df


def get_turn_data(datapath):
    # readExcel
    # datapath = r'D:\logdata\1.xlsx'

    data = pd.read_excel(datapath,
                         names=['id', 'turn_id', 'service_id', 'user_id', 'generator_time', 'turn_json', 'dialog_type'])
    # data = readdir()
    data = data.iloc[:, 2:]
    data_df = data.loc[np.where(pd.isna(data['dialog_type']))]
    data_df.sort_values(by="generator_time", ascending=True, inplace=True)
    count = 0
    for index, row in data_df.iterrows():
        count = count + 1
        user_id = row['user_id']
        service_id = row['service_id']
        try:
            turn_json = json.loads(row['turn_json'])
            # 存入字典
            if (user_id, service_id) not in cases:
                cases[(user_id, service_id)] = [turn_json]
            else:
                cases[(user_id, service_id)].append(turn_json)
        except ValueError:
            pass
        continue

    print("共%d条数据……" % count)
    keys = list(cases.keys())
    for k in keys:
        if len(cases[k]) < 5:
            del cases[k]
    return cases


def get_all_turn_data(path):
    # readExcel
    # datapath = r'D:\logdata\us.xls'
    #
    # data = pd.read_excel(datapath,
    #                      names=['id', 'turn_id', 'service_id', 'user_id', 'generator_time', 'turn_json', 'dialog_type'])
    data = readdir(path)
    data = data.iloc[:, 2:]
    data_df = data.loc[np.where(pd.isnull(data['dialog_type']))]
    data_df.sort_values(by="generator_time", ascending=True, inplace=True)
    count = 0
    for index, row in data_df.iterrows():
        count = count + 1
        user_id = row['user_id']
        service_id = row['service_id']
        try:
            turn_json = json.loads(row['turn_json'])
            # 存入字典
            if (user_id, service_id) not in cases:
                cases[(user_id, service_id)] = [turn_json]
            else:
                cases[(user_id, service_id)].append(turn_json)
        except ValueError:
            pass
        continue

    print("共%d条数据……" % count)
    # keys = list(cases.keys())
    # for k in keys:
    #     # if len(cases[k]) < 5:
    #     if len(cases[k]) >= 5:
    #         del cases[k]
    return cases

def get_turn_structure_data(cases, botactname_dic):
    dialogs_info = {}
    test_list = []
    for k in cases:
        session_turns = []
        for turn in cases[k]:
            cur_turn_obj = Turn(turn, k[0], k[1])
            if cur_turn_obj.botactid in botactname_dic:
                cur_turn_obj.botact_name = botactname_dic[cur_turn_obj.botactid]

            session_turns.append(cur_turn_obj)
            test_list.append(cur_turn_obj)
        dialogs_info[k[1]] = session_turns
    return dialogs_info

# 根据意图筛选图
def get_user_bot_act_route_time_pil_subintent(dialogs_info, botactname_dic, subintent, path):
    """
    根据意图名输出数据集中包含该意图的case
    :param dialogs_info:
    :param botactname_dic:
    :param subintent:
    :param path 日志输出路径
    :return:
    """

    for key in dialogs_info:
        test2 = []
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        for turn in session:
            # if str(key) == '8a6cba4e783e48dc946bf0c16ed2a622':
            #     print str(key)
            subintent_list.append(turn.subintent)
            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                b_time = b_time.seconds
            b_time_start = turn.generator_time
            if turn.botactid in botactname_dic:
                cur_bot = botactname_dic[turn.botactid]
            else:
                cur_bot = turn.botactid

            if turn.type == "answer":

                if len(turn.answerid) > 1 and "answer::answerID$" in turn.answerid:
                    cur_bot = turn.answerid.replace('answer::answerID$', '')
                    if turn.answer_step is not None and len(turn.answer_step) > 0:
                        cur_bot = cur_bot + '__'+turn.answer_step
                else:
                    cur_bot = "answer"
            if turn.answer_step is not None and len(turn.answer_step) > 0:
                cur_bot = turn.answer_step
            #     phone wont turn on特殊处理
            if turn.answer_step is None or len(turn.answer_step) == 0 and "answer::answerID$" not in turn.answerid and "power" in str(turn.multi_botact):
                cur_bot = str(turn.multi_botact)

            if "subintentCode" in turn.user_query:
                turn.user_query = json.loads(turn.user_query)['subintentCode']
            # html标签处理
            dr = re.compile(r'<[^>]+>', re.S)
            turn.user_query = dr.sub('', turn.user_query)
            # http网址处理
            dr = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
            turn.user_query = dr.sub("", turn.user_query)
            dr = re.compile(r'https://[a-zA-Z0-9.?/&=:]*', re.S)
            turn.user_query = dr.sub("", turn.user_query)
            #     过长query处理
            query_reorganize = ""
            query_seperate = turn.user_query
            sep_index = 0
            j = 0
            while len(query_seperate) > 100 and j < 4:
                pos_list = [a.start() for a in re.finditer(' ', query_seperate)]
                j += 1
                for i in pos_list:
                    if i > 80:
                        sep_index = i
                        break
                query_begin = query_seperate[:sep_index]
                query_seperate = query_seperate[sep_index:]
                query_reorganize += query_begin + "\n"
            turn.user_query = query_reorganize + query_seperate
            cur_ = "[" + str(turn.user_query) + "]" + "\n" + str(cur_bot)
            if len(turn.faq_question) > 1:
                cur_ = "[" + str(turn.user_query) + "]" + "\n" + str(turn.faq_question)
            time_list.append(b_time)
            print(cur_)
            test2.append(cur_)
        chnl = session[0].chnl
        if subintent in subintent_list:
            # log_content = read_chatlog(str(key), chatlog_path)
            # print(log_content)

            f = Digraph('test case', filename=path + str(key) + chnl + '.gv')
            for i in range(len(test2)):
                if subintent_list[i] == subintent:

                    # 选定的意图输出box形状
                    f.node(name=str(test2[i]), shape='box')
                if i == 0:
                    f.node(name=str(test2[i]), color='red', style='filled')
                elif i == len(test2) - 1:
                    f.node(name=str(test2[i]), color='lightgrey', style='filled')
                elif str(test2[i]) != str(test2[0]):
                    f.node(name=str(test2[i]), color='black')
                if i < len(test2) - 1:
                    f.edge(str(test2[i]), str(test2[i + 1]), label=str(i) + "(" + str(time_list[i + 1]) + "s)")
            f.render(view=False, format='png')

# 根据serviceid筛选图
def get_user_bot_act_route_time_pil_serviceid(dialogs_info, botactname_dic, serviceid, path):
    """
    根据serviceid 筛选对应的case
    :param dialogs_info:
    :param botactname_dic:
    :param serviceid:
    :return:
    """

    for key in dialogs_info:
        test2 = []
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        for turn in session:

            if turn.serviceid != serviceid:
                continue
            else:
                if (b_time_start == 0):
                    b_time = 0
                else:
                    time_end = (turn.generator_time) / 1000.0
                    time_start = (b_time_start) / 1000.0
                    b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                    b_time = b_time.seconds
                b_time_start = turn.generator_time
                if turn.botactid in botactname_dic:
                    cur_bot = botactname_dic[turn.botactid]
                else:
                    cur_bot = turn.botactid

                if turn.type == "answer":
                    if len(turn.answerid) > 1 and "answer::answerID$" in turn.answerid:
                        cur_bot = turn.answerid.replace('answer::answerID$', '')
                    else:
                        cur_bot = "answer"
                if turn.answer_step is not None and len(turn.answer_step) > 0:
                    cur_bot = turn.answer_step
                #     phone wont turn on特殊处理
                if turn.answer_step is None or len(turn.answer_step) == 0 and "answer::answerID$" not in turn.answerid and "power" in str(turn.multi_botact):
                    cur_bot = str(turn.multi_botact)
                if "subintentCode" in turn.user_query:
                    turn.user_query = json.loads(turn.user_query)['subintentCode']
                # html标签处理
                dr = re.compile(r'<[^>]+>', re.S)
                turn.user_query = dr.sub('', turn.user_query)
                # http网址处理
                dr = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
                turn.user_query = dr.sub("", turn.user_query)
                #     过长query处理
                query_reorganize = ""
                query_seperate = turn.user_query
                sep_index = 0
                while len(query_seperate) > 100:
                    pos_list = [a.start() for a in re.finditer(' ', query_seperate)]
                    for i in pos_list:
                        if i > 80:
                            sep_index = i
                            break
                    query_begin = query_seperate[:sep_index]
                    query_seperate = query_seperate[sep_index:]
                    query_reorganize += query_begin + "\n"
                turn.user_query = query_reorganize + query_seperate
                cur_ = "[" + str(turn.user_query) + "]" + "\n" + str(cur_bot)
                if len(turn.faq_question) > 1:
                    cur_ = "[" + str(turn.user_query) + "]" + "\n" + str(turn.faq_question)
                time_list.append(b_time)
                print(cur_)
                test2.append(cur_)
            chnl = session[0].chnl
            # if subintent in subintent_list:
                # log_content = read_chatlog(str(key), chatlog_path)
                # print(log_content)

            f = Digraph('test case', filename=path + str(key) + chnl + '.gv')
            for i in range(len(test2)):
                    # if subintent_list[i] == subintent:
                    #     # 选定的意图输出box形状
                    #     f.node(name=str(test2[i]), shape='box')
                if i == 0:
                    f.node(name=str(test2[i]), color='red', style='filled')
                elif i == len(test2) - 1:
                    f.node(name=str(test2[i]), color='lightgrey', style='filled')
                elif str(test2[i]) != str(test2[0]):
                    f.node(name=str(test2[i]), color='black')
                if i < len(test2) - 1:
                    f.edge(str(test2[i]), str(test2[i + 1]), label=str(i) + "(" + str(time_list[i + 1]) + "s)")
            f.render(view=False, format='png')


# 根据serviceid筛选图,path
def get_user_bot_act_route_time_pil_serviceid_path(dialogs_info, subintent, botactname_dic, serviceid, path):
    """
    选定的subintent输出指定形状，且只输出指定serviceid的case
    :param dialogs_info:
    :param subintent:
    :param botactname_dic:
    :param serviceid: 要输出的日志的serviceid
    :param path: 输出日志的路径
    :return:
    """
    for key in dialogs_info:
        test2 = []
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        for turn in session:
            subintent_list.append(turn.subintent)
            if turn.serviceid != serviceid:
                continue
            else:
                if (b_time_start == 0):
                    b_time = 0
                else:
                    time_end = (turn.generator_time) / 1000.0
                    time_start = (b_time_start) / 1000.0
                    b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                    b_time = b_time.seconds
                b_time_start = turn.generator_time
                if turn.botactid in botactname_dic:
                    cur_bot = botactname_dic[turn.botactid]
                else:
                    cur_bot = turn.botactid

                if turn.type == "answer":
                    if len(turn.answerid) > 1 and "answer::answerID$" in turn.answerid:
                        cur_bot = turn.answerid.replace('answer::answerID$', '')
                    else:
                        cur_bot = "answer"

                if turn.answer_step is not None and len(turn.answer_step) > 0:
                    cur_bot = turn.answer_step
                #     phone wont turn on特殊处理
                if turn.answer_step is None or len(turn.answer_step) == 0 and "answer::answerID$" not in turn.answerid and "power" in str(turn.multi_botact):
                    cur_bot = str(turn.multi_botact)

                if "subintentCode" in turn.user_query:
                    turn.user_query = json.loads(turn.user_query)['subintentCode']
                # html标签处理
                dr = re.compile(r'<[^>]+>', re.S)
                turn.user_query = dr.sub('', turn.user_query)
                # http网址处理
                dr = re.compile(r'http://[a-zA-Z0-9.?/&=:]*', re.S)
                turn.user_query = dr.sub("", turn.user_query)
                turn.user_query = turn.user_query.replace(":", "")
                #     过长query处理
                query_reorganize = ""
                query_seperate = turn.user_query
                sep_index = 0
                while len(query_seperate) > 100:
                    pos_list = [a.start() for a in re.finditer(' ', query_seperate)]
                    for i in pos_list:
                        if i > 80:
                            sep_index = i
                            break
                    query_begin = query_seperate[:sep_index]
                    query_seperate = query_seperate[sep_index:]
                    query_reorganize += query_begin + "\n"
                turn.user_query = query_reorganize + query_seperate
                cur_ = "[" + str(turn.user_query) + "]" + "\n" + str(cur_bot)
                if len(turn.faq_question) > 1:
                    cur_ = "[" + str(turn.user_query) + "]" + "\n" + str(turn.faq_question)
                time_list.append(b_time)
                print(cur_)
                test2.append(cur_)
            chnl = session[0].chnl
            # if subintent in subintent_list:
            #     log_content = read_chatlog(str(key), chatlog_path)
                # print(log_content)

            f = Digraph('test case', filename=path + '_' + str(key) + chnl + '.gv')
            for i in range(len(test2)):
                if subintent_list[i] == subintent:
                        # 选定的意图输出box形状
                    f.node(name=str(test2[i]), shape='box')
                if i == 0:
                    f.node(name=str(test2[i]), color='red', style='filled')
                elif i == len(test2) - 1:
                    f.node(name=str(test2[i]), color='lightgrey', style='filled')
                elif str(test2[i]) != str(test2[0]):
                    f.node(name=str(test2[i]), color='black')
                if i < len(test2) - 1:
                    f.edge(str(test2[i]), str(test2[i + 1]), label=str(i) + "(" + str(time_list[i + 1]) + "s)")
            f.render(view=False, format='png')


def read_chatlog(service_id, chatlog_path):
    """
    chatlog 读取（暂时保留）
    :param service_id:
    :param chatlog_path:
    :return:
    """
    cl_df = pd.read_excel(chatlog_path, sheet_name='rptChatVisit', header=1)
    # cl_df = pd.read_excel('../chatlog0901.xlsx', sheet_name='rptChatVisit', header=1)
    log_data = cl_df.loc[cl_df['serviceid'] == service_id]['chatcontent']
    log_process_str = ""
    if pd.notna(log_data.values):
        log_content = cl_df.loc[cl_df['serviceid'] == service_id]['chatcontent'].values[0]
        # 无关字符去除
        log_list = log_content.split("\n")
        log_process_list = []

        for log_one in log_list:
            # print(log_one)
            # log_one = re.sub('\\(.*?\\)]', '', log_one[13:])
            log_one = log_one[13:]
            log_process_list.append(log_one)
            log_process_str += log_one
            log_process_str += "\n"

        # print(log_one)

    return log_process_str

def time_statistic(cases, botactname_dic, subintent):

    dialogs_info = get_turn_structure_data(cases, botactname_dic)
    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []
    for key in dialogs_info:
        answer_time_dic = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_name = ""
        last_answer_subintent = ""
        for turn in session:
            subintent_list.append(turn.subintent)
            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                b_time = b_time.seconds
            b_time_start = turn.generator_time
            time_list.append(b_time)
            if subintent_show in turn.answer_name and "___" in turn.answer_name:
                answer_title_list = turn.answer_name.split('___')
                if last_answer_subintent == answer_title_list[0] and len(answer_title_list[1]) > 1 and last_answer_name != answer_title_list[1]:
                    answer_time_dic[(last_answer_name, answer_title_list[1])] = b_time

                last_answer_subintent = answer_title_list[0]
                last_answer_name = answer_title_list[1]
            else:
                last_answer_name = turn.answerid
                last_answer_subintent = turn.subintent
        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
    # print answer_list
    answer_df = pd.DataFrame(answer_list)
    f, ax_l = plt.subplots(1, len(answer_df.columns), figsize=(20, 4))
    for e, col_name in enumerate(answer_df.columns):
        df = answer_df[col_name].to_frame()
        df = df.dropna()
        df['index'] = df.index.tolist()
        ax_l[e].scatter(df['index'], df[col_name], alpha=0.5, color='r')
        ax_l[e].set_xlabel(str(col_name))
        ax_l[e].set_ylabel('feedback_time')
    plt.show()

def split_row(data, column):
    '''拆分成行

    :param data: 原始数据
    :param column: 拆分的列名
    :type data: pandas.core.frame.DataFrame
    :type column: str
    '''
    row_len = list(map(len, data[column].values))
    rows = []
    for i in data.columns:
        if i == column:
            row = np.concatenate(data[i].values)
        else:
            row = np.repeat(data[i].values, row_len)
        rows.append(row)
    return pd.DataFrame(np.dstack(tuple(rows))[0], columns=data.columns)

def figure_plot(answer_df, path):
    del answer_df['chnl']
    del answer_df['serviceid']
    f, ax_l = plt.subplots(1, len(answer_df.columns), figsize=(20, 4))
    for e, col_name in enumerate(answer_df.columns):
        # if col_name != "chnl" and col_name != "serviceid":
        df = answer_df[col_name].to_frame()
        df = df.dropna()
        if df.empty:
            continue
        df['index'] = df.index.tolist()
        # 多值拆分成多行
        df = split_row(df, column=col_name)
        df[col_name].astype('int')
        ax_l[e].scatter(df['index'], df[col_name], alpha=0.5, color='r')
        ax_l[e].set_xlabel(str(col_name))
        ax_l[e].set_ylabel('feedback_time')
    plt.show()
    plt.savefig(path)

def df_data_process(answer_df, col_name):
    """
    时间密度分布图数据处理
    :param answer_df:
    :param col_name:
    :return: 去掉空值，多个值会拆分成多行
    """
    df = pd.DataFrame()
    if col_name in answer_df.columns.tolist():
        df = answer_df[col_name].to_frame()
        df = df.dropna()
    if df.empty:
        return df
    df['index'] = df.index.tolist()
    # 多值拆分成多行
    df = split_row(df, column=col_name)
    # useract去掉
    # df[col_name].astype('float')

    return df

def df_data_process2(answer_df, col_name):
    """
    用户行为数据处理
    :param answer_df: dataframe 数据
    :param col_name: 列名
    :return:
    """
    df = pd.DataFrame()
    if col_name in answer_df.columns.tolist():
        df = answer_df[col_name].to_frame()
        df = df.dropna()
    if df.empty:
        return df
    df['index'] = df.index.tolist()
    # 多值拆分成多行
    df = split_row(df, column=col_name)
    # pd.concat([df, df[col_name].str.split(':', expand=True)], axis=1, names='time')

    df['time'] = df[col_name].map(lambda x: x.split(':')[1])
    df[col_name] = df[col_name].map(lambda x: x.split(':')[0])
    # useract去掉
    # df[col_name].astype('float')
    return df

def figure_plot_single(answer_df, answer_df2, path):
    """
    暂时不用
    :param answer_df:
    :param answer_df2:
    :param path:
    :return:
    """
    del answer_df['chnl']
    del answer_df['serviceid']
    del answer_df2['chnl']
    del answer_df2['serviceid']
    for e, col_name in enumerate(answer_df.columns):
        df = df_data_process(answer_df, col_name)
        df2 = df_data_process(answer_df2, col_name)
        if df.empty and df2.empty:
            continue
        elif df.empty:
            plt.scatter(df2['index'], df2[col_name], alpha=0.5, color='r', marker='x', label='facebook')
        elif df2.empty:
            plt.scatter(df['index'], df[col_name], alpha=0.5, color='b', marker='o', label='pc')
        else:
            plt.scatter(df['index'], df[col_name], alpha=0.5, color='b', marker='o', label='pc')
            plt.scatter(df2['index'], df2[col_name], alpha=0.5, color='r', marker='x', label='facebook')
        plt.xlabel(str(col_name))
        plt.ylabel('feedback_time')
        plt.legend()
        plt.savefig(path + col_name + '.png')
        plt.show()

# 曲线拟合
def figure_plot_line(answer_df, path):
    """
    所有步骤的数量曲线曲线图（暂时不用）
    :param answer_df:
    :param path:
    :return:
    """
    del answer_df['chnl']
    del answer_df['serviceid']
    result = pd.DataFrame()
    for e, col_name in enumerate(answer_df.columns):
        if len(col_name) < 1:
            continue
        df = df_data_process(answer_df, col_name)
        result = result.append(df, ignore_index=True)

    result.drop(['index'], axis=1, inplace=True)
    result.plot()

    plt.ylabel('feedback_time')
    plt.legend()
    plt.savefig(path + '.pdf', dpi=200, bbox_inches='tight')
    plt.show()

# 线性回归图(每一列一个图）散点图回归曲线
def figure_plot_reg(answer_df, subintent, botactname_dic, path ,chnl_step_dict, channel):
    """
    时间散点图，看总体的时间分布
    :param answer_df:
    :param subintent:
    :param botactname_dic:
    :param path:
    :param chnl_step_dict: 各个渠道对应的答案步骤字典
    :param channel:
    :return:
    """
    answer_df_bak = answer_df.copy()
    del answer_df_bak['chnl']
    # del answer_df['serviceid']
    # result = pd.DataFrame()
    step_list = chnl_step_dict[channel]
    for step in step_list:
        # step_code = step['stepStandardCode']
        step_name = step['stepStandardName']
        step_order = step['stepOrder']

    # for e, col_name in enumerate(answer_df.columns):
    #     if len(col_name) < 1 or col_name == 'serviceid':
    #         continue
        col_name = step_name
        df = df_data_process(answer_df_bak, col_name)
        if df.empty:
            continue

        sns_figure = sns.regplot(x='index', y=col_name, data=df)
        sns_figure.text(5, 5, str(df[col_name].describe()), color="b")

        sns_figure.set_ylabel(str(step_order) + ":" + step_name)
        f = sns_figure.get_figure()
        f.savefig(path + col_name + '.png', dpi=400, bbox_inches='tight')
        # plt.close()
        plt.show()
        # std_value = df[col_name].std()
        # u = df[col_name].mean()

        # x = df.loc[df[col_name] > (u + (3*std_value))]['index']
        # x = df.loc[df[col_name] > 0]['index']
        # if x.empty:
        #     continue
        # else:
        #     for i in x.tolist():
        #         serviceid = answer_df_bak.loc[int(i)]['serviceid']
        #         # graphviz install
        #         get_user_bot_act_route_time_pil_serviceid_path(dialogs_info, subintent, botactname_dic, serviceid, r'D:\dialoggraph0916serviceid\us_dialog'+col_name)

        # print str(col_name) + '标准差:' + str(df[col_name].std())
        # print str(col_name) + '中位数:' + str(df[col_name].median())
        # print col_name + '均值:' + str(df[col_name].mean())
        # print col_name + '多指标统计:' + str(df[col_name].describe())

        # result = result.append(df, ignore_index=True)

# 直方图分布
def figure_plot_bar(answer_df, subintent, botactname_dic, chnl_step_dict, channel, path, loggraph_or_not, path2):
    """

    :param answer_df:
    :param subintent:
    :param botactname_dic:
    :param chnl_step_dict:各个渠道答案的字典（不同渠道答案会有细微差异）
    :param channel:
    :param path: distplot图生成路径
    :param loggraph_or_not: 是否需要根据每一步反馈时间筛选日志并输出
    :param path2: loggraph_or_not==True时，graphviz输出日志图路径
    :return:
    """
    answer_df_bak = answer_df.copy()

    # del answer_df['chnl']
    # del answer_df['serviceid']
    # result = pd.DataFrame()
    step_list = chnl_step_dict[channel]
    for step in step_list:
        # step_code = step['stepStandardCode']
        step_name = step['stepStandardName']
        step_order = step['stepOrder']
        col_name = step_name
        df = df_data_process(answer_df_bak, col_name)
        if df.empty:
            continue

        # sns_figure = sns.distplot(df[col_name], bins=20,  color="blue")
        sns_figure = sns.distplot(df[col_name], bins=range(0, 1200, 50), color="blue")
        sns_figure.set_ylabel(str(step_order) + ":" + step_name)
        f = sns_figure.get_figure()
        f.savefig(path + col_name + 'distplot_a.png', dpi=400, bbox_inches='tight')

        plt.show()
        std_value = df[col_name].std()
        u = df[col_name].mean()
        # 根据步骤时间筛选日志，并生成日志流程图
        if loggraph_or_not == True:
            # x = df.loc[df[col_name] > (u + (3*std_value))]['index']
            # x = df.loc[df[col_name] < 3]['index']
            x = df.loc[df[col_name] > 400]['index']
            # x = df.loc[(df[col_name] <= 200)&(df[col_name] >= 3)]['index']
            # 打印对应日志
            if x.empty:
                continue
            else:
                for i in x.tolist():
                    serviceid = answer_df.loc[int(i)]['serviceid']

                    get_user_bot_act_route_time_pil_serviceid_path(dialogs_info, subintent, botactname_dic, serviceid, path2 + col_name)

        # print str(col_name) + '标准差:' + str(df[col_name].std())
        # print str(col_name) + '中位数:' + str(df[col_name].median())
        # print col_name + '均值:' + str(df[col_name].mean())
        # print col_name + '多指标统计:' + str(df[col_name].describe())

        # result = result.append(df, ignore_index=True)


# 用户行为直方图分布
def figure_plot_useract_bar(answer_df, subintent, botactname_dic, chnl_step_dict, channel, path):
    """
    用户行为分布和每一步的下一步用户行为时间密度分布
    :param answer_df:
    :param subintent:
    :param botactname_dic:
    :param chnl_step_dict: 各个渠道答案的字典（不同渠道答案会有细微差异）
    :param channel: 筛选的渠道
    :param path: 图形生成存放的路径
    :return:
    """
    del answer_df['chnl']
    # del answer_df['serviceid']
    # result = pd.DataFrame()
    step_list = chnl_step_dict[channel]
    for step in step_list:
        # step_code = step['stepStandardCode']
        step_name = step['stepStandardName']
        step_order = step['stepOrder']
        col_name = step_name
        df = df_data_process2(answer_df, col_name)
        if df.empty:
            continue

        # sns_figure = sns.distplot(df[col_name], bins=20,  color="blue")

        # df['count'] = 1
        dfg = df.groupby(col_name).count().reset_index()

        dfg.rename(columns={'index': 'count'}, inplace=True)
        dfg['count'].astype('int')
        dfg.plot.bar(x=col_name, y='count')
        # sns_figure.set_xlabel(df[col_name].values)

        plt.xlabel(str(step_order) + ":" + step_name)
        plt.tight_layout()
        plt.savefig(path + col_name + 'bar_useract.png', dpi=300, bbox_inches='tight')
        plt.show()
        # df.drop(['index'], axis=1, inplace=True)
        f = plt.figure()
        i = 1
        for step_name in dfg[col_name]:
            step_count = math.ceil(len(dfg)/2)
            f.add_subplot(step_count, 2, i)
            df_s = df.loc[df[col_name] == step_name]
            df_s.loc[:, 'time'] = df_s.loc[:, 'time'].astype('float').copy()
            print(df_s)
            print(df_s)
            sns_figure = sns.distplot(df_s['time'], bins=10)
            sns_figure.set_xlabel(str(step_order) + ":" + col_name)
            sns_figure.set_ylabel(str(step_name))
            i = i + 1
            # f = sns_figure.get_figure()
            # f.savefig(path + step_name + 'distplot_step.png', dpi=400, bbox_inches='tight')
            # plt.show()
        f = sns_figure.get_figure()
        f.savefig(path + col_name + 'distplot_step.png', dpi=400, bbox_inches='tight')
        plt.show()

# 用户行为直方图分布(多个)
def figure_plot_useract_bar_multi(answer_df, subintent, botactname_dic, chnl_step_dict, channel, path):
    del answer_df['chnl']
    # del answer_df['serviceid']
    # result = pd.DataFrame()
    step_list = chnl_step_dict[channel]
    for step in step_list:
        # step_code = step['stepStandardCode']
        step_name = step['stepStandardName']
        step_order = step['stepOrder']
        col_name = step_name
        df = df_data_process2(answer_df, col_name)
        if df.empty:
            continue

        # sns_figure = sns.distplot(df[col_name], bins=20,  color="blue")

        # df['count'] = 1
        dfg = df.groupby(col_name).count().reset_index()

        dfg.rename(columns={'index': 'count'}, inplace=True)
        dfg['count'].astype('int')
        dfg.plot.bar(x=col_name, y='count')
        # sns_figure.set_xlabel(df[col_name].values)

        plt.xlabel(str(step_order) + ":" + step_name)
        plt.tight_layout()
        plt.savefig(path + col_name + 'bar_useract.png', dpi=400, bbox_inches='tight')

        plt.show()
        # df.drop(['index'], axis=1, inplace=True)
        # f = plt.figure()
        # i = 1
        for step_name in dfg[col_name]:
            # step_count = math.ceil(float(len(dfg))/2)
            # f.add_subplot(step_count, 2, i)
            df_s = df.loc[df[col_name] == step_name]
            df_s['time'] = df_s['time'].astype('float')
            sns_figure = sns.distplot(df_s['time'], bins=10)
            sns_figure.set_xlabel(str(step_order) + ":" + col_name)
            sns_figure.set_ylabel(str(step_name))
            f = sns_figure.get_figure()
            f.savefig(path + step_name + 'distplot_step.png', dpi=400, bbox_inches='tight')
            plt.show()



# 直方图分布(log放大）
def figure_plot_bar_log(answer_df, subintent, botactname_dic,chnl_step_dict, channel, path):
    del answer_df['chnl']
    # del answer_df['serviceid']
    # result = pd.DataFrame()
    step_list = chnl_step_dict[channel]
    for step in step_list:
        # step_code = step['stepStandardCode']
        step_name = step['stepStandardName']
        step_order = step['stepOrder']
        col_name = step_name
        df = df_data_process(answer_df, col_name)
        if df.empty:
            continue

        sns_figure = sns.distplot(df[col_name], bins=20,  color="blue")
        sns_figure.set_ylabel(str(step_order) + ":" + step_name)
        # sns_figure = sns.distplot(df[col_name], bins=range(0, 1200, 50), color="blue")
        f = sns_figure.get_figure()
        f.savefig(path + col_name + 'distplot_log.png', dpi=400, bbox_inches='tight')

        plt.show()
        # case shaixuan
        # std_value = df[col_name].std()
        # u = df[col_name].mean()
        #
        # x = df.loc[df[col_name] > (u + (3*std_value))]['index']
        # if x.empty:
        #     continue
        # else:
        #     for i in x.tolist():
        #         serviceid = answer_df.loc[int(i)]['serviceid']
        #         print serviceid
        #         get_user_bot_act_route_time_pil_serviceid_path(dialogs_info, subintent, botactname_dic, serviceid, 'D:\dialoggraph0916serviceid\dialog'+col_name)

        # print str(col_name) + '标准差:' + str(df[col_name].std())
        # print str(col_name) + '中位数:' + str(df[col_name].median())
        # print col_name + '均值:' + str(df[col_name].mean())
        # print col_name + '多指标统计:' + str(df[col_name].describe())

        # result = result.append(df, ignore_index=True)



# pairplot 对角线图
def figure_plot_pairplot(answer_df, path):
    """
    暂时不用
    :param answer_df:
    :param path:
    :return:
    """
    del answer_df['chnl']
    del answer_df['serviceid']
    result = pd.DataFrame()
    columns_list = []
    for e, col_name in enumerate(answer_df.columns):
        if len(col_name) < 1:
            continue
        columns_list.append(col_name)
        df = df_data_process(answer_df, col_name)
        result = result.append(df, ignore_index=True)

    # result.drop(['index'], axis=1, inplace=True)
    # sns_figure = sns.pairplot(result, x_vars=["index"],
    #          y_vars=["7fFnbg2VBS5428fo", "clear_cache_and_data", "contact_your_carrier_4", "reboot_the_phone", "reset_the_phone", "s3LnvlCXx3vel879", "WWT43Vlbf5UsbfQ3"])
    sns_figure = sns.pairplot(result, x_vars=["index"], y_vars=columns_list, diag_kind="hist", kind="reg")

    sns_figure.savefig(path + 'pairplot2' + '.png', dpi=200, bbox_inches='tight')
    plt.show()


def time_statistic2(cases, botactname_dic, subintent, related_botact):

    dialogs_info = get_turn_structure_data(cases, botactname_dic)
    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []
    for key in dialogs_info:
        # if str(key) == '4075f60181e844b2b220d0459b52075f':
        #     print str(key)
        answer_time_dic = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_name = ""
        last_answer_subintent = ""
        last_multi_botact = ""
        for turn in session:
            subintent_list.append(turn.subintent)
            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                b_time = b_time.seconds
            b_time_start = turn.generator_time
            time_list.append(b_time)
            if subintent_show in turn.answer_name and "___" in turn.answer_name:
                answer_title_list = turn.answer_name.split('___')
                if last_answer_subintent == answer_title_list[0]:
                    # answer_time_dic[last_answer_name].append(b_time)
                    answer_time_dic.setdefault(last_answer_name, []).append(b_time)
                    answer_time_dic['serviceid'] = turn.serviceid
                    answer_time_dic['chnl'] = turn.chnl

                last_answer_subintent = answer_title_list[0]
                last_answer_name = answer_title_list[1]
                last_multi_botact = turn.multi_botact
            else:
                if len(last_multi_botact) > 0:
                    for botact_m in last_multi_botact:
                        # if botact_m in related_botact:
                        if related_botact in botact_m:
                            # answer_time_dic[botact_m].append(b_time)
                            answer_time_dic.setdefault(botact_m, []).append(b_time)
                            answer_time_dic['serviceid'] = turn.serviceid
                            answer_time_dic['chnl'] = turn.chnl
                            break
                last_answer_name = turn.answerid
                last_answer_subintent = turn.subintent
                last_multi_botact = turn.multi_botact

        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
    # print answer_list
    answer_df = pd.DataFrame(answer_list)
    answer_df.to_excel('../graphtime/answer_step5.xlsx')
    answer_df_pc = answer_df.loc[answer_df['chnl'] == 'pc']
    answer_df_mobile = answer_df.loc[answer_df['chnl'] == 'mobile']
    answer_df_facebook = answer_df.loc[answer_df['chnl'] == 'facebook']

    # figure_plot(answer_df_pc, r'../graphtime/pc.png')
    # figure_plot(answer_df_facebook, r'../graphtime/facebook.png')
    # figure_plot(answer_df_mobile, r'../graphtime/mobile.png')

    # figure_plot_single(answer_df_pc, answer_df_mobile, r'../graphtime/all_')
    # figure_plot_single(answer_df_pc, answer_df_facebook, r'../graphtime/all_facebook')
    # 直接plot显示图
    figure_plot_line(answer_df_pc, r'../graphtime/line')

def time_statistic2_new(dialogs_info, botactname_dic, subintent, related_botact, channel):
    """
    复杂流程分析
    :param dialogs_info:
    :param botactname_dic:
    :param subintent:
    :param related_botact:跟复杂流程步骤相关的话术关键词
    :return:
    """

    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []
    for key in dialogs_info:
        # if str(key) == '4075f60181e844b2b220d0459b52075f':
        #     print str(key)
        answer_time_dic = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_name = ""
        last_answer_subintent = ""
        last_multi_botact = ""

        for turn in session:
            subintent_list.append(turn.subintent)
            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                b_time = b_time.seconds
                # 取对数放大
                # if b_time.seconds > 0:
                #     b_time = math.log(b_time.seconds)
                # elif b_time.seconds == 0:
                #     tmp = round(float(b_time.microseconds)/1000000, 2)
                #     print tmp
                #     if tmp > 0:
                #         b_time = round(math.log(tmp), 2)
                #     else:
                #         b_time = 0

            b_time_start = turn.generator_time
            time_list.append(b_time)
            if subintent_show in turn.answer_name and "___" in turn.answer_name:
                answer_title_list = turn.answer_name.split('___')
                turn.subintent = answer_title_list[0]
                turn.answer_name = answer_title_list[1]
                last_multi_botact = turn.multi_botact
            if subintent_show in last_answer_subintent:
                    # answer_time_dic[last_answer_name].append(b_time)
                answer_time_dic.setdefault(last_answer_name, []).append(b_time)
                answer_time_dic['serviceid'] = turn.serviceid
                answer_time_dic['chnl'] = turn.chnl

            else:
                if len(last_multi_botact) > 0:
                    for botact_m in last_multi_botact:
                        # if botact_m in related_botact:
                        if related_botact in botact_m:
                         # answer_time_dic[botact_m].append(b_time)
                            answer_time_dic.setdefault(botact_m, []).append(b_time)
                            answer_time_dic['serviceid'] = turn.serviceid
                            answer_time_dic['chnl'] = turn.chnl
                            break
            last_answer_name = turn.answer_name
            last_answer_subintent = turn.subintent
            last_multi_botact = turn.multi_botact

        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
    # print answer_list
    answer_df = pd.DataFrame(answer_list)
    answer_df.to_excel('../graphtime/answer_step_' + subintent + '.xlsx')

    answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]

    # 回归图（多图）
    # figure_plot_reg(answer_df_pc, subintent, botactname_dic, r'../graphtime/us_reg_' + subintent)
    # 直方图（密度拟合曲线）
    figure_plot_bar(answer_df_channel, subintent, botactname_dic, r'../graphtime/us_reg_' + channel + "_" + subintent, False, r'D:\dialogGraph0918\dialog_l_')
    # 多列回归图
    # figure_plot_pairplot(answer_df_pc, r'../graphtime/reg_' + subintent)

def time_statistic2_new_log(dialogs_info, botactname_dic, subintent, related_botact, channel):
    """
    复杂流程分析log放大（暂时不用）
    :param dialogs_info:
    :param botactname_dic:
    :param subintent:
    :param related_botact:
    :param channel:
    :return:
    """
    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []
    for key in dialogs_info:
        # if str(key) == '4075f60181e844b2b220d0459b52075f':
        #     print str(key)
        answer_time_dic = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_name = ""
        last_answer_subintent = ""
        last_multi_botact = ""

        for turn in session:
            subintent_list.append(turn.subintent)
            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                # b_time = b_time.seconds
                # 取对数放大
                if b_time.seconds > 0:
                    b_time = math.log(b_time.seconds)
                elif b_time.seconds == 0:
                    tmp = round(float(b_time.microseconds)/1000000, 2)

                    if tmp > 0:
                        b_time = round(math.log(tmp), 2)
                    else:
                        b_time = 0

            b_time_start = turn.generator_time
            time_list.append(b_time)
            if subintent_show in turn.answer_name and "___" in turn.answer_name:
                answer_title_list = turn.answer_name.split('___')
                turn.subintent = answer_title_list[0]
                turn.answer_name = answer_title_list[1]
                last_multi_botact = turn.multi_botact
            if subintent_show in last_answer_subintent:
                    # answer_time_dic[last_answer_name].append(b_time)
                answer_time_dic.setdefault(last_answer_name, []).append(b_time)
                answer_time_dic['serviceid'] = turn.serviceid
                answer_time_dic['chnl'] = turn.chnl

            else:
                if len(last_multi_botact) > 0:
                    for botact_m in last_multi_botact:
                        # if botact_m in related_botact:
                        if related_botact in botact_m:
                         # answer_time_dic[botact_m].append(b_time)
                            answer_time_dic.setdefault(botact_m, []).append(b_time)
                            answer_time_dic['serviceid'] = turn.serviceid
                            answer_time_dic['chnl'] = turn.chnl
                            break
            last_answer_name = turn.answer_name
            last_answer_subintent = turn.subintent
            last_multi_botact = turn.multi_botact

        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
    # print answer_list
    answer_df = pd.DataFrame(answer_list)
    answer_df.to_excel('../graphtime/answer_step_log' + subintent + '.xlsx')

    answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]

    # figure_plot_line(answer_df_pc, r'../graphtime/line0914')
    # 回归图（多图）
    # figure_plot_reg(answer_df_pc, subintent, botactname_dic, r'../graphtime/us_reg_' + subintent)
    # 直方图（密度拟合曲线）
    figure_plot_bar_log(answer_df_channel, subintent, botactname_dic, r'../graphtime/us_reg_log_' + subintent)
    # 多列回归图
    # figure_plot_pairplot(answer_df_pc, r'../graphtime/reg_' + subintent)

def time_statistic3(dialogs_info, botactname_dic, subintent, channel):
    """
    不区分第一次和以后进入的散点图和时间密度分布图
    :param dialogs_info:
    :param botactname_dic:
    :param subintent:
    :return:
    """

    # dialogs_info = get_turn_structure_data(cases, botactname_dic)
    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []

    chnl_step_dict = {}

    for key in dialogs_info:
        # if str(key) == '4075f60181e844b2b220d0459b52075f':
        #     print str(key)
        answer_time_dic = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_step = ""
        last_answer_name = ""
        for turn in session:
            kg_step_dict_list = []
            subintent_list.append(turn.subintent)
            if turn.subintent == subintent:
                if turn.chnl not in chnl_step_dict.keys():
                    if len(kg_step_dict_list) == 0 and len(turn.answer_step) > 0 and len(turn.kgResult) > 0:
                        kg_result = json.loads(turn.kgResult)['result']
                        if "answerPart" in kg_result.keys():
                            result_k = kg_result['answerPart']
                            if "steps" in result_k.keys():
                                steps_k = result_k['steps']
                                if len(steps_k) > 0:
                                    for step in steps_k:
                                        kg_step_dict = {}
                                        kg_step_dict['stepStandardCode'] = step['stepStandardCode']
                                        kg_step_dict['stepStandardName'] = step['stepStandardName']
                                        kg_step_dict['stepOrder'] = step['stepOrder']
                                        kg_step_dict_list.append(kg_step_dict)
                                    chnl_step_dict[turn.chnl] = kg_step_dict_list


            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                b_time = b_time.seconds
                # 取对数放大
                # if b_time.seconds > 0:
                #     b_time = math.log(b_time.seconds)
                # elif b_time.seconds == 0:
                #     tmp = round(float(b_time.microseconds)/1000000, 2)
                #     print tmp
                #     if tmp > 0:
                #         b_time = round(math.log(tmp), 2)
                #     else:
                #         b_time = 0
            b_time_start = turn.generator_time
            time_list.append(b_time)
            if len(last_answer_name) > 0 and subintent_show in last_answer_name:
                if len(last_answer_step) == 0:
                    # ts单步答案的处理（answer_name代替answer_step)
                    answer_time_dic.setdefault(last_answer_name, []).append(b_time)
                else:
                    answer_time_dic.setdefault(last_answer_step, []).append(b_time)
                answer_time_dic['serviceid'] = turn.serviceid
                answer_time_dic['chnl'] = turn.chnl
                # answer_time_dic['useract'] = turn.user_query


            last_answer_step = turn.answer_step_name
            last_answer_name = turn.answer_name


        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
    # print answer_list
    answer_df = pd.DataFrame(answer_list)
    # answer_df.to_excel('../graphtime/answer_step_' + subintent + '.xlsx')

    answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]

    # 散点图
    figure_plot_reg(answer_df_channel, subintent, botactname_dic, r'../graphtime/std_all_' + subintent, chnl_step_dict, channel)
    # figure_plot_reg(answer_df_mobile, subintent, botactname_dic, r'../graphtime/reg_std_mobile_' + subintent, chnl_step_dict, 'mobile')
    # figure_plot_reg(answer_df_facebook, subintent, botactname_dic, r'../graphtime/reg_std_mobile_' + subintent, chnl_step_dict, 'facebook')

    # 柱状图
    figure_plot_bar(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/distplot_all_' + channel + "_" + subintent, False, "")
    # figure_plot_bar(answer_df_mobile, subintent, botactname_dic, chnl_step_dict, 'mobile', r'../graphtime/reg_mobile_' + subintent)

    # figure_plot_pairplot(answer_df_pc, r'../graphtime/reg_' + subintent)

def time_statistic3_choice(dialogs_info, botactname_dic, subintent, channel, choice_l):
    """
    区分第一次和以后进入的散点图和时间密度分布图（柱状拟合图）
    :param dialogs_info: 数据源
    :param botactname_dic: botactid到botactname映射
    :param subintent: 要查询的意图
    :param channel: 渠道
    :param choice_l: 选择图形生成的类型 1：只统计第一次进入 2：只统计第二次及以后进入 12：1和2都统计
    :return: 绘制的图表(散点图和柱状拟合图）
    """

    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []
    answer_second_list = []
    chnl_step_dict = {}

    for key in dialogs_info:
        answer_time_dic_second = {}
        answer_time_dic = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_step = ""
        last_answer_name = ""
        for turn in session:
            kg_step_dict_list = []
            subintent_list.append(turn.subintent)
            if turn.subintent == subintent:
                if turn.chnl not in chnl_step_dict.keys():
                    if len(kg_step_dict_list) == 0 and len(turn.answer_step) > 0 and len(turn.kgResult) > 0:
                        kg_result = json.loads(turn.kgResult)['result']
                        if "answerPart" in kg_result.keys():
                            result_k = kg_result['answerPart']
                            if "steps" in result_k.keys():
                                steps_k = result_k['steps']
                                if len(steps_k) > 0:
                                    for step in steps_k:
                                        kg_step_dict = {}
                                        kg_step_dict['stepStandardCode'] = step['stepStandardCode']
                                        kg_step_dict['stepStandardName'] = step['stepStandardName']
                                        kg_step_dict['stepOrder'] = step['stepOrder']
                                        kg_step_dict_list.append(kg_step_dict)
                                    chnl_step_dict[turn.chnl] = kg_step_dict_list


            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                b_time = b_time.seconds
                # 取对数放大
                # if b_time.seconds > 0:
                #     b_time = math.log(b_time.seconds)
                # elif b_time.seconds == 0:
                #     tmp = round(float(b_time.microseconds)/1000000, 2)
                #     print tmp
                #     if tmp > 0:
                #         b_time = round(math.log(tmp), 2)
                #     else:
                #         b_time = 0
            b_time_start = turn.generator_time
            time_list.append(b_time)
            if len(last_answer_name) > 0 and subintent_show in last_answer_name:
                if len(last_answer_step) == 0:
                    # ts单步答案的处理（answer_name代替answer_step)
                    if last_answer_name in answer_time_dic:
                        answer_time_dic_second.setdefault(last_answer_name, []).append(b_time)
                    else:
                        answer_time_dic.setdefault(last_answer_name, []).append(b_time)
                else:
                    if last_answer_step in answer_time_dic:
                        answer_time_dic_second.setdefault(last_answer_step, []).append(b_time)
                    else:
                        answer_time_dic.setdefault(last_answer_step, []).append(b_time)
                answer_time_dic['serviceid'] = turn.serviceid
                answer_time_dic['chnl'] = turn.chnl
                # answer_time_dic['useract'] = turn.user_query

                answer_time_dic_second['serviceid'] = turn.serviceid
                answer_time_dic_second['chnl'] = turn.chnl
                # answer_time_dic_second.setdefault('useract', []).append(turn.user_query)
                # answer_time_dic_second['useract'] = turn.user_query

            last_answer_step = turn.answer_step_name
            last_answer_name = turn.answer_name

        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
        if len(answer_time_dic_second) > 0:
            answer_second_list.append(answer_time_dic_second)
            # print answer_list
    answer_df = pd.DataFrame(answer_list)
    answer_second_df = pd.DataFrame(answer_second_list)
    # answer_df.to_excel('../graphtime/answer_step_' + subintent + '.xlsx')
    if choice_l == 1:
        answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]
        # 散点图
        figure_plot_reg(answer_df_channel, subintent, botactname_dic, r'../graphtime/reg_std_1' + "_" + channel + "_" + subintent, chnl_step_dict, channel)
        # 柱状图
        figure_plot_bar(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_pc_1' + "_" + channel + "_" + subintent, False, "")
    if choice_l == 2:
        answer_df_channel_second = answer_second_df.loc[answer_df['chnl'] == channel]
        # 散点图
        figure_plot_reg(answer_df_channel_second, subintent, botactname_dic, r'../graphtime/reg_std_2' + "_" + channel + "_" + subintent, chnl_step_dict, channel)
        # 柱状图
        figure_plot_bar(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_pc_2' + "_" + channel + "_" + subintent, False, "")
    if choice_l == 12:
        answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]
        # 散点图
        figure_plot_reg(answer_df_channel, subintent, botactname_dic,
                        r'../graphtime/reg_std_1' + "_" + channel + "_" + subintent, chnl_step_dict, channel)
        # 柱状图
        figure_plot_bar(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel,
                        r'../graphtime/reg_pc_1' + "_" + channel + "_" + subintent, False, "")

        answer_df_channel_second = answer_second_df.loc[answer_df['chnl'] == channel]
        # 散点图
        figure_plot_reg(answer_df_channel_second, subintent, botactname_dic,
                        r'../graphtime/reg_std_2' + "_" + channel + "_" + subintent, chnl_step_dict, channel)
        # 柱状图
        figure_plot_bar(answer_df_channel_second, subintent, botactname_dic, chnl_step_dict, channel,
                        r'../graphtime/reg_pc_2' + "_" + channel + "_" + subintent, False, "")





def time_statistic3_useract_choice(dialogs_info, botactname_dic, subintent, channel, choice_l):
    """
    每一步的下一步用户行为反馈统计和每一种用户行为的响应时间分布图（原始柱状拟合）
    :param dialogs_info: 数据源
    :param botactname_dic: botactid——botactname的映射
    :param subintent: 要统计的subintent
    :param channel: 要统计的channel
    :param choice_l: 1 只统计第一次进入 2 统计第二次及以后 12：1和2 都统计
    :return: 每一步的下一步用户行为数量分布和每一个用户行为的时间分布
    """

    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []
    answer_second_list = []
    chnl_step_dict = {}

    for key in dialogs_info:
        # if str(key) == '4075f60181e844b2b220d0459b52075f':
        #     print str(key)
        answer_time_dic = {}
        answer_time_dic_second = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_step = ""
        last_answer_name = ""
        for turn in session:
            kg_step_dict_list = []
            subintent_list.append(turn.subintent)
            if turn.subintent == subintent:
                if turn.chnl not in chnl_step_dict.keys():
                    if len(kg_step_dict_list) == 0 and len(turn.answer_step) > 0 and len(turn.kgResult) > 0:
                        kg_result = json.loads(turn.kgResult)['result']
                        if "answerPart" in kg_result.keys():
                            result_k = kg_result['answerPart']
                            if "steps" in result_k.keys():
                                steps_k = result_k['steps']
                                if len(steps_k) > 0:
                                    for step in steps_k:
                                        kg_step_dict = {}
                                        kg_step_dict['stepStandardCode'] = step['stepStandardCode']
                                        kg_step_dict['stepStandardName'] = step['stepStandardName']
                                        kg_step_dict['stepOrder'] = step['stepOrder']
                                        kg_step_dict_list.append(kg_step_dict)
                                    chnl_step_dict[turn.chnl] = kg_step_dict_list


            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                b_time = b_time.seconds
                # 取对数放大
                # if b_time.seconds > 0:
                #     b_time = math.log(b_time.seconds)
                # elif b_time.seconds == 0:
                #     tmp = round(float(b_time.microseconds)/1000000, 2)
                #     print tmp
                #     if tmp > 0:
                #         b_time = round(math.log(tmp), 2)
                #     else:
                #         b_time = 0
            b_time_start = turn.generator_time
            time_list.append(b_time)
            if len(last_answer_name) > 0 and subintent_show in last_answer_name:
                if len(last_answer_step) == 0:
                    # ts单步答案的处理（answer_name代替answer_step)
                    if last_answer_name in answer_time_dic:
                        answer_time_dic_second.setdefault(last_answer_name, []).append(turn.user_query+":"+str(b_time))
                    else:
                        answer_time_dic.setdefault(last_answer_name, []).append(turn.user_query+":"+str(b_time))
                else:
                    if last_answer_step in answer_time_dic:
                        answer_time_dic_second.setdefault(last_answer_step, []).append(turn.user_query+":"+str(b_time))
                    else:
                        answer_time_dic.setdefault(last_answer_step, []).append(turn.user_query+":"+str(b_time))
                answer_time_dic['serviceid'] = turn.serviceid
                answer_time_dic['chnl'] = turn.chnl
                # answer_time_dic['useract'] = turn.user_query

                answer_time_dic_second['serviceid'] = turn.serviceid
                answer_time_dic_second['chnl'] = turn.chnl
                # answer_time_dic_second.setdefault('useract', []).append(turn.user_query)
                # answer_time_dic_second['useract'] = turn.user_query

            last_answer_step = turn.answer_step_name
            last_answer_name = turn.answer_name

        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
        if len(answer_time_dic_second) > 0:
            answer_second_list.append(answer_time_dic_second)
    # print answer_list
    answer_df = pd.DataFrame(answer_list)
    answer_second_df = pd.DataFrame(answer_second_list)
    # answer_df.to_excel('../graphtime/answer_step_useract_' + subintent + '.xlsx')
    # answer_second_df.to_excel('../graphtime/answer_step_useract_second_' + subintent + '.xlsx')
    answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]
    answer_df_channel_second = answer_second_df.loc[answer_second_df['chnl'] == channel]
    if choice_l == 1:
        figure_plot_useract_bar(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_useract_1' + "_" + channel + "_" + subintent)
    if choice_l == 2:
        figure_plot_useract_bar(answer_df_channel_second, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_useract_2' + "_" + channel + "_" + subintent)
    if choice_l == 12:
        # 同时输出第一次和第二次及以后数据
        figure_plot_useract_bar(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_useract_1_' + "_" + channel + "_" + subintent)
        figure_plot_useract_bar(answer_df_channel_second, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_useract_2_' + "_" + channel + "_" + subintent)


def time_statistic3_log(dialogs_info, botactname_dic, subintent, channel):
    """
    log放大后的时间分布图(不区分第一次和以后进入）
    :param dialogs_info: 数据源
    :param botactname_dic: botactid-botactname映射
    :param subintent: 要统计的意图
    :param channel: 要统计的渠道
    :return:
    """
    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []
    kg_step_dict_list = []
    chnl_step_dict = {}
    for key in dialogs_info:
        # if str(key) == '4075f60181e844b2b220d0459b52075f':
        #     print str(key)
        answer_time_dic = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_step = ""
        last_answer_name = ""

        for turn in session:
            subintent_list.append(turn.subintent)
            kg_step_dict_list = []
            if turn.subintent == subintent:
                if turn.chnl not in chnl_step_dict.keys():
                    if len(kg_step_dict_list) == 0 and len(turn.answer_step) > 0 and len(turn.kgResult) > 0:
                        kg_result = json.loads(turn.kgResult)['result']
                        if "answerPart" in kg_result.keys():
                            result_k = kg_result['answerPart']
                            if "steps" in result_k.keys():
                                steps_k = result_k['steps']
                                if len(steps_k) > 0:
                                    for step in steps_k:
                                        kg_step_dict = {}
                                        kg_step_dict['stepStandardCode'] = step['stepStandardCode']
                                        kg_step_dict['stepStandardName'] = step['stepStandardName']
                                        kg_step_dict['stepOrder'] = step['stepOrder']
                                        kg_step_dict_list.append(kg_step_dict)
                                    chnl_step_dict[turn.chnl] = kg_step_dict_list

            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                # b_time = b_time.seconds
                # 取对数放大
                if b_time.seconds > 0:
                    b_time = round(math.log(b_time.seconds), 2)
                elif b_time.seconds == 0:
                    tmp = float(b_time.microseconds)/1000000
                    if tmp > 0:
                        b_time = round(math.log(tmp), 2)
                    else:
                        b_time = 0

            b_time_start = turn.generator_time
            time_list.append(b_time)
            if len(last_answer_name) > 0 and subintent_show in last_answer_name:
                if len(last_answer_step) == 0:
                    # ts单步答案的处理（answer_name代替answer_step)
                    answer_time_dic.setdefault(last_answer_name, []).append(b_time)
                else:
                    answer_time_dic.setdefault(last_answer_step, []).append(b_time)
                answer_time_dic['serviceid'] = turn.serviceid
                answer_time_dic['chnl'] = turn.chnl


            last_answer_step = turn.answer_step_name
            last_answer_name = turn.answer_name


        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
    # print answer_list
    answer_df = pd.DataFrame(answer_list)
    answer_df.to_excel('../graphtime/answer_step_log_' + subintent + '.xlsx')
    answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]


    # figure_plot_reg(answer_df_pc, subintent, botactname_dic, r'../graphtime/reg_' + subintent)
    # figure_plot_bar_log(answer_df_pc, subintent, botactname_dic, chnl_step_dict, 'pc', r'../graphtime/reg_log_pc_' + subintent)
    figure_plot_bar_log(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_log_' + channel + "_" + subintent)


def time_statistic3_log_choice(dialogs_info, botactname_dic, subintent, channel, choice_l):
    """
    log放大后的时间分布图(区分第一次和以后进入）
    :param dialogs_info: 数据源
    :param botactname_dic: botactid-botactname映射
    :param subintent: 要统计的意图
    :param channel: 要统计的渠道
    :param choice_l: 选择的方式： 1 只统计第一次进入 2 只统计第二次进入 12：1和2全部统计
    :return:时间分布统计图表
    """

    if "ts__" in subintent:
        subintent_show = subintent.replace('ts__', '')
    answer_list = []
    kg_step_dict_list = []
    answer_second_list = []
    chnl_step_dict = {}
    for key in dialogs_info:
        answer_time_dic = {}
        answer_time_dic_second = {}
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_answer_step = ""
        last_answer_name = ""

        for turn in session:
            subintent_list.append(turn.subintent)
            kg_step_dict_list = []
            if turn.subintent == subintent:
                if turn.chnl not in chnl_step_dict.keys():
                    if len(kg_step_dict_list) == 0 and len(turn.answer_step) > 0 and len(turn.kgResult) > 0:
                        kg_result = json.loads(turn.kgResult)['result']
                        if "answerPart" in kg_result.keys():
                            result_k = kg_result['answerPart']
                            if "steps" in result_k.keys():
                                steps_k = result_k['steps']
                                if len(steps_k) > 0:
                                    for step in steps_k:
                                        kg_step_dict = {}
                                        kg_step_dict['stepStandardCode'] = step['stepStandardCode']
                                        kg_step_dict['stepStandardName'] = step['stepStandardName']
                                        kg_step_dict['stepOrder'] = step['stepOrder']
                                        kg_step_dict_list.append(kg_step_dict)
                                    chnl_step_dict[turn.chnl] = kg_step_dict_list

            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                # b_time = b_time.seconds
                # 取对数放大
                if b_time.seconds > 0:
                    b_time = round(math.log(b_time.seconds), 2)
                elif b_time.seconds == 0:
                    tmp = float(b_time.microseconds) / 1000000
                    if tmp > 0:
                        b_time = round(math.log(tmp), 2)
                    else:
                        b_time = 0

            b_time_start = turn.generator_time
            time_list.append(b_time)
            if len(last_answer_name) > 0 and subintent_show in last_answer_name:
                if len(last_answer_step) == 0:
                    # ts单步答案的处理（answer_name代替answer_step)
                    if last_answer_name in answer_time_dic:
                        answer_time_dic_second.setdefault(last_answer_name, []).append(b_time)
                    else:
                        answer_time_dic.setdefault(last_answer_name, []).append(b_time)
                else:
                    if last_answer_step in answer_time_dic:
                        answer_time_dic_second.setdefault(last_answer_step, []).append(b_time)
                    else:
                        answer_time_dic.setdefault(last_answer_step, []).append(b_time)
                if len(answer_time_dic) > 0:
                    answer_time_dic['serviceid'] = turn.serviceid
                    answer_time_dic['chnl'] = turn.chnl
                # answer_time_dic['useract'] = turn.user_query
                if len(answer_time_dic_second) > 0:
                    answer_time_dic_second['serviceid'] = turn.serviceid
                    answer_time_dic_second['chnl'] = turn.chnl
                # answer_time_dic_second.setdefault('useract', []).append(turn.user_query)
                # answer_time_dic_second['useract'] = turn.user_query

            last_answer_step = turn.answer_step_name
            last_answer_name = turn.answer_name

        if len(answer_time_dic) > 0:
            answer_list.append(answer_time_dic)
        if len(answer_time_dic_second) > 0:
            answer_second_list.append(answer_time_dic_second)

    answer_df = pd.DataFrame(answer_list)
    answer_second_df = pd.DataFrame(answer_second_list)
        # answer_df.to_excel('../graphtime/answer_step_log_' + subintent + '.xlsx')
    if choice_l == 1:

        answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]
        figure_plot_bar_log(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_log_1_' + channel + "_" + subintent)
    if choice_l == 2:

        answer_df_channel_second = answer_second_df.loc[answer_df['chnl'] == channel]
        figure_plot_bar_log(answer_df_channel_second, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_log_2_' + channel + "_" + subintent)

    if choice_l == 12:
        answer_df_channel = answer_df.loc[answer_df['chnl'] == channel]
        figure_plot_bar_log(answer_df_channel, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_log_1_' + channel + "_" + subintent)
        answer_df_channel_second = answer_second_df.loc[answer_df['chnl'] == channel]
        figure_plot_bar_log(answer_df_channel_second, subintent, botactname_dic, chnl_step_dict, channel, r'../graphtime/reg_log_2_' + channel + "_" + subintent)



def multi_intent_statistic(dialogs_info, botactname_dic, channel, subintent):
    """
    多意图选择时间统计（初版未修改）暂时不用
    :param dialogs_info:
    :param botactname_dic:
    :param channel:
    :param subintent:
    :return:
    """

    multi_intent_time_dic = {}
    multi_intent_list = []
    for key in dialogs_info:
        b_time_start = 0
        session = dialogs_info[key]
        time_list = []
        subintent_list = []
        last_multi_intent = ""
        for turn in session:
            # if turn.chnl != channel:
            #     break
            subintent_list.append(turn.subintent)
            if (b_time_start == 0):
                b_time = 0
            else:
                time_end = (turn.generator_time) / 1000.0
                time_start = (b_time_start) / 1000.0
                b_time = datetime.utcfromtimestamp(time_end) - datetime.utcfromtimestamp(time_start)
                b_time = b_time.seconds
                # 取对数放大
                # if b_time.seconds > 0:
                #     b_time = math.log(b_time.seconds)
                # elif b_time.seconds == 0:
                #     tmp = round(float(b_time.microseconds)/1000000, 2)
                #     print tmp
                #     if tmp > 0:
                #         b_time = round(math.log(tmp), 2)
                #     else:
                #         b_time = 0
            b_time_start = turn.generator_time
            time_list.append(b_time)

            if len(last_multi_intent) > 0:
                # multi_intent_time_dic.setdefault(last_multi_intent, []).append(b_time)
                if last_multi_intent in multi_intent_time_dic:
                    multi_intent_time_dic[last_multi_intent] += (',' + str(b_time))
                else:
                    multi_intent_time_dic[last_multi_intent] = str(b_time)
                # multi_intent_time_dic['serviceid'] = turn.serviceid
                # multi_intent_time_dic['chnl'] = turn.chnl

            last_multi_intent = turn.multi_subintent

        # if len(multi_intent_time_dic) > 0:
        #     multi_intent_list.append(multi_intent_time_dic)
    # print answer_list
    answer_df = pd.Series(multi_intent_time_dic)
    answer_df.to_excel('../graphtime/multi_intent_' + channel +'.xlsx')



if __name__ == "__main__":
    importlib.reload(sys)

    botactname_dic = {}
    cases = {}

    botactname_dic = get_botact_data()
    # us数据
    # datapath = r'D:\logdata\2.xlsx'
    # cases = get_turn_data(datapath)
    # 全部数据
    cases = get_all_turn_data(r'D:\logdata\\')
    dialogs_info = get_turn_structure_data(cases, botactname_dic)

    # 1. 根据意图筛选日志并输出graphviz图
    # get_user_bot_act_route_time_pil_subintent(dialogs_info, botactname_dic, 'ts__battery_drains_fast', r'D:\dialoggraph0928\dialog')
    # get_user_bot_act_route_time_pil_subintent(dialogs_info, botactname_dic, 'ts__can_t_make_a_call')
    # get_user_bot_act_route_time_pil_subintent(dialogs_info, botactname_dic, 'ts__message_issue')
    # 2. 输出指定serviceid的graphviz图
    # get_user_bot_act_route_time_pil_serviceid(dialogs_info, botactname_dic, '4075f60181e844b2b220d0459b52075f', r'D:\dialoggraph0928serviceid\dialog')
    # 3. 不区分第一次和以后进入
    # time_statistic3(dialogs_info, botactname_dic, 'ts__battery_drains_fast', 'pc')
    # time_statistic3(dialogs_info, botactname_dic, 'ts__phone_charges_slow', 'pc')
    # 4. 区分第一次和以后进入的时间密度分布图（可根据反馈时间筛选日志（可选)
    # time_statistic3_choice(dialogs_info, botactname_dic, 'ts__battery_drains_fast', 'mobile', 12)
    # 5. 区分第一次和以后进入的自然对数放大后的时间密度分布图
    # time_statistic3_log_choice(dialogs_info, botactname_dic, 'ts__phone_charges_slow', 'pc', 12)
    # 6. 区分第一次和以后进入的用户行为数量分布和每一个行为时间密度分布图
    time_statistic3_useract_choice(dialogs_info, botactname_dic, 'ts__phone_charges_slow', 'mobile', 12)

    # 多意图选择（暂时不用）
    # multi_intent_statistic(dialogs_info, botactname_dic, 'pc', 'ts__rear_camera_broken')
