#! usr/bin/python3
import numpy as np
import pandas as pd
import os

train_vali_df = pd.read_csv('sales_train_validation.csv')                       #UTF-8编码格式csv文件数据读取,返回一个DataFrame的对象，这个是pandas的一个数据结构
train_eval_df = pd.read_csv('sales_train_evaluation.csv')
calendar_df = pd.read_csv('calendar.csv')
sell_prices_df = pd.read_csv('sell_prices.csv', iterator=True, chunksize=10000) #sell_prices.csv

################################################################################
def data_generate(data_frame):                                                  #train_vali_df
    """
    generate the sale_train data
    """

    np_array = np.array(data_frame.values.tolist())                             #将DataFrame转化为numpy数组

    name_list = []
    a = np.ones((6,), dtype=np.int)
    b = np.zeros((6,len(np_array)), dtype=np.int)
    c = np_array[:, 6:].astype(np.float)

    for i in range(6):                                                          # 初始化a[i,j]以及name_list
        name_list.append(np_array[0, i])

    dict1 = dict(zip(name_list, a))

    for j in range(6):
        count = 1
        for i in range(len(np_array)):

            if np_array[i, j] in dict1.keys():                                  # 如果有相同的键，则输出对应的值
                b[j, i] = dict1[str(np_array[i, j])]
            else:                                                               # 如果没有，则新增键值对，并把最近一次的a[i,j]加1，并作为新的值
                dict1[str(np_array[i, j])] = count + 1
                b[j, i] = dict1[str(np_array[i, j])]
                count += 1

    return b, c

def calendar_data(data_frame):                                                  #calendar_df
    """
    get the calendar data
    """

    np_array = np.array(data_frame.values.tolist())

    c0 = np.zeros(len(np_array), dtype=np.int)
    #创建周信息字典
    week_dict = {}
    #count设置为11101，因为price10000里不知道为什么k必须+11101，不能+1，不然zip之后丢数据
    count = 11101
    #用c0记录week_dict.values(),方便与sell_prices做对比
    for i in range(len(np_array)):

        if np_array[i, 1] in week_dict.keys():                                  # 如果有相同的键，则输出对应的值
            c0[i] = week_dict[str(np_array[i, 1])]
        else:                                                                   # 如果没有，则新增键值对，并把最近一次的a[i,j]加1，并作为新的值
            week_dict[str(np_array[i, 1])] = count
            c0[i] = week_dict[str(np_array[i, 1])]
            count += 1
    #1.直接读取日期值作为影响
    c1 = np_array[:, 3:6].astype(np.int)
    c2 = np.zeros((4, len(np_array)), dtype=np.int)
    #2.直接读取，输出SNAP影响
    c3 = np_array[:, 11:14].astype(np.int)

    #3.读取节日影响
    dict2 = {}
    for j in range(7, 11):
        count2 = 1
        for i in range(len(np_array)):
            if np_array[i, j] == 'nan':
                continue
            else:
                if np_array[i, j] in dict2.keys():
                    c2[j-7, i] = dict2[np_array[i, j]]
                else:
                    dict2[np_array[i, j]] = count2
                    c2[j-7, i] = dict2[np_array[i, j]]
                    count2 += 1

    #4.合并c0-3，得到c，并输出
    c = np.vstack((c0.T, c1.T, c2, c3.T))
    # print(len(c[0]))
    return c, week_dict

def price10000_data():
    data_frame = pd.read_csv('sell_prices.csv')
    np_array = np.array(data_frame.values.tolist())

    # print(max(np_array[:, 2]), min(np_array[:, 2]))

    store_dict = {}
    item_dict = {}
    price10000 = []
    key1 = []

    # 将第一列存入字典store_dict
    count1 = 1
    for i in range(len(np_array)):
        if np_array[i, 0] in store_dict.keys():
            continue
        else:
            store_dict[str(np_array[i, 0])] = count1
            count1 += 1

    # 将第二列存入字典item_dict
    count2 = 1
    for i in range(len(np_array)):
        if np_array[i, 1] in item_dict.keys():
            continue
        else:
            item_dict[str(np_array[i, 1])] = count2
            count2 += 1

    week_len = 282

    # 构造price = 10000的数组,并制作成字典
    for i in range(len(store_dict)):
        for j in range(len(item_dict)):
            for k in range(282):
                # print('k=', k)
                key1.append(str(i+1) + str(j+1) + str(k+11101))#k+1就不行，就只能+11101，为啥？
                price10000.append(649)

    price_dict = dict(zip(key1, price10000))

    return price_dict, store_dict, item_dict

def read_price(data_frame_piece, store_dict, item_dict, week_dict):
    np_array = np.array(data_frame_piece.values.tolist())

    # 给出元数据中的键值对
    str1 = []
    str2 = []
    key2 = []
    price = []
    for i in range(len(np_array)):
        if np_array[i, 0] in store_dict.keys():
            str1.append(store_dict[str(np_array[i, 0])])
            if np_array[i, 1] in item_dict.keys():
                if np_array[i, 2] in week_dict.keys():
                    str2.append(item_dict[str(np_array[i, 1])])

                    key2.append(str(str1[i]) + str(str2[i]) + str(week_dict[str(np_array[i, 2])]))
                    price.append(float(np_array[i, 3]))

    old_dict = dict(zip(key2, price))

    return old_dict

def get_price(sell_prices_df, calendar_df):
    """
    this function is to get the price info and return a row of normalized price
    """
    price_piece = []

    for chunk in sell_prices_df:
        price_piece.append(chunk)

    week_dict = calendar_data(calendar_df)[1]
    dict_result = price10000_data()
    # print(len(dict_result[0]))

    # lence = 0
    for i in range(len(price_piece)):
    # for i in range(2):
        old_dict = {}
        old_dict = read_price(price_piece[i], dict_result[1], dict_result[2], week_dict)
        dict_result[0].update(old_dict)
        old_dict.clear()
        # print(len(dict_result[0]))

    price = np.array(list(dict_result[0].values()))                             #price的标准化
    p = price.T
    p -= np.mean(p, axis=0)
    p /= np.std(p, axis=0)

    return p

def normalization(init_data):                                                   #price不在这标准化
    init_data = init_data.astype(float)

    for i in range(len(init_data)):
        init_data[i] -= np.mean(init_data[i], axis=0)
        init_data[i] /= np.std(init_data[i], axis=0)

    return init_data

def sales_train_days(train_vali_df, train_eval_df):
    c1 = data_generate(train_vali_df)[1]
    c2 = data_generate(train_eval_df)[1]
    c = np.hstack((c1,c2))

 
    c_mean = np.mean(c)
    c -= np.mean(c)
    # print('std=', np.std(c))
    c_std = np.std(c)
    c /= np.std(c)

    day_vali = c[:, :len(c1[0])]
    day_eval = c[:, len(c1[0]):]

    return day_vali, day_eval, c_mean, c_std

################################################################################
def main(train_df, train_info, sell_prices_df, calendar_df):
    #载入函数
    id_info = normalization(data_generate(train_df)[0])
    day_info = normalization(calendar_data(calendar_df)[0])
    price_info = get_price(sell_prices_df, calendar_df)
    # train_info = sales_train_days(train_vali_df, train_eval_df)

    #载入目录
    name_df = pd.read_csv('sales_train_validation.csv', usecols=['id'])
    name_id = np.array(name_df.values.tolist())
    newname_id = name_id.flatten()

    #做切片裁去day_info和id_info中的第一行
    id_info = id_info[1:, :]
    day_info = day_info[1:, :]

    #转置并设置小数位为4位
    id_info = np.around(id_info.T, decimals=4)
    day_info = np.around(day_info.T, decimals=4)
    price_info = np.around(price_info.T, decimals=8)
    #输出vali时用vali_info,输出eval时用eavl_info
    train_vali_info = np.around(train_info[0].T, decimals=8)
    train_eval_info = np.around(train_info[1].T, decimals=8)

    #创建每一个商品每天的销售价格
    day_each_price = np.zeros((len(id_info), 282, 7))
    final_info = []

    step1 = 282
    each_price = []
    for i in range(0,len(price_info),step1):
        c = price_info[i:i+step1]
        each_price.append(c)

    each_price = np.array(each_price)
    for i in range(len(id_info)):
        for j in range(282):
            for k in range(7):
                day_each_price[i, j, k] = each_price[i, j]

    #输出数据文件,打包成10个文件
    number_of_pack = 10
    pack_lence = int(len(id_info)/number_of_pack)

    print('pack lence =', pack_lence, ':begin output...')

    for o in range(number_of_pack):
        for i in range(o*pack_lence, (o+1)*pack_lence): #len(id_info)
            for j in range(282): #len(day_info)
                for k in range(7):
                    l = k + j*7
                    if l < 1941: # 1913/for evaluation, this number is 1941
                        final_info.append(np.hstack((id_info[i], day_each_price[i, j, k], day_info[l], train_eval_info[l, i])))
                    else:
                        break
            # print(len(final_info))
        output_file=pd.DataFrame(columns = None, data=final_info)
        output_file.to_csv('F:\\Mingda Li\\M5_Forecasting_Comp\\data\\eval_%d.csv'%(o), encoding='utf-8', index=False, header = None)
        final_info.clear()

        print('the file %d is complete'%(o))
    print('finished')

train_info = sales_train_days(train_vali_df, train_eval_df)
c_mean = train_info[2]
c_std = train_info[3]
print('train_info: load complete')
main(train_eval_df, train_info, sell_prices_df, calendar_df)
