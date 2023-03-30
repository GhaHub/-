#本部分代码用于为卫星数据差值
import numpy as np
from scipy.interpolate import interp1d, Rbf
from scipy.interpolate import griddata
from pyhdf.SD import SD, SDC
import calendar, h5py, os, time
import matplotlib.pyplot as plt
import matplotlib, sys
import multiprocessing as mp
from scipy.stats import find_repeats
from tqdm import *
from colorama import Fore

# 加载默认数据
# Sup的100层气压
PSup_axis = np.asarray(
    [0.0161, 0.0384, 0.0769, 0.1370, 0.2244, 0.3454, 0.5064, 0.7140, 0.9753, 1.2972, 1.6872, 2.1526, 2.7009, 3.3398, 4.0770, 4.9204, 5.8776, 6.9567, 8.1655, 9.5119, 11.0038, 12.6492, 14.4559, 16.4318, 18.5847, 20.9224, 23.4526, 26.1829, 29.1210, 32.2744, 35.6505, 39.2566, 43.1001, 47.1882, 51.5278, 56.1260, 60.9895, 66.1253, 71.5398, 77.2396, 83.2310, 89.5204, 96.1138, 103.0170, 110.2370, 117.7770, 125.6460, 133.8460, 142.3850, 151.2660, 160.4960, 170.0780, 180.0180, 190.3200, 200.9890,
     212.0280, 223.4410, 235.2340, 247.4080, 259.9690, 272.9190, 286.2620, 300.0000, 314.1370, 328.6750, 343.6180, 358.9660, 374.7240, 390.8930, 407.4740, 424.4700, 441.8820, 459.7120, 477.9610, 496.6300, 515.7200, 535.2320, 555.1670, 575.5250, 596.3060, 617.5110, 639.1400, 661.1920, 683.6670, 706.5650, 729.8860, 753.6280, 777.7900, 802.3710, 827.3710, 852.7880, 878.6200, 904.8660, 931.5240, 958.5910, 986.0670, 1013.9500, 1042.2300, 1070.9200, 1100.0000], dtype=np.float32)
# Std的24层气压
PStd_axis = np.asarray([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 15, 10, 7, 5, 3, 2, 1.5, 1], dtype=np.float32)
# 需要拟合的76层高度
HMod_axis = np.asarray(
    [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000, 52000, 54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000, 70000, 72000, 74000, 76000, 78000, 80000, 82000, 84000, 86000, 88000, 90000, 92000,
     94000, 96000, 98000, 100000], dtype=np.float32)
# 重力加速度随纬度变化
grav_data = np.asarray(
    [9.780327, 9.780343, 9.78039, 9.780468, 9.780579, 9.780719, 9.780891, 9.781094, 9.781327, 9.78159, 9.781884, 9.7822075, 9.782559, 9.78294, 9.78335, 9.783787, 9.784251, 9.784742, 9.785259, 9.785802, 9.78637, 9.786974, 9.787577, 9.788215, 9.788875, 9.7895565, 9.790257, 9.790978, 9.791718, 9.792475, 9.793249, 9.794039, 9.794844, 9.795663, 9.7964945, 9.7973385, 9.798193, 9.799057, 9.799931, 9.800811, 9.801699, 9.802592, 9.80349, 9.804392, 9.805295, 9.8062, 9.807105, 9.808009, 9.808911, 9.80981,
     9.810704, 9.811593, 9.812476, 9.813352, 9.814218, 9.815075, 9.815921, 9.816756, 9.817577, 9.818385, 9.819179, 9.819957, 9.820717, 9.821461, 9.8221855, 9.822895, 9.823576, 9.82424, 9.824882, 9.8255005, 9.826097, 9.826668, 9.827214, 9.827736, 9.82823, 9.828698, 9.829139, 9.829551, 9.829935, 9.83029, 9.830616, 9.830912, 9.831178, 9.831412, 9.831617, 9.831791, 9.831933, 9.832044, 9.832123, 9.8321705, 9.832186], dtype=np.float32)
#纬度格点插值得到不同纬度的重力加速度
lat_grid = np.linspace(-89.5, 89.5, 180, dtype=np.float32)
grav_lat = interp1d(np.arange(91), grav_data, kind='cubic')(np.abs(lat_grid)) / 9.80665

#地球赤道半径及极地半径，计算不同纬度的地球半径
radius_equator, radius_polar = 6378137.0, 6356752.3
radis_lat = (((radius_equator ** 2) * np.cos(lat_grid)) ** 2 + ((radius_polar ** 2) * np.sin(lat_grid)) ** 2) / ((radius_equator * np.cos(lat_grid)) ** 2 + (radius_polar * np.cos(lat_grid)) ** 2)

######################################################################
# 加载2005年1月1日的数据
day_year = lambda year: 366 if calendar.isleap(int(year)) else 365  #判断每年天数
day_month = lambda year, month: calendar.monthrange(int(year), int(month))[1]   #判断每月天数
make_dir = lambda path: os.makedirs(path) if (not os.path.exists(path)) else None   #建立文件夹


#本函数扫描给定路径下缺乏的日期数据等
def check_data(data_dict,year_list):
    lack_day = []
    for year in year_list:
        for month in range(1,13):
            for day in range(day_month(year,month)):
                date = '%04d.%02d.%02d'%(year,month,day+1)
                if date not in data_dict.keys():
                    lack_day.append(date)
    print('缺乏以下日期的数据:',lack_day)

#本函数扫描给定路径,并给出Std,Sup两个文件内所有日期及路径对应关系
def get_data_path(file_path):
    '''
    本函数扫描给定路径,并给出Std,Sup两个文件内所有日期及路径对应关系
    :param file_path: AirsData的文件路径
    :return: dict_stdsup: 返回字典中有Std,Sup,Com三个Key,Com是Std与Sup中共有的文件
    '''

    def is_valid_date(data_str):
        try:
            time.strptime(data_str, "%Y.%m.%d")
            return True
        except:
            return False

    global std_lack_ls, sup_lack_ls
    dict_std = {}
    dict_sup = {}
    dict_stdsup = {}
    dict_common = {}

    # 扫描文件路径,并将其按照日期存入字典
    for curDir, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".hdf") and file[:4] == 'AIRS':
                if is_valid_date(file[5:15]):
                    if file[19:25] == 'RetStd':
                        dict_std[file[5:15]] = os.path.join(curDir, file)
                    if file[19:25] == 'RetSup':
                        dict_sup[file[5:15]] = os.path.join(curDir, file)
    # 寻找Std与Sup两个文件夹下都有的日期,以及缺乏的日期
    common_date = list(set(dict_sup.keys()) & (set(dict_std.keys())))
    std_lack_ls = list(set(dict_sup.keys()).difference(set(dict_std.keys())))
    sup_lack_ls = list(set(dict_std.keys()).difference(set(dict_sup.keys())))

    if len(std_lack_ls):
        print('Std文件夹中缺乏如下日期数据:\n' + '\n'.join([date for date in std_lack_ls]))
    if len(sup_lack_ls):
        print('Sup文件夹中缺乏如下日期数据:\n' + '\n'.join([date for date in sup_lack_ls]))

    # 构造共有日期的字典
    for date in common_date:
        dict_common[date] = {'Std': dict_std[date], 'Sup': dict_sup[date]}

    dict_stdsup['Std'] = dict_std
    dict_stdsup['Sup'] = dict_sup
    dict_stdsup['Com'] = dict_common
    return dict_stdsup

# 本函数从势高度转化为几何高度
def gph2geh(gph_ary, radis_lat, grav_lat):
    assert gph_ary.shape[1] == radis_lat.shape[0] and gph_ary.shape[1] == grav_lat.shape[0]
    geh_ary = (radis_lat[np.newaxis, :, np.newaxis] * gph_ary) / (
            grav_lat[np.newaxis, :, np.newaxis] * radis_lat[np.newaxis, :, np.newaxis] - gph_ary)
    return geh_ary

#计算几何高度坐标下的温度气压值
def get_tmpas_vs_geh(gph_ary, tmp_ary, gph_pas, tmp_pas, hmod_alti, nan_ratio=(0.8, 0.8), kind='linear'):
    gph_log_pas = np.log(gph_pas)
    tmp_log_pas = np.log(tmp_pas)

    gph_nan_ratio = np.isnan(gph_ary).sum(axis=0) / gph_ary.shape[0]
    tmp_nan_ratio = np.isnan(tmp_ary).sum(axis=0) / tmp_ary.shape[0]

    # 对高度按照纬度进行差值以去除nan
    gph_ary = interp_layer_wape_nan(gph_ary, extd_ratio=(0.15, 0.1))
    # 对温度按照纬度进行插值去除nan
    tmp_ary[:-3] = interp_layer_wape_nan(tmp_ary[:-3], extd_ratio=(0.15, 0.1))

    # 消去所有Nan值,然后将24层气压的势高度拟合至100层的气压
    if np.isnan(gph_ary).any():
        gph_ary = interp_batch(gph_log_pas, gph_ary, None, kind=kind)   #高度去Nan
    gph_sup_ary = interp_batch(gph_log_pas, gph_ary, tmp_log_pas, kind=kind)    #一维线性插值
    geh_sup_ary = gph2geh(gph_sup_ary, radis_lat, grav_lat)

    # 对温度和气压同时进行插值处理
    tmp_ary[-3:] = tmp_ary[-4][np.newaxis,...]
    geh_sup_ary += 0.05 * np.random.randn(geh_sup_ary.shape[0],)[:,np.newaxis, np.newaxis]
    tmp_geh_ary = interp_batch(geh_sup_ary, tmp_ary, hmod_alti, kind='linear', scale=1e2)  # 对温度直接插值
    hund_level = np.linspace(0, geh_sup_ary[0].min(), 100)  # 对气压先插值到一定范围,然后再由该范围外插至0~100公里
    logpas_temp_ary = interp_batch(geh_sup_ary, tmp_log_pas[:,np.newaxis,np.newaxis]+np.zeros_like(geh_sup_ary), hund_level, kind='linear', scale=1e3)  #内插可以考虑cubic需要认真修改
    logpas_geh_ary = interp_batch(hund_level, logpas_temp_ary, hmod_alti, kind='linear')
    # logpas_geh_ary[~np.isnan(logpas_geh_ary)][logpas_geh_ary[~np.isnan(logpas_geh_ary)] > np.log(1100)] = np.log(1100)
    logpas_geh_ary = np.clip(logpas_geh_ary,None, np.log(1100))
    pas_geh_ary = np.exp(logpas_geh_ary)
    return geh_sup_ary, tmp_ary, gph_nan_ratio, tmp_nan_ratio, tmp_geh_ary, pas_geh_ary

# 此函数用于对同一层数据进行拟合以消除该层的nan数据
def interp_layer_wape_nan(layer_ary, extd_ratio=(0.3, 0.15),kind='linear'):
    '''
    :param layer_ary:   每层矩阵原始数据
    :param extd_ratio:  每层矩阵数据扩展
    :return:    返回插值后消除nan的数据
    '''
    ary_shape = layer_ary.shape
    if len(layer_ary.shape) == 2:
        layer_ary = layer_ary[np.newaxis, ...]
    #在数据外面补充额外的行和列以防止外差问题
    extd_col_num, extd_row_num = int(ary_shape[-1] * extd_ratio[0]), int(ary_shape[-2] * extd_ratio[1])
    extd_ary = np.concatenate((layer_ary[..., -extd_col_num:], layer_ary, layer_ary[..., :extd_col_num]), axis=-1)
    extd_ary = np.concatenate((extd_ary[:, -extd_row_num:], extd_ary, extd_ary[:, :extd_row_num]), axis=-2)
    extd_shape = extd_ary.shape
    extd_ary = extd_ary.reshape(-1)
    # 构造一维X坐标数组用于插值
    xaxis = (4 * (extd_shape[2] * extd_shape[1]) * np.arange(extd_shape[0])[:, np.newaxis, np.newaxis] + 2 * extd_shape[2] * np.arange(extd_shape[1])[np.newaxis, :, np.newaxis] + 0.5 * np.arange(extd_shape[2])[np.newaxis, np.newaxis, :]).reshape(-1)

    interp_func = interp1d(xaxis[~np.isnan(extd_ary)], extd_ary[~np.isnan(extd_ary)], kind=kind, bounds_error=False, fill_value=np.nan)
    xfit = xaxis.reshape(extd_shape)[:, extd_row_num:-extd_row_num, extd_col_num:-extd_col_num].reshape(-1)
    yfit = extd_ary.reshape(extd_shape)[:, extd_row_num:-extd_row_num, extd_col_num:-extd_col_num].reshape(-1)
    yfit[np.isnan(yfit)] = interp_func(xfit[np.isnan(yfit)])

    #如仍存在nan数据，则采用最近邻方式插值。
    if np.isnan(yfit).any():
        yfit[np.isnan(yfit)] = interp1d(xfit[~np.isnan(yfit)], yfit[~np.isnan(yfit)], kind='nearest', bounds_error=False, fill_value='extrapolate')(xfit[np.isnan(yfit)])
    yfit = yfit.reshape(ary_shape)
    assert (not np.isnan(yfit).any())
    return yfit

# 批量插值函数
def interp_batch(xaxis_ary, yaxis_ary, xfit_ary=None, scale=1, kind='linear', fill_val='extrapolate'):
    '''
    本函数用于批量拟合,其中又分为四种情况:
    情况1: xfit为None,yaxis中无nan,直接返回;
    情况2: xfit与xaxis均为一维,yaxis中无nan,无需展开直接1维插值,注意axis=-3;
    情况3: xfit为None,xaxis与yaxis都要展开,且无效值的坐标应该为或的关系;
    情况4: xfit非None,需要yaxis与xaxis展开成1维作用于xfit的1维展开;

    :param xaxis_ary: 自变量矩阵,一维或者三维
    :param yaxis_ary: 值矩阵
    :param xfit_ary: 需要拟合的自变量
    :param scale:   一层结束到另一层时开始时坐标差参数
    :param kind:
    :param fill_val:
    :return:
    '''

    assert len(yaxis_ary.shape) >= 3 and xaxis_ary.shape[0] == yaxis_ary.shape[-3]
    assert xaxis_ary.shape == yaxis_ary.shape[-3:] or len(xaxis_ary.shape) == 1

    if xfit_ary is None:
        range_per_lay = xaxis_ary.max() - xaxis_ary.min()
    else:
        range_per_lay = max(xaxis_ary.max() - xaxis_ary.min(), xfit_ary.max() - xfit_ary.min())

    yaxis_shape = yaxis_ary.shape

    # yaxis_ary中无nan数据
    if not np.isnan(yaxis_ary).any():
        if xfit_ary is None:
            # print('情况1: xfit为None,yaxis中无nan,直接返回;')
            return yaxis_ary
        # 无nan数据,且自变量均为一维事直接1维插值
        elif len(xaxis_ary.shape) == 1 and len(xfit_ary.shape) == 1:
            yfit_ary = interp1d(xaxis_ary, yaxis_ary, axis=-3, kind=kind, bounds_error=False, fill_value=fill_val)(xfit_ary)
            # print('情况2: xfit与xaxis均为一维,yaxis中无nan,无需展开直接1维插值,注意axis=-3;')
            return yfit_ary

    # 构造出4维的yaxis1d以及1维xaxis1d
    yaxis1d_ary = yaxis_ary.reshape(-1, np.prod(yaxis_shape[-3:]))
    nan_index = (np.isnan(yaxis1d_ary).sum(axis=0) > 0).reshape(-1)

    # 构造1维xaxis1d
    if len(xaxis_ary.shape) == 1:
        xaxis1d_ary = (xaxis_ary[:, np.newaxis, np.newaxis] + scale * range_per_lay * np.arange(np.prod(yaxis_shape[-2:]), dtype=np.float64).reshape(1, *yaxis_shape[-2:])).reshape(-1)
    elif xaxis_ary.shape == yaxis_shape[-3:]:
        xaxis1d_ary = (xaxis_ary + scale * range_per_lay * np.arange(np.prod(yaxis_shape[-2:]), dtype=np.float64).reshape(1, *yaxis_shape[-2:])).reshape(-1)

    if xfit_ary is None:
        # print('情况3: xfit为None,xaxis与yaxis都要展开,且无效值的坐标应该为或的关系;')
        yaxis1d_ary[:, nan_index] = interp1d(xaxis1d_ary[~nan_index], yaxis1d_ary[:,~nan_index],axis=-1,kind=kind,bounds_error=False,fill_value=fill_val)(xaxis1d_ary[nan_index])
        yfit_ary = yaxis1d_ary.reshape(yaxis_shape)
    else:
        #对xfit展开称1维进行差值
        if len(xfit_ary.shape) == 1:
            xfit1d_ary = (xfit_ary[:,np.newaxis,np.newaxis]+scale*range_per_lay*np.arange(np.prod(yaxis_shape[-2:]),dtype=np.float64).reshape(1,*yaxis_shape[-2:])).reshape(-1)
        elif len(xfit_ary.shape) == 3 and xfit_ary.shape[-2:] == yaxis_shape[-2:]:
            xfit1d_ary = (xfit_ary+scale*range_per_lay*np.arange(np.prod(yaxis_shape[-2:]),dtype=np.float64).reshape(1,*yaxis_shape[-2:])).reshape(-1)
        # print('情况4: xfit非None,需要yaxis与xaxis展开成1维作用于xfit的1维展开;')
        try:
            yfit_ary = interp1d(xaxis1d_ary[~nan_index], yaxis1d_ary[:, ~nan_index], axis=-1, kind=kind, bounds_error=False, fill_value=fill_val)(xfit1d_ary)
        except:
            repeat_data = find_repeats(xaxis1d_ary)
            for repeat_val in repeat_data[0]:
                repeat_index = np.where(xaxis1d_ary == repeat_val)[0]
                nan_index[repeat_index] = True
            yfit_ary = interp1d(xaxis1d_ary[~nan_index], yaxis1d_ary[:, ~nan_index], axis=-1, kind=kind, bounds_error=False, fill_value=fill_val)(xfit1d_ary)
        yfit_ary = yfit_ary.reshape(-1, xfit_ary.shape[0], *yaxis_shape[-2:])
        if yfit_ary.shape[0] == 1:
            yfit_ary = yfit_ary[0]
    return yfit_ary.astype(yaxis_ary.dtype)

def get_vablay_range(ary, nan_lay_ratio):
    nanum_per_lay = np.isnan(ary).sum(axis=(-1, -2))
    layvab_index = np.where(nanum_per_lay < ary.shape[1] * ary.shape[2] * nan_lay_ratio)[0]
    return (layvab_index.min(), layvab_index.max())

def create_h5dataset(h5file, year, month, date_path_dict, gph_pas, tmp_pas, hmod_alti, grid_shape):
    tot_dates = calendar.monthrange(int(year), int(month))[1]

    # 构造数据集
    # 插值到76层高度对应的温度
    tmpA_geh_dset = h5file.require_dataset('./%04d/%02d/Mod/TAirGeH/ascending' % (year, month), shape=(tot_dates, hmod_alti.shape[0], *grid_shape), chunks=(1, 1, *grid_shape), dtype=np.float32, fillvalue=np.nan)
    # 插值到76层高度对应的气压
    pasA_geh_dset = h5file.require_dataset('./%04d/%02d/Mod/PAirGeH/ascending' % (year, month), shape=(tot_dates, hmod_alti.shape[0], *grid_shape), chunks=(1, 1, *grid_shape), dtype=np.float32, fillvalue=np.nan)
    # nan数量
    gphA_nan_ratio = h5file.require_dataset('./%04d/%02d/GPHNanRatio/ascending' % (year, month), shape=(tot_dates, *grid_shape), chunks=(1, *grid_shape), dtype=np.float32, fillvalue=np.nan)
    gphA_nan_ratio.attrs['nan_ratio_grid'] = gph_pas.shape[0]
    tmpA_nan_ratio = h5file.require_dataset('./%04d/%02d/TAirNanRatio/ascending' % (year, month), shape=(tot_dates, *grid_shape), chunks=(1, *grid_shape), dtype=np.float32, fillvalue=np.nan)
    tmpA_nan_ratio.attrs['nan_ratio_grid'] = tmp_pas.shape[0]
    # 插值到76层高度对应的温度
    tmpD_geh_dset = h5file.require_dataset('./%04d/%02d/Mod/TAirGeH/decending' % (year, month), shape=(tot_dates, hmod_alti.shape[0], *grid_shape), chunks=(1, 1, *grid_shape), dtype=np.float32, fillvalue=np.nan)
    # 插值到76层高度对应的气压
    pasD_geh_dset = h5file.require_dataset('./%04d/%02d/Mod/PAirGeH/decending' % (year, month), shape=(tot_dates, hmod_alti.shape[0], *grid_shape), chunks=(1, 1, *grid_shape), dtype=np.float32, fillvalue=np.nan)
    # nan数量
    gphD_nan_ratio = h5file.require_dataset('./%04d/%02d/GPHNanRatio/decending' % (year, month), shape=(tot_dates, *grid_shape), chunks=(1, *grid_shape), dtype=np.float32, fillvalue=np.nan)
    gphD_nan_ratio.attrs['nannum'] = gph_pas.shape[0]
    tmpD_nan_ratio = h5file.require_dataset('./%04d/%02d/TAirNanRatio/decending' % (year, month), shape=(tot_dates, *grid_shape), chunks=(1, *grid_shape), dtype=np.float32, fillvalue=np.nan)
    tmpD_nan_ratio.attrs['nannum'] = tmp_pas.shape[0]
    # 创造并添加标尺
    date_scale = h5file.require_dataset('./%04d/%02d/date' % (year, month), shape=(tot_dates,), dtype=np.dtype('S10'))
    date_scale[:] = np.asarray(['%04d.%02d.%02d' % (year, month, date) for date in range(1, 1 + tot_dates)]).astype(np.dtype('S10'))
    date_scale.make_scale('日期')

    alti_scale = h5file.require_dataset('./%04d/%02d/alti' % (year, month), shape=hmod_alti.shape, dtype=np.float32)
    alti_scale[:] = hmod_alti.astype(np.float32)
    alti_scale.make_scale('插值海拔')
    lat_scale = h5file.require_dataset('./%04d/%02d/lat' % (year, month), shape=(grid_shape[0],), dtype=np.float32)
    lat_scale[:] = np.linspace(89.5, -89.5, 180, dtype=np.float32)
    lat_scale.make_scale('纬度')
    lon_scale = h5file.require_dataset('./%04d/%02d/lon' % (year, month), shape=(grid_shape[1],), dtype=np.float32)
    lon_scale[:] = np.linspace(-179.5, 179.5, 360, dtype=np.float32)
    lon_scale.make_scale('经度')
    # std24层气压标尺
    stdpas_scale = h5file.require_dataset('./%04d/%02d/std_pressure' % (year, month), shape=(gph_pas.shape[0],), dtype=np.float32)
    stdpas_scale[:] = gph_pas
    stdpas_scale.make_scale('Std气压')

    date_scale_valid = h5file.require_dataset('./%04d/%02d/date_valid' % (year, month), shape=(tot_dates,), dtype=bool)
    date_scale_valid[:] = np.asarray([True if '%04d.%02d.%02d' % (year, month, date + 1) in date_path_dict.keys() else False for date in range(tot_dates)], dtype=bool)
    date_scale_valid.make_scale('有效日期')
    # 添加标尺
    for dset in [tmpA_geh_dset, tmpD_geh_dset, pasA_geh_dset, pasD_geh_dset]:
        dset.dims[0].attach_scale(date_scale)  # 日期
        dset.dims[0].attach_scale(date_scale_valid)  # 日期
        dset.dims[1].attach_scale(alti_scale)  # 气压
        dset.dims[2].attach_scale(lat_scale)  # 纬度
        dset.dims[3].attach_scale(lon_scale)  # 经度
    for dset in [gphA_nan_ratio, gphD_nan_ratio, tmpA_nan_ratio, tmpD_nan_ratio]:
        dset.dims[0].attach_scale(date_scale)  # 日期
        dset.dims[0].attach_scale(date_scale_valid)  # 日期
        dset.dims[1].attach_scale(lat_scale)  # 纬度
        dset.dims[2].attach_scale(lon_scale)  # 经度

    tmp_geh_group = h5file['./%04d/%02d/Mod/TAirGeH/' % (year, month)]
    pas_geh_group = h5file['./%04d/%02d/Mod/PAirGeH/' % (year, month)]
    gph_nan_group = h5file['./%04d/%02d/GPHNanRatio/' % (year, month)]
    tmp_nan_group = h5file['./%04d/%02d/TAirNanRatio/' % (year, month)]

    return h5file, tmp_geh_group, pas_geh_group, gph_nan_group, tmp_nan_group

def calc_month_data(year, month, date_path_dict, pbar_pos, file_path='F:/GARIM/Interp', gph_pas=PStd_axis, tmp_pas=PSup_axis, hmod_alti=HMod_axis, gph_kind='linear', grid_shape=(180, 360)):
    year, month = int(year), int(month)
    # 根据年份建立相应文件夹
    tot_dates = calendar.monthrange(int(year), int(month))[1]
    pbar = tqdm(range(tot_dates), position=pbar_pos + 2, desc='%4d年%2d月' % (year, month), leave=True, miniters=0.1, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.LIGHTBLUE_EX, Fore.RESET))
    # 根据年份建立相应文件夹
    os.makedirs(os.path.join(file_path, '%4d' % year),exist_ok=True)
    try:
        hdf5file = h5py.File(os.path.join(file_path, '%4d/%04d.%02d.hdf5' % (year, year, month)), 'a')
    except:
        os.remove(os.path.join(file_path, '%4d/%04d.%02d.hdf5' % (year, year, month)))
        hdf5file = h5py.File(os.path.join(file_path, '%4d/%04d.%02d.hdf5' % (year, year, month)), 'a')
    
    # 创建hdf5文件
    with hdf5file as h5file:
        # 创建数据集
        h5file, tmp_geh_group, pas_geh_group, gph_nan_group, tmp_nan_group = create_h5dataset(h5file, year, month, date_path_dict, gph_pas, tmp_pas, hmod_alti, grid_shape)
        # 开始处理
        for date in range(tot_dates):
            date_key = '%04d.%02d.%02d' % (year, month, date + 1)
            if date_key not in date_path_dict.keys():
                #print('缺乏日期%s的数据'%date_key)
                continue
            # print('正在计算%s的数据;'%date_key)
            try:
                std_data, sup_data = SD(date_path_dict[date_key]['Std'], SDC.READ), SD(date_path_dict[date_key]['Sup'], SDC.READ)
            except:
                print('当前日期%s的HDF数据有误,请重新下载！'%date_key)
                continue
            
            # 处理昼夜数据
            for day_state in ['ascending', 'decending']:
                if day_state == 'ascending':
                    gph_ary = std_data.select('GPHeight_A').get()
                    tmp_ary = sup_data.select('TAirSup_A').get()
                elif day_state == 'decending':
                    gph_ary = std_data.select('GPHeight_D').get()
                    tmp_ary = sup_data.select('TAirSup_D').get()

                gph_ary[gph_ary < -999] = np.nan
                tmp_ary[tmp_ary < -999] = np.nan

                tot_result = get_tmpas_vs_geh(gph_ary, tmp_ary, gph_pas=gph_pas, tmp_pas=tmp_pas, hmod_alti=hmod_alti, nan_ratio=(0.5, 0.5), kind=gph_kind)
                gph_nan_group[day_state][date], tmp_nan_group[day_state][date], tmp_geh_group[day_state][date], pas_geh_group[day_state][date] = tot_result[2:]

            std_data.end()
            sup_data.end()
            pbar.update()
            # print('已完成%s的数据处理,此次用时%03.2f秒;'%(date_key,time.time()-ts))
        pbar.close()
        h5file.close()


if __name__ == '__main__':
    Airs_path = 'E:\AirsData'       #原始卫星数据路径
    Interp_Path = 'D:/GARIM/Interp_Data_ver01'      #插值数据路径
    
    ts = time.time()
    date_path_dict = get_data_path(Airs_path)['Com']
    year_list = range(2005,2015)
    check_data(date_path_dict,year_list)
    
    para_list, pbar_pos = [], 0     #pbar_pos为显示计算进度条所用,并无实际用处
    for year in year_list:
        for month in range(1, 13):
            para_list.append((year, month, date_path_dict, pbar_pos, Interp_Path))
            pbar_pos += 1
    
    mpool = mp.Pool(processes=26)
    ts = time.time()
    mpool.starmap_async(calc_month_data, para_list)
    mpool.close()
    mpool.join()
    te = time.time()
    print('总耗时%5.2f' % (te - ts))
    
    # print('处理2005年总耗时%.2f秒'%(te-ts))

