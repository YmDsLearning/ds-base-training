from glob import glob
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

#_get_df() ※private関数には頭に'_'をつける
#指定されたフォルダのファイルリストをDataFrameにする
#param1: base_path - ルートパス
#param2: folder - ファイルが格納されているフォルダ名
#return: data_dict - ファイルパスとファイル名のDataFrame
def _get_df(base_path='public-covid-data', folder='rp_im'):
    data_dict = pd.DataFrame({'FilePath': glob('{}/{}/*'.format(base_path, folder)),
                              'FileName': [p.split('/')[-1] for p in glob('{}/{}/*'.format(base_path, folder))]})
    return data_dict


#get_df_all()
#_get_df()をコールしてDataFrame(完成形)を作成する
#param: base_path - ルートパス
#return: DataFrame完成形
def get_df_all(base_path='public-covid-data'):
    rp_im_df = _get_df(base_path, folder='rp_im')
    rp_msk_df = _get_df(base_path, folder='rp_msk')
    return rp_im_df.merge(rp_msk_df, on='FileName', suffixes=('Image', 'Mask'))    


#load_nifti()
#NIfTIデータをNumpy Arrayとしてロードする
#param: path - Numpy Arrayを取り出したい画像ファイル
#return: Numpy Array(回転済み)
def load_nifti(path):
    #NIfTIデータのロード→Numpy Arrayに変換
    data = nib.load(path).get_fdata()
    return np.rollaxis(data, axis=1)

#label_color()
#マスクデータをRGBにする
#1.値が全て0のRGBの箱を用意する
#2.ラベル別に値を入れる
#param1: mask_volume - 変換対象のマスクデータ
#param2: ggo_color - Red[255, 0, 0],
#param3: consolidation_colorz - Green[0, 255, 0],
#param4: effusion_color - Blue[0, 0, 255]):
#return: mask_color　 - RGB変換後のマスクデータ
def label_color(mask_volume,
                ggo_color=[255, 0, 0],
                consolidation_color=[0, 255, 0],
                effusion_color=[0, 0, 255]):

    #値が全て0のRGBの箱を用意する
    shp = mask_volume.shape
    mask_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    #色付け
    mask_color[np.equal(mask_volume, 1)] = ggo_color
    mask_color[np.equal(mask_volume, 2)] = consolidation_color
    mask_color[np.equal(mask_volume, 3)] = effusion_color

    return mask_color

#hu_to_gray()
#CTデータをグレイスケール(0〜255)に変換する
#1.CTデータの値を0〜1に変換する
# 1−1.全ての値について、最小値を引く
# 1-2.全ての値について、最大値で割る
#2.1を０〜２５５に変換する
#param: volume - 変換対象のデータ(HU)
#return: volume_color - グレイスケール変換後のデータ(0〜255)
def hu_to_gray(volume):
    maxhu = np.max(volume)
    minhu = np.min(volume)

    #ブロードキャストが働いてそれぞれの要素単位で計算してくれる
    #最大値と最小値が同じだったら0になってしまうので、それは回避する
    volume_rerange = (volume - minhu) / max((maxhu - minhu), 1e-3)
    volume_rerange = volume_rerange * 255

    #CTとアノテーションマスクでshapeを合わせる
    volume_rerange = np.stack([volume_rerange, volume_rerange, volume_rerange], axis=-1)
    
    return volume_rerange.astype(np.uint8)


#overlay
#1つのチャネルしか計算されないので、マスクの色が薄くなる
#そのチャネルにデータがある場合は、その他のチャネルの同じ箇所のデータも計算されるようにする
#param1: gray_volume - CTデータ(グレイスケール)
#param2: mask_volume - アノテーションデータ
#param3: mask_color - RGBデータ(HU)
#param4: alpha - 透明度
#return: overlayed - オーバーレイ後のデータ(0〜255)
def overlay(gray_volume, mask_volume, mask_color, alpha=0.3):
    #元のデータをTru/Falseに変換する
    mask_filter = np.greater(mask_volume, 0)
    #RGBのチャネル分stackする
    mask_filter = np.stack([mask_filter, mask_filter, mask_filter], axis=-1)
    
    #オーバーレイ
    overlayed = np.where(mask_filter > 0,
                         ((1-alpha)*gray_volume + alpha*mask_color).astype(np.uint8),
                         gray_volume)
    
    return overlayed


#vis_overlay()
#対象データを一覧表示する
#param1: overlayed - 対象データ
#param2: original_volume - 元のHUデータ
#param3: mask_volume - マスクデータ
#param4: cols - 列数
#param5: display_num - 描画数
#param6: figsize - 1プロットのサイズ
#return: none
def vis_overlay(overlayed,
                original_volume,
                mask_volume,
                cols=5,
                display_num=25,
                figsize=(15,15)):
    rows = (display_num - 1) // cols + 1 #行数計算
    total_num = overlayed.shape[-2]      #(630, 630, 45, 3)の"45"
    interval = total_num / display_num   #何枚飛ばしで表示するか
    if interval < 1:
        interval = 1 #display_numがtotaol_numより大きい場合の処置(１枚ずつ表示)

    #プロットの描画
    fig, ax = plt.subplots(rows, cols, figsize=figsize)

    #画像の描画
    for i in range(display_num):
        row_i = i // cols #何番目の行か
        cols_i = i % cols #何番目の列か
        idx = int(i * interval) #スライスのインデックス
        #全部表示し終わったら終了する
        if idx >= total_num:
            break
        
        #HUの統計量を取得
        stats = get_hu_stats(original_volume[:, :, idx], mask_volume[:, :, idx])
        title = 'slice #:{}'.format(idx)
        title += '\nggo mean: {:.0f}±{:.0f}'.format(stats['ggo_mean'], stats['ggo_std'])
        title += '\nconsolidation mean: {:.0f}±{:.0f}'.format(stats['consolidation_mean'], stats['consolidation_std'])
        title += '\neffusion mean: {:.0f}±{:.0f}'.format(stats['effusion_mean'], stats['effusion_std'])
        #描画
        ax[row_i, cols_i].imshow(overlayed[:, :, idx])
        #タイトル(スライス番号)
        ax[row_i, cols_i].set_title(title)
        #軸を消す
        ax[row_i, cols_i].axis('off')

    #データ表示の重なりの解消
    fig.tight_layout()


#get_hu_stat
#HUの統計量を求める
#param1: volume - HUデータ
#param2: mask_volume - マスクデータ
#parame3: label_dict - ラベルデータ(Dictionary型)
#return: HUの統計量
def get_hu_stats(volume,
                 mask_volume,
                 label_dict={1: 'ggo', 2: 'consolidation', 3: 'effusion'}):
    result = {} #結果格納用ディクショナリ
    for label in label_dict.keys():
        prefix = label_dict[label] #label_dictのvalueが入る
        #アノテーションデータが1のところにTrue、それ以外にFalseを入れる
        #roi: region of interest
        roi_hu = volume[np.equal(mask_volume, label)]
        result[prefix + '_mean'] = np.mean(roi_hu)
        result[prefix + '_std'] = np.std(roi_hu)
        
    return result
