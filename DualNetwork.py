from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os
from State import MAZE_CNT

# パラメータの準備
DN_FILTERS = 128                         # 畳み込み層のカーネル層 ※普通は256
DN_RESIDUAL_NUM = 16                     # 残渣ブロックの数 ※普通は19
DN_INPUT_SHAPE = (3, 3, 2)               # 入力シェイプ
DN_OUTPUT_SIZE = MAZE_CNT**2             # 行動数(配置先(MAZE_CNT*MAZE_CNT))


def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))


def residual_bloack():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x, sc])
        x = Activation('relu')(x)
        return x
    return f


def dual_network():
    # モデル作成済みの場合は無処理
    if os.path.exists('./model/best.h5'):
        return

    # 入力層
    input = Input(shape=DN_INPUT_SHAPE)

    # 畳み込み層
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 残渣ブロックX16
    for i in range(DN_RESIDUAL_NUM):
        x = residual_bloack()(x)

    # プーリング層
    x = GlobalAveragePooling2D()(x)

    # ポリシー出力
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation='softmax', name='pi')(x)

    # バリュー出力
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)

    # モデルの作成
    model = Model(inputs=input, outputs=[p, v])

    # モデルの保存
    os.makedirs('./model/', exist_ok=True)
    model.save('./model/best.h5')  # ベストプレイヤーのモデル

    # モデルの破棄
    K.clear_session()
    del model


if __name__ == '__main__':
    dual_network()
