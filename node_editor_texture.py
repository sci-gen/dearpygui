import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from PIL import Image
from typing import List, Union, Tuple, Optional


def normalize_numpy_image(image_array: np.ndarray) -> np.ndarray:
    """NumPy画像配列を0.0-1.0の範囲に正規化する関数

    Args:
        image_array: 正規化対象のNumPy配列

    Returns:
        np.ndarray: 正規化されたfloat32配列 (0.0-1.0)
    """
    if image_array.dtype == np.uint8:
        # uint8の場合: 0-255 → 0.0-1.0
        return image_array.astype(np.float32) / 255.0
    elif image_array.dtype == np.uint16:
        # uint16の場合: 0-65535 → 0.0-1.0
        return image_array.astype(np.float32) / 65535.0
    elif np.issubdtype(image_array.dtype, np.floating):
        # float型の場合: min-max正規化
        img_min = image_array.min()
        img_max = image_array.max()

        if img_max == img_min:
            # 全て同じ値の場合
            return np.zeros_like(image_array, dtype=np.float32)
        else:
            # min-max正規化
            normalized = (image_array - img_min) / (img_max - img_min)
            return normalized.astype(np.float32)
    else:
        # その他の整数型: min-max正規化
        img_min = float(image_array.min())
        img_max = float(image_array.max())

        if img_max == img_min:
            return np.zeros_like(image_array, dtype=np.float32)
        else:
            normalized = (image_array.astype(np.float32) - img_min) / (
                img_max - img_min
            )
            return normalized


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """PIL画像をNumPy配列に変換する関数

    Args:
        pil_image: PIL.Imageオブジェクト

    Returns:
        np.ndarray: NumPy配列形式の画像データ

    Raises:
        ValueError: 入力データが不正な場合
    """
    if not isinstance(pil_image, Image.Image):
        raise ValueError("入力はPIL.Imageオブジェクトである必要があります")

    # PIL画像をNumPy配列に変換
    numpy_array = np.array(pil_image)

    return numpy_array


def swap_rgb_to_bgr(image_array: np.ndarray) -> np.ndarray:
    """RGB配列のRとBチャンネルを入れ替えてBGRにする関数

    Args:
        image_array: RGB形式のNumPy配列 (H, W, 3) または (H, W, 4)

    Returns:
        np.ndarray: BGR形式に変換された配列

    Raises:
        ValueError: 入力データが不正な場合
    """
    if image_array.ndim != 3 or image_array.shape[2] not in [3, 4]:
        raise ValueError(
            f"入力は(H, W, 3)または(H, W, 4)形状である必要があります。実際: {image_array.shape}"
        )

    # 配列をコピーしてRとBチャンネルを入れ替え
    bgr_array = image_array.copy()
    bgr_array[:, :, [0, 2]] = bgr_array[:, :, [2, 0]]  # RとBを交換

    return bgr_array


def numpy_to_dpg_texture(
    image_array: np.ndarray, alpha: Optional[float] = None, is_rgb: bool = False
) -> List[float]:
    """NumPy配列をDearPyGuiテクスチャ形式に変換する統一関数

    Args:
        image_array: NumPy配列形式の画像データ
        alpha: アルファ値の指定 (None の場合は自動設定またはデフォルト1.0)
        is_rgb: TrueならRGB、FalseならBGR（OpenCVデフォルト）として処理

    Returns:
        List[float]: DearPyGui用RGBA形式のテクスチャデータ

    Raises:
        ValueError: サポートされていない画像形状の場合
    """
    if not isinstance(image_array, np.ndarray):
        raise ValueError("入力はNumPy配列である必要があります")

    # 画像の正規化
    normalized_img = normalize_numpy_image(image_array)

    # 次元数による処理分岐
    if normalized_img.ndim == 2:
        # Grayscale画像 (H, W)
        height, width = normalized_img.shape

        # アルファ値の設定
        alpha_value = 1.0 if alpha is None else alpha

        # RGBA配列を作成
        rgba_array = np.zeros((height, width, 4), dtype=np.float32)
        rgba_array[:, :, 0] = normalized_img  # R
        rgba_array[:, :, 1] = normalized_img  # G
        rgba_array[:, :, 2] = normalized_img  # B
        rgba_array[:, :, 3] = alpha_value  # A

    elif normalized_img.ndim == 3:
        height, width, channels = normalized_img.shape

        if channels == 3:
            # RGB/BGR画像 (H, W, 3)
            if is_rgb:
                # 既にRGB → そのまま使用
                rgb_img = normalized_img
                print("  → RGB画像として処理")
            else:
                # BGR → RGB変換（OpenCVデフォルト）
                rgb_img = normalized_img[:, :, [2, 1, 0]]  # BGRをRGBに
                print("  → BGR画像として処理（BGR→RGB変換）")

            # アルファ値の設定
            alpha_value = 1.0 if alpha is None else alpha

            # RGBA配列を作成
            rgba_array = np.zeros((height, width, 4), dtype=np.float32)
            rgba_array[:, :, :3] = rgb_img  # RGB
            rgba_array[:, :, 3] = alpha_value  # A

        elif channels == 4:
            # RGBA/BGRA画像 (H, W, 4)
            if is_rgb:
                # 既にRGBA → そのまま使用
                rgba_array = normalized_img.copy()
                print("  → RGBA画像として処理")
            else:
                # BGRA → RGBA変換
                rgba_array = normalized_img[:, :, [2, 1, 0, 3]]  # BGRAをRGBAに
                print("  → BGRA画像として処理（BGRA→RGBA変換）")

            # アルファ値が指定されている場合は上書き
            if alpha is not None:
                rgba_array[:, :, 3] = alpha

        else:
            raise ValueError(
                f"3次元配列の場合、チャンネル数は3または4である必要があります。実際: {channels}"
            )
    else:
        raise ValueError(
            f"2次元または3次元配列である必要があります。実際: {normalized_img.ndim}次元"
        )

    return rgba_array.flatten().tolist()


# 画像をテクスチャにしたいときはこれを呼び出す
def auto_convert_to_dpg_texture(
    image_data: Union[np.ndarray, Image.Image, List[float]],
    alpha: Optional[float] = None,
    is_rgb: bool = False,
) -> List[float]:
    """任意の画像データを自動判別してDearPyGuiテクスチャ形式に変換する統一関数

    Args:
        image_data: 画像データ（PIL、NumPy、またはテクスチャリスト）
        alpha: アルファ値の指定（任意）
        is_rgb: NumPy配列がRGB形式かどうか（False=BGR、True=RGB）

    Returns:
        List[float]: DearPyGui用RGBA形式のテクスチャデータ

    Raises:
        ValueError: サポートされていない形式の場合
    """
    # データ型による処理分岐
    if isinstance(image_data, Image.Image):
        # PIL → NumPy → テクスチャ変換
        numpy_array = pil_to_numpy(image_data)
        # PILはRGB形式なのでis_rgb=Trueを設定
        return numpy_to_dpg_texture(numpy_array, alpha, is_rgb=True)

    elif isinstance(image_data, np.ndarray):
        # NumPy → テクスチャ変換
        return numpy_to_dpg_texture(image_data, alpha, is_rgb)

    elif isinstance(image_data, list) and all(
        isinstance(x, (int, float)) for x in image_data
    ):
        # 既にテクスチャ形式のリスト
        return image_data

    else:
        raise ValueError(f"サポートされていないデータ型: {type(image_data)}")


# メタデータ 取得 保留
# def get_image_info(image_data: Union[np.ndarray, Image.Image]) -> dict:
#     """画像データの詳細情報を取得する関数

#     Args:
#         image_data: 情報を取得する画像データ

#     Returns:
#         dict: 画像の詳細情報
#     """
#     info = {
#         "type": type(image_data).__name__,
#         "is_numpy": isinstance(image_data, np.ndarray),
#         "is_pil": isinstance(image_data, Image.Image),
#         "dtype": None,
#         "shape": None,
#         "channels": None,
#         "size_mb": 0.0,
#     }

#     if isinstance(image_data, Image.Image):
#         info.update(
#             {
#                 "size": image_data.size,  # (width, height)
#                 "mode": image_data.mode,
#                 "format": image_data.format,
#             }
#         )

#     elif isinstance(image_data, np.ndarray):
#         info.update(
#             {
#                 "dtype": str(image_data.dtype),
#                 "shape": image_data.shape,
#                 "size_mb": image_data.nbytes / (1024 * 1024),
#             }
#         )

#         # チャンネル数の判定
#         if image_data.ndim == 2:
#             info["channels"] = 1
#         elif image_data.ndim == 3:
#             info["channels"] = image_data.shape[2]

#     return info


# ファイルパスから画像を読み込んでテクスチャを生成する 保留
# def create_texture_from_file(file_path: str, texture_tag: str) -> Tuple[int, int]:
#     """ファイルから画像を読み込んでDearPyGuiテクスチャを作成する関数

#     Args:
#         file_path: 画像ファイルのパス
#         texture_tag: 作成するテクスチャのタグ

#     Returns:
#         Tuple[int, int]: 作成されたテクスチャの(width, height)

#     Raises:
#         FileNotFoundError: ファイルが見つからない場合
#         ValueError: 画像の読み込みに失敗した場合
#     """
#     try:
#         # PILで画像を読み込み
#         im = cv2.imread(file_path)

#         # 統一変換関数を使用
#         texture_data = auto_convert_to_dpg_texture(im)

#         height, width = im.shape[:2]

#         # DearPyGuiテクスチャを作成
#         dpg.add_dynamic_texture(
#             width=width, height=height, default_value=texture_data, tag=texture_tag
#         )

#         print(f"テクスチャを作成しました: {texture_tag} ({width}x{height})")
#         return width, height

#     except FileNotFoundError:
#         raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
#     except Exception as e:
#         raise ValueError(f"画像の読み込みに失敗しました: {e}")


# Tagを指定してテクスチャを更新する 保留
# def update_texture_from_any(
#     texture_tag: str,
#     image_data: Union[np.ndarray, Image.Image, List[float]],
#     alpha: Optional[float] = None,
#     is_rgb: bool = False,
# ) -> None:
#     """任意の画像データからテクスチャを更新する便利関数

#     Args:
#         texture_tag: 更新するテクスチャのタグ
#         image_data: 画像データ
#         alpha: アルファ値の指定（任意）
#         is_rgb: NumPy配列がRGB形式かどうか
#     """
#     try:
#         texture_data = auto_convert_to_dpg_texture(image_data, alpha, is_rgb)
#         dpg.set_value(texture_tag, texture_data)

#         info = (
#             get_image_info(image_data)
#             if not isinstance(image_data, list)
#             else {"type": "list"}
#         )
#         color_format = "RGB" if is_rgb else "BGR"
#         print(
#             f"テクスチャ '{texture_tag}' を更新しました: {info['type']} ({color_format})"
#         )

#     except Exception as e:
#         print(f"テクスチャ更新エラー: {e}")


# 使用例とテスト関数
def test_unified_conversion_functions() -> None:
    """統一変換関数のテスト用関数"""

    print("=== 統一変換関数テスト（is_rgbフラグ対応） ===")

    # 1. PIL画像テスト（自動的にRGBとして処理）
    pil_img = Image.new("RGB", (100, 100), color="red")
    texture_data_pil = auto_convert_to_dpg_texture(pil_img)
    print(f"PIL変換: {len(texture_data_pil)}要素 (期待値: {100*100*4})")

    # 2. NumPy grayscale テスト
    grayscale_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    texture_data_gray = auto_convert_to_dpg_texture(grayscale_img)
    print(f"Grayscale変換: {len(texture_data_gray)}要素 (期待値: {100*100*4})")

    # 3. NumPy RGB テスト（is_rgb=True）
    rgb_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    texture_data_rgb = auto_convert_to_dpg_texture(rgb_img, is_rgb=True)
    print(f"RGB変換（is_rgb=True）: {len(texture_data_rgb)}要素")

    # 4. NumPy BGR テスト（is_rgb=False、デフォルト）
    bgr_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    texture_data_bgr = auto_convert_to_dpg_texture(bgr_img, is_rgb=False)
    print(f"BGR変換（is_rgb=False）: {len(texture_data_bgr)}要素")

    # 5. Float型画像テスト（min-max正規化）
    float_img = np.random.random((50, 50, 3)).astype(np.float32) * 10.0  # 0-10の範囲
    texture_data_float = auto_convert_to_dpg_texture(float_img, is_rgb=True)
    print(f"Float RGB変換（min-max正規化）: {len(texture_data_float)}要素")

    # 6. アルファ値指定テスト
    texture_data_alpha = auto_convert_to_dpg_texture(grayscale_img, alpha=0.5)
    print(f"アルファ値指定変換: 半透明設定")

    # 7. RGB/BGR入れ替えテスト
    test_img = np.zeros((10, 10, 3), dtype=np.uint8)
    test_img[:, :, 0] = 255  # 最初のチャンネルを255に

    swapped_img = swap_rgb_to_bgr(test_img)
    print(f"RGB→BGR入れ替えテスト: {test_img[0,0,:]} → {swapped_img[0,0,:]}")


def init_texture(texture_tag: str) -> None:
    """DearPyGuiのテクスチャレジストリを作成する関数"""

    initial_data = [1.0, 0.5, 0.0, 1.0] * (200 * 100)
    dpg.add_dynamic_texture(
        width=200, height=100, default_value=initial_data, tag=texture_tag
    )


def add_instant_image_node(
    image_data: Union[np.ndarray, Image.Image], node_label: str = "画像ノード"
) -> None:
    """即座に画像ノードを作成する関数

    Args:
        image_data: 表示する画像データ
        node_label: ノードのラベル
    """
    import time

    # ユニークなIDを生成
    unique_id = int(time.time() * 1000)
    texture_tag = f"instant_texture_{unique_id}"

    # 画像をテクスチャに変換
    texture_data = auto_convert_to_dpg_texture(image_data)

    # 画像サイズを取得
    if isinstance(image_data, np.ndarray):
        height, width = image_data.shape[:2]
    else:  # PIL Image
        width, height = image_data.size

    # テクスチャを作成
    with dpg.texture_registry():
        dpg.add_dynamic_texture(
            width=width,
            height=height,
            default_value=texture_data,
            tag=texture_tag,
        )

    # ノードを作成して即座に表示
    with dpg.node(label=f"{node_label} ({width}x{height})", parent="node_editor"):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
            dpg.add_image(texture_tag=texture_tag)


def on_create_pattern_image() -> None:
    """パターン画像を作成して表示"""
    # チェッカーボードパターンを作成
    size = 120
    pattern_image = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            if (x // 20 + y // 20) % 2 == 0:
                pattern_image[y, x] = [255, 255, 255]  # 白
            else:
                pattern_image[y, x] = [0, 0, 0]  # 黒

    # 即座にノードとして追加
    add_instant_image_node(pattern_image, "chekerboard")


def on_create_circle_image() -> None:
    """円画像を作成して表示するコールバック関数"""
    size = 150
    circle_image = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius = size // 3

    y, x = np.ogrid[:size, :size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
    circle_image[mask] = [100, 255, 100]  # 緑の円

    add_instant_image_node(circle_image, "手動追加：円画像")


####
def main() -> None:
    """メインアプリケーション関数"""
    dpg.create_context()
    texture_tag = "example_texture"
    with dpg.texture_registry():
        # テクスチャレジストリを作成
        init_texture(texture_tag=texture_tag)
    with dpg.window(
        tag="main_window", label="Node Editor Texture Example", width=800, height=600
    ):
        # コントロールパネルを追加
        with dpg.group(horizontal=True):
            dpg.add_button(label="add checkerboard", callback=on_create_pattern_image)
            dpg.add_button(label="add circle", callback=on_create_circle_image)
            # dpg.add_button(label="add startup nodes", callback=create_startup_nodes)
            dpg.add_button(
                label="add test functions",
                callback=lambda: test_unified_conversion_functions(),
            )

        dpg.add_separator()

        with dpg.node_editor(tag="node_editor"):
            with dpg.node(label="Image Node", tag="image_node"):
                # 画像ノードの属性
                with dpg.node_attribute(
                    tag="image_node_attr", attribute_type=dpg.mvNode_Attr_Static
                ):
                    # 画像を表示するためのイメージウィジェット
                    dpg.add_image(texture_tag=texture_tag)

            # 統一変換関数のテスト
            # test_unified_conversion_functions()

    dpg.set_primary_window("main_window", True)
    dpg.create_viewport(title="Node Editor Texture Example", width=800, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

    # on_create_pattern_image()


if __name__ == "__main__":
    main()
