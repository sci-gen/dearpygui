import dearpygui.dearpygui as dpg
import numpy as np


def create_sample_image(width: int = 1, height: int = 1) -> list:
    """Create a sample image (RGBA format)"""

    # RGBA形式で直接作成
    image_data = []
    for i in range(height):
        for j in range(width):
            # 正規化された値（0.0-1.0）
            r = i / height
            g = j / width
            b = 0.5
            a = 1.0  # アルファチャンネル

            image_data.extend([r, g, b, a])

    return image_data


def update_image(texture_tag: str) -> None:
    """テクスチャの情報を取得して画像を更新する関数"""

    try:
        # テクスチャの設定情報を取得
        texture_config = dpg.get_item_configuration(texture_tag)
        width = texture_config.get("width", 100)
        height = texture_config.get("height", 100)

        print(f"texture size: {width}x{height}")

    except Exception as e:
        # テクスチャが見つからない場合はデフォルト値を使用
        print(f"Failed to retrieve texture information: {e}")
        width, height = 200, 200

    # 取得したサイズで新しいランダム画像を生成
    image_data = []

    # ランダムな色で画像を更新
    for i in range(height):
        for j in range(width):
            r = (i / height) * np.random.uniform(0.5, 1.0)
            g = (j / width) * np.random.uniform(0.5, 1.0)
            b = np.random.uniform(0.3, 0.8)
            a = 1.0

            image_data.extend([r, g, b, a])

    # テクスチャデータを更新
    dpg.set_value(texture_tag, image_data)


def main():
    """Main function"""
    texture_tag = "sample_texture"
    w, h = 200, 200  # 画像の幅と高さ

    # DearPyGuiコンテキストを作成
    dpg.create_context()

    # 最初に画像データを作成
    image_data = create_sample_image(width=w, height=h)

    # テクスチャレジストリでテクスチャを作成
    with dpg.texture_registry():
        dpg.add_dynamic_texture(
            width=w, height=h, default_value=image_data, tag=texture_tag
        )

    # メインウィンドウ
    with dpg.window(
        label="Image Display - NodeEditor",
        width=800,
        height=600,
        tag="main_window",
    ):
        # ノードエディターの設定
        with dpg.node_editor(tag="node_editor"):
            # 画像表示ノード
            with dpg.node(label="Image Display", tag="image_node", pos=[50, 50]):
                # Node attribute for image display
                with dpg.node_attribute(
                    label="Image",
                    tag="image_attr",
                    attribute_type=dpg.mvNode_Attr_Static,
                ):
                    # Place image inside node_attribute
                    dpg.add_image(texture_tag=texture_tag, width=150, height=150)

            # 情報表示ノード
            with dpg.node(label="Image Info", tag="info_node", pos=[300, 50]):
                with dpg.node_attribute(
                    label="Details",
                    tag="info_details",
                    attribute_type=dpg.mvNode_Attr_Static,
                ):
                    dpg.add_text(f"Size: {w}x{h}")
                    dpg.add_text("Format: RGBA")
                    dpg.add_text(f"Data Type: {type(image_data[0]).__name__}")

            # コントロールノード
            with dpg.node(label="Control", tag="control_node", pos=[50, 280]):

                with dpg.node_attribute(
                    label="Operations",
                    tag="control_attr",
                    attribute_type=dpg.mvNode_Attr_Static,
                ):
                    dpg.add_button(
                        label="Update Image", callback=lambda: update_image(texture_tag)
                    )

    # ビューポートの設定
    dpg.create_viewport(title="DearPyGui NodeEditor Image App", width=900, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    dpg.set_primary_window("main_window", True)  # メインウィンドウをプライマリに設定
    dpg.start_dearpygui()  # DearPyGuiのメインループを開始
    dpg.destroy_context()  # クリーンアップ


if __name__ == "__main__":
    main()
