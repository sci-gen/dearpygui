# -*- coding: utf-8 -*-
"""
シンプル画像ノードエディタ

画像入力ノードと画像表示ノードのみを持つシンプルなノードエディタです。
"""
import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import os
import logging
from typing import Optional, Dict, Any, Set, Tuple

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('node_editor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class NodeEditor:
    """ノードエディタを管理するクラス"""

    def __init__(self) -> None:
        """ノードエディタを初期化する"""
        self.nodes: Dict[int, Dict[str, Any]] = {}
        self.connections: Set[Tuple[int, int]] = set()  # (input_node_id, display_node_id)
        self.next_node_id: int = 1
        self.node_editor_id: Optional[int] = None

    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        画像ファイルを読み込む

        Args:
            file_path: 画像ファイルのパス

        Returns:
            読み込まれた画像配列、失敗時はNone
        """
        try:
            if os.path.exists(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
        return None

    def create_texture(self, image: np.ndarray) -> int:
        """画像からテクスチャを作成する"""
        # 表示サイズに調整
        height, width = image.shape[:2]
        if max(height, width) > 200:
            scale = 200 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        # RGBA形式に変換
        rgba = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])

        # テクスチャ作成
        rgba_float = rgba.astype(np.float32) / 255.0
        with dpg.texture_registry():
            return dpg.add_raw_texture(
                width=rgba.shape[1],
                height=rgba.shape[0],
                default_value=rgba_float.flatten(),
                format=dpg.mvFormat_Float_rgba
            )

    def create_input_node(self, pos: tuple) -> int:
        """画像入力ノードを作成する"""
        node_id = self.next_node_id
        self.next_node_id += 1

        with dpg.node(label=f"Image Input {node_id}", pos=pos, parent=self.node_editor_id):
            # 入力属性内にUIを配置
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                file_input = dpg.add_input_text(
                    label="File Path",
                    width=200,
                    callback=lambda sender, value: self.on_file_path_changed(node_id, value)
                )
                dpg.add_button(label="Browse", callback=lambda: self.open_file_dialog(node_id))

            # 出力属性
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_attr:
                dpg.add_text("Image")

            self.nodes[node_id] = {
                'type': 'input',
                'file_input': file_input,
                'output_attr': output_attr,
                'image_data': None
            }
        return node_id

    def create_display_node(self, pos: tuple) -> int:
        """画像表示ノードを作成"""
        node_id = self.next_node_id
        self.next_node_id += 1

        with dpg.node(label=f"Image Display {node_id}", pos=pos, parent=self.node_editor_id):
            # 入力属性
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_attr:
                dpg.add_text("Image")

            # 表示属性内に画像ウィジェットを配置
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as display_attr:
                image_widget = dpg.add_text("画像なし")

            self.nodes[node_id] = {
                'type': 'display',
                'input_attr': input_attr,
                'display_attr': display_attr,
                'image_widget': image_widget,
                'texture_id': None
            }
        return node_id

    def open_file_dialog(self, node_id: int) -> None:
        """ファイル選択ダイアログを開く"""
        def file_selected(sender, app_data) -> None:
            file_path = app_data['file_path_name']
            dpg.set_value(self.nodes[node_id]['file_input'], file_path)
            self.load_image_to_node(node_id, file_path)
            # ファイル選択後も接続された表示ノードを更新
            self.update_connected_display_nodes(node_id)

        with dpg.file_dialog(show=True, callback=file_selected, width=600, height=400):
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".bmp")

    def on_file_path_changed(self, node_id: int, file_path: str) -> None:
        """
        ファイルパスが変更されたときの処理

        Args:
            node_id: ノードのID
            file_path: 新しいファイルパス
        """
        # 画像を読み込み
        self.load_image_to_node(node_id, file_path)
        # 接続されている表示ノードを更新
        self.update_connected_display_nodes(node_id)

    def load_image_to_node(self, node_id: int, file_path: str) -> None:
        """
        ノードに画像を読み込む

        Args:
            node_id: ノードのID
            file_path: 画像ファイルのパス
        """
        logger.info(f"Loading image to node {node_id}: {file_path}")
        image = self.load_image(file_path)
        if image is not None:
            self.nodes[node_id]['image_data'] = image
            logger.info(f"Image loaded successfully: {file_path}")
            logger.info(f"Image shape: {image.shape}")
        else:
            self.nodes[node_id]['image_data'] = None
            if file_path.strip():  # 空文字列でない場合のみエラー表示
                logger.warning(f"Failed to load image: {file_path}")

        # ノードのimage_dataの状態をログ出力
        logger.info(f"Node {node_id} image_data status: {self.nodes[node_id]['image_data'] is not None}")

    def update_connected_display_nodes(self, input_node_id: int) -> None:
        """
        指定された入力ノードに接続されている全ての表示ノードを更新する

        Args:
            input_node_id: 入力ノードのID
        """
        logger.info(f"Updating connected display nodes for input node {input_node_id}")
        logger.info(f"Current connections: {self.connections}")

        # 入力ノードの画像データ状態を確認
        input_node = self.nodes.get(input_node_id)
        if input_node:
            logger.info(f"Input node {input_node_id} has image data: {input_node.get('image_data') is not None}")
        else:
            logger.error(f"Input node {input_node_id} not found!")

        for connection in self.connections:
            if connection[0] == input_node_id:  # (input_node_id, display_node_id)
                display_node_id = connection[1]
                logger.info(f"Updating display node {display_node_id}")
                self.update_display_node(display_node_id, input_node_id)

    def update_display_node(self, display_node_id: int, input_node_id: int) -> None:
        """
        表示ノードを更新する

        Args:
            display_node_id: 表示ノードのID
            input_node_id: 入力ノードのID
        """
        display_node = self.nodes[display_node_id]
        input_data = self.nodes[input_node_id]['image_data']

        logger.info(f"Updating display node {display_node_id} with data from input node {input_node_id}")
        logger.info(f"Input data exists: {input_data is not None}")

        if input_data is not None:
            # 古いテクスチャを削除（ただし、ウィジェットの削除は慎重に行う）
            if display_node['texture_id']:
                try:
                    dpg.delete_item(display_node['texture_id'])
                    logger.info("Old texture deleted")
                except Exception as e:
                    logger.warning(f"Failed to delete old texture: {e}")

            # 新しいテクスチャを作成
            try:
                texture_id = self.create_texture(input_data)
                display_node['texture_id'] = texture_id
                logger.info(f"New texture created: {texture_id}")

                # 画像ウィジェットを更新（エッジを保持するため、より慎重に更新）
                old_widget = display_node['image_widget']
                parent_attr = display_node['display_attr']

                # 新しい画像ウィジェットを作成
                new_widget = dpg.add_image(
                    texture_id,
                    parent=parent_attr
                )

                # 古いウィジェットを削除（新しいウィジェット作成後に削除）
                try:
                    dpg.delete_item(old_widget)
                    logger.info("Old image widget deleted")
                except Exception as e:
                    logger.warning(f"Failed to delete old image widget: {e}")

                # ウィジェット参照を更新
                display_node['image_widget'] = new_widget
                logger.info("Image displayed successfully")

            except Exception as e:
                logger.error(f"Failed to create/display texture: {e}")
        else:
            # データがない場合は表示をクリア
            logger.info("No input data, clearing display node")
            self.clear_display_node(display_node_id)

    def clear_display_node(self, display_node_id: int) -> None:
        """
        表示ノードをクリアする

        Args:
            display_node_id: 表示ノードのID
        """
        display_node = self.nodes[display_node_id]

        # テクスチャを削除
        if display_node['texture_id']:
            try:
                dpg.delete_item(display_node['texture_id'])
                logger.info("Texture cleared")
            except Exception as e:
                logger.warning(f"Failed to delete texture: {e}")
            display_node['texture_id'] = None

        # 画像ウィジェットを初期状態に戻す（エッジを保持するため慎重に更新）
        old_widget = display_node['image_widget']
        parent_attr = display_node['display_attr']

        # 新しいテキストウィジェットを作成
        new_widget = dpg.add_text(
            "画像なし",
            parent=parent_attr
        )

        # 古いウィジェットを削除
        try:
            dpg.delete_item(old_widget)
            logger.info("Image widget cleared")
        except Exception as e:
            logger.warning(f"Failed to delete image widget: {e}")

        # ウィジェット参照を更新
        display_node['image_widget'] = new_widget

    def handle_connection(self, sender: int, connection: tuple) -> None:
        """
        ノード接続処理

        Args:
            sender: 送信者のID
            connection: 接続情報 (output_attr, input_attr)
        """
        output_attr, input_attr = connection

        # ノードを特定
        input_node_id = None
        display_node_id = None

        for node_id, node_data in self.nodes.items():
            if node_data.get('output_attr') == output_attr:
                input_node_id = node_id
            if node_data.get('input_attr') == input_attr:
                display_node_id = node_id

        if input_node_id and display_node_id:
            # 接続情報を記録
            connection_key = (input_node_id, display_node_id)
            self.connections.add(connection_key)

            logger.info(f"Nodes connected: {input_node_id} -> {display_node_id} (link: {sender})")
            logger.info(f"Updated connections: {self.connections}")

            # 接続時に既存の画像データがあるかチェック
            input_node = self.nodes.get(input_node_id)
            if input_node and input_node.get('image_data') is not None:
                logger.info(f"Input node {input_node_id} already has image data, updating display immediately")
                # 少し遅延を設けてエッジの描画を確実にする
                dpg.set_frame_callback(1, callback=lambda: self.update_display_node(display_node_id, input_node_id))
            else:
                logger.info(f"Input node {input_node_id} has no image data yet")

    def handle_disconnection(self, sender: int, connection: tuple) -> None:
        """
        ノード切断処理

        Args:
            sender: 送信者のID
            connection: 切断情報 (output_attr, input_attr)
        """
        output_attr, input_attr = connection

        # ノードを特定
        input_node_id = None
        display_node_id = None

        for node_id, node_data in self.nodes.items():
            if node_data.get('output_attr') == output_attr:
                input_node_id = node_id
            if node_data.get('input_attr') == input_attr:
                display_node_id = node_id

        if input_node_id and display_node_id:
            # 接続情報を削除
            connection_key = (input_node_id, display_node_id)
            self.connections.discard(connection_key)

            # 表示ノードを初期状態に戻す
            self.clear_display_node(display_node_id)
            logger.info(f"Nodes disconnected: {input_node_id} -> {display_node_id}")
            logger.info(f"Updated connections: {self.connections}")

    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        画像ファイルを読み込む

        Args:
            file_path: 画像ファイルのパス

        Returns:
            読み込まれた画像配列、失敗時はNone
        """
        try:
            if os.path.exists(file_path):
                image = cv2.imread(file_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
        return None

    def create_texture(self, image: np.ndarray) -> int:
        """画像からテクスチャを作成する"""
        # 表示サイズに調整
        height, width = image.shape[:2]
        if max(height, width) > 200:
            scale = 200 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        # RGBA形式に変換
        rgba = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])

        # テクスチャ作成
        rgba_float = rgba.astype(np.float32) / 255.0
        with dpg.texture_registry():
            return dpg.add_raw_texture(
                width=rgba.shape[1],
                height=rgba.shape[0],
                default_value=rgba_float.flatten(),
                format=dpg.mvFormat_Float_rgba
            )

    def create_input_node(self, pos: tuple) -> int:
        """画像入力ノードを作成する"""
        node_id = self.next_node_id
        self.next_node_id += 1

        with dpg.node(label=f"Image Input {node_id}", pos=pos, parent=self.node_editor_id):
            # 入力属性内にUIを配置
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                file_input = dpg.add_input_text(
                    label="File Path",
                    width=200,
                    callback=lambda sender, value: self.on_file_path_changed(node_id, value)
                )
                dpg.add_button(label="Browse", callback=lambda: self.open_file_dialog(node_id))

            # 出力属性
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_attr:
                dpg.add_text("Image")

            self.nodes[node_id] = {
                'type': 'input',
                'file_input': file_input,
                'output_attr': output_attr,
                'image_data': None
            }
        return node_id

    def create_display_node(self, pos: tuple) -> int:
        """画像表示ノードを作成"""
        node_id = self.next_node_id
        self.next_node_id += 1

        with dpg.node(label=f"Image Display {node_id}", pos=pos, parent=self.node_editor_id):
            # 入力属性
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_attr:
                dpg.add_text("Image")

            # 表示属性内に画像ウィジェットを配置
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as display_attr:
                image_widget = dpg.add_text("画像なし")

            self.nodes[node_id] = {
                'type': 'display',
                'input_attr': input_attr,
                'display_attr': display_attr,
                'image_widget': image_widget,
                'texture_id': None
            }
        return node_id

    def open_file_dialog(self, node_id: int) -> None:
        """ファイル選択ダイアログを開く"""
        def file_selected(sender, app_data) -> None:
            file_path = app_data['file_path_name']
            dpg.set_value(self.nodes[node_id]['file_input'], file_path)
            self.load_image_to_node(node_id, file_path)
            # ファイル選択後も接続された表示ノードを更新
            self.update_connected_display_nodes(node_id)

        with dpg.file_dialog(show=True, callback=file_selected, width=600, height=400):
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".bmp")

    def on_file_path_changed(self, node_id: int, file_path: str) -> None:
        """
        ファイルパスが変更されたときの処理

        Args:
            node_id: ノードのID
            file_path: 新しいファイルパス
        """
        # 画像を読み込み
        self.load_image_to_node(node_id, file_path)
        # 接続されている表示ノードを更新
        self.update_connected_display_nodes(node_id)

    def load_image_to_node(self, node_id: int, file_path: str) -> None:
        """
        ノードに画像を読み込む

        Args:
            node_id: ノードのID
            file_path: 画像ファイルのパス
        """
        logger.info(f"Loading image to node {node_id}: {file_path}")
        image = self.load_image(file_path)
        if image is not None:
            self.nodes[node_id]['image_data'] = image
            logger.info(f"Image loaded successfully: {file_path}")
            logger.info(f"Image shape: {image.shape}")
        else:
            self.nodes[node_id]['image_data'] = None
            if file_path.strip():  # 空文字列でない場合のみエラー表示
                logger.warning(f"Failed to load image: {file_path}")

        # ノードのimage_dataの状態をログ出力
        logger.info(f"Node {node_id} image_data status: {self.nodes[node_id]['image_data'] is not None}")

    def update_connected_display_nodes(self, input_node_id: int) -> None:
        """
        指定された入力ノードに接続されている全ての表示ノードを更新する

        Args:
            input_node_id: 入力ノードのID
        """
        logger.info(f"Updating connected display nodes for input node {input_node_id}")
        logger.info(f"Current connections: {self.connections}")

        # 入力ノードの画像データ状態を確認
        input_node = self.nodes.get(input_node_id)
        if input_node:
            logger.info(f"Input node {input_node_id} has image data: {input_node.get('image_data') is not None}")
        else:
            logger.error(f"Input node {input_node_id} not found!")

        for connection in self.connections:
            if connection[0] == input_node_id:  # (input_node_id, display_node_id)
                display_node_id = connection[1]
                logger.info(f"Updating display node {display_node_id}")
                self.update_display_node(display_node_id, input_node_id)

    def update_display_node(self, display_node_id: int, input_node_id: int) -> None:
        """
        表示ノードを更新する

        Args:
            display_node_id: 表示ノードのID
            input_node_id: 入力ノードのID
        """
        display_node = self.nodes[display_node_id]
        input_data = self.nodes[input_node_id]['image_data']

        logger.info(f"Updating display node {display_node_id} with data from input node {input_node_id}")
        logger.info(f"Input data exists: {input_data is not None}")

        if input_data is not None:
            # 古いテクスチャを削除（ただし、ウィジェットの削除は慎重に行う）
            if display_node['texture_id']:
                try:
                    dpg.delete_item(display_node['texture_id'])
                    logger.info("Old texture deleted")
                except Exception as e:
                    logger.warning(f"Failed to delete old texture: {e}")

            # 新しいテクスチャを作成
            try:
                texture_id = self.create_texture(input_data)
                display_node['texture_id'] = texture_id
                logger.info(f"New texture created: {texture_id}")

                # 画像ウィジェットを更新（エッジを保持するため、より慎重に更新）
                old_widget = display_node['image_widget']
                parent_attr = display_node['display_attr']

                # 新しい画像ウィジェットを作成
                new_widget = dpg.add_image(
                    texture_id,
                    parent=parent_attr
                )

                # 古いウィジェットを削除（新しいウィジェット作成後に削除）
                try:
                    dpg.delete_item(old_widget)
                    logger.info("Old image widget deleted")
                except Exception as e:
                    logger.warning(f"Failed to delete old image widget: {e}")

                # ウィジェット参照を更新
                display_node['image_widget'] = new_widget
                logger.info("Image displayed successfully")

            except Exception as e:
                logger.error(f"Failed to create/display texture: {e}")
        else:
            # データがない場合は表示をクリア
            logger.info("No input data, clearing display node")
            self.clear_display_node(display_node_id)

    def clear_display_node(self, display_node_id: int) -> None:
        """
        表示ノードをクリアする

        Args:
            display_node_id: 表示ノードのID
        """
        display_node = self.nodes[display_node_id]

        # テクスチャを削除
        if display_node['texture_id']:
            try:
                dpg.delete_item(display_node['texture_id'])
                logger.info("Texture cleared")
            except Exception as e:
                logger.warning(f"Failed to delete texture: {e}")
            display_node['texture_id'] = None

        # 画像ウィジェットを初期状態に戻す（エッジを保持するため慎重に更新）
        old_widget = display_node['image_widget']
        parent_attr = display_node['display_attr']

        # 新しいテキストウィジェットを作成
        new_widget = dpg.add_text(
            "画像なし",
            parent=parent_attr
        )

        # 古いウィジェットを削除
        try:
            dpg.delete_item(old_widget)
            logger.info("Image widget cleared")
        except Exception as e:
            logger.warning(f"Failed to delete image widget: {e}")

        # ウィジェット参照を更新
        display_node['image_widget'] = new_widget

    def handle_connection(self, sender: int, connection: tuple) -> None:
        """
        ノード接続処理

        Args:
            sender: 送信者のID
            connection: 接続情報 (output_attr, input_attr)
        """
        output_attr, input_attr = connection

        # ノードを特定
        input_node_id = None
        display_node_id = None

        for node_id, node_data in self.nodes.items():
            if node_data.get('output_attr') == output_attr:
                input_node_id = node_id
            if node_data.get('input_attr') == input_attr:
                display_node_id = node_id

        if input_node_id and display_node_id:
            # 接続情報を記録
            connection_key = (input_node_id, display_node_id)
            self.connections.add(connection_key)

            logger.info(f"Nodes connected: {input_node_id} -> {display_node_id} (link: {sender})")
            logger.info(f"Updated connections: {self.connections}")

            # 接続時に既存の画像データがあるかチェック
            input_node = self.nodes.get(input_node_id)
            if input_node and input_node.get('image_data') is not None:
                logger.info(f"Input node {input_node_id} already has image data, updating display immediately")
                # 少し遅延を設けてエッジの描画を確実にする
                dpg.set_frame_callback(1, callback=lambda: self.update_display_node(display_node_id, input_node_id))
            else:
                logger.info(f"Input node {input_node_id} has no image data yet")

    def handle_disconnection(self, sender: int, connection: tuple) -> None:
        """
        ノード切断処理

        Args:
            sender: 送信者のID
            connection: 切断情報 (output_attr, input_attr)
        """
        output_attr, input_attr = connection

        # ノードを特定
        input_node_id = None
        display_node_id = None

        for node_id, node_data in self.nodes.items():
            if node_data.get('output_attr') == output_attr:
                input_node_id = node_id
            if node_data.get('input_attr') == input_attr:
                display_node_id = node_id

        if input_node_id and display_node_id:
            # 接続情報を削除
            connection_key = (input_node_id, display_node_id)
            self.connections.discard(connection_key)

            # 表示ノードを初期状態に戻す
            self.clear_display_node(display_node_id)
            logger.info(f"Nodes disconnected: {input_node_id} -> {display_node_id}")
            logger.info(f"Updated connections: {self.connections}")

def main() -> None:
    """メイン関数"""
    dpg.create_context()
    editor = NodeEditor()

    with dpg.window(label="シンプル画像ノードエディタ", width=1000, height=700) as main_window:
        with dpg.node_editor(
            callback=editor.handle_connection,
            delink_callback=editor.handle_disconnection,
            width=980,
            height=650
        ) as node_editor:
            editor.node_editor_id = node_editor

        # デフォルトノードを作成
        editor.create_input_node((50, 50))
        editor.create_display_node((350, 50))

    dpg.create_viewport(title="画像ノードエディタ", width=1000, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(main_window, True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
