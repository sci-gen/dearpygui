import logging
from os import read
import dearpygui.dearpygui as dpg
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dpg.create_context()

# リンク情報を管理する辞書
links = {}  # {link_id: (output_input_id, input_input_id)}
node_values = {}  # ノードの値を保存

def update_connected_values():
    """接続されているノード間で値を同期"""
    for link_id, (output_input_id, input_input_id) in links.items():
        try:
            # 出力側の値を取得
            output_value = dpg.get_value(output_input_id)
            # 入力側に値を設定
            dpg.set_value(input_input_id, output_value)
        except:
            pass
    logging.info(f"Updated connected values: {node_values}")

def value_changed_callback(sender, app_data):
    """値が変更された時のコールバック"""
    node_values[sender] = app_data
    update_connected_values()

# callback runs when user attempts to connect attributes
def link_callback(sender, app_data):
    """リンクが作成された時のコールバック"""
    # app_data -> (output_attr_id, input_attr_id)
    output_attr_id, input_attr_id = app_data

    # 各属性の子要素（input_float）を取得
    output_children = dpg.get_item_children(output_attr_id, slot=1)
    input_children = dpg.get_item_children(input_attr_id, slot=1)

    if output_children and input_children:
        output_input_id = output_children[0]  # 最初の子要素（input_float）
        input_input_id = input_children[0]    # 最初の子要素（input_float）

        # リンクを作成
        link_id = dpg.add_node_link(output_attr_id, input_attr_id, parent=sender)

        # リンク情報を保存
        links[link_id] = (output_input_id, input_input_id)

        # 初期値を同期
        try:
            output_value = dpg.get_value(output_input_id)
            dpg.set_value(input_input_id, output_value)
        except:
            pass
    logging.info(f"Link created: {output_attr_id} -> {input_attr_id}")
    logging.info(f"Current links: {links}")

# callback runs when user attempts to disconnect attributes
def delink_callback(sender, app_data):
    """リンクが削除された時のコールバック"""
    # app_data -> link_id
    if app_data in links:
        del links[app_data]
    dpg.delete_item(app_data)

with dpg.window(label="Tutorial", width=600, height=400):
    with dpg.node_editor(callback=link_callback, delink_callback=delink_callback):
        with dpg.node(label="Node 1"):
            with dpg.node_attribute(label="Node A1"):
                dpg.add_input_float(label="F1", width=150, readonly=True, callback=value_changed_callback)

            with dpg.node_attribute(label="Node A2", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="F2", width=150, readonly=False, callback=value_changed_callback)

        with dpg.node(label="Node 2"):
            with dpg.node_attribute(label="Node A3"):
                dpg.add_input_float(label="F3", width=200, readonly=True, callback=value_changed_callback)

            with dpg.node_attribute(label="Node A4", attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_float(label="F4", width=200, readonly=False, callback=value_changed_callback)

dpg.create_viewport(title='Custom Title', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()