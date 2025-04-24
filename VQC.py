import ctypes
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import font as tkfont
import shutil
import subprocess
from datetime import datetime
import re
from time import sleep
import os
import webbrowser
from threading import Thread
import configparser
import platform

import cv2
from PIL import Image, ImageTk
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mplcursors


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def hyperlink_jump(hyperlink: str):
    webbrowser.open(hyperlink)


PROGRAM_NAME = "Video Quality Compare"
VERSION = "1.1.0"
HOME_LINK = "https://github.com/op200/Video_Quality_Compare"


# 日志
class log:
    @staticmethod
    def _output(info: str):
        info = f"{datetime.now().strftime('%Y.%m.%d %H:%M:%S')} {info}"
        log_Text.insert(tk.END, info + "\n")
        log_Text.see(tk.END)
        print(info)

    @staticmethod
    def error(info: str):
        log._output(f"[VQC ERROR] {info}")

    @staticmethod
    def warning(info: str):
        log._output(f"[VQC WARNING] {info}")

    @staticmethod
    def info(info: str):
        log._output(f"[VQC INFO] {info}")


if os.name == "nt":
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        log.warning("Windows DPI Aware failed")


# 判断系统对应路径
os_type = platform.system()
if os_type == "Windows":
    config_dir = os.path.join(os.getenv("APPDATA") or "", "VQC")
elif os_type == "Linux" or os_type == "Darwin":
    config_dir = os.path.join(os.path.expanduser("~"), ".config", "VQC")
else:
    config_dir = ""
    log.warning("无法确认系统")

os.makedirs(config_dir, exist_ok=True)


# 不存在配置则写入默认配置
config = configparser.ConfigParser()
config_file_pathname = os.path.join(config_dir, "config.ini")
if (
    not os.path.exists(config_file_pathname)
    or config.read(config_file_pathname)
    and config.get("DEFAULT", "version") != VERSION
):
    config["DEFAULT"] = {
        "version": VERSION,
        "cmp_type": "ssim",
        "cmp_hwaccel_1": "none",
        "cmp_hwaccel_2": "none",
    }
    with open(config_file_pathname, "w") as config_file:
        config.write(config_file)


def save_config():
    config["DEFAULT"]["cmp_type"] = cmp_select_Combobox.get()
    config["DEFAULT"]["cmp_hwaccel_1"] = cmp_hwaccel_1_Combobox.get()
    config["DEFAULT"]["cmp_hwaccel_2"] = cmp_hwaccel_2_Combobox.get()

    try:
        with open(config_file_pathname, "w") as configfile:
            config.write(configfile)
    except Exception as e:
        log.error("保存配置文件失败: " + repr(e))


input_video_path_1: str
input_video_path_2: str
input_video_path_now: str
scale, cut_scale = False, False
frame_height: int
frame_width: int
new_frame_height: int
new_frame_width: int
fps: float

is_listener_data = True
VIDEO_FRAME_IMG_HEIGHT = 6


root_Tk = tk.Tk()
root_Tk.title(PROGRAM_NAME)
root_Tk.resizable(False, False)


def root_Tk_Close():
    global is_listener_data
    is_listener_data = False
    save_config()
    root_Tk.destroy()


root_Tk.protocol("WM_DELETE_WINDOW", root_Tk_Close)


TkDefaultFont = tkfont.nametofont("TkDefaultFont").actual()["family"]
underline_font = tkfont.Font(
    family=TkDefaultFont,
    size=tkfont.nametofont("TkDefaultFont").actual()["size"],
    underline=True,
)


# 菜单
menu_Menu = tk.Menu(root_Tk)

menu_setting_Menu = tk.Menu(menu_Menu, tearoff=0)


def open_temp():
    os.startfile(config_dir)


menu_Menu.add_cascade(label="设置", menu=menu_setting_Menu)
menu_setting_Menu.add_command(label="打开缓存位置", command=open_temp)


def remove_config_dir():
    if os.path.exists(config_dir):
        shutil.rmtree(config_dir)
        log.info("已删除" + config_dir)
    else:
        log.error("未找到配置文件目录" + config_dir)


menu_setting_Menu.add_command(label="清除配置文件", command=remove_config_dir)


def create_help_about_Toplevel():
    about_Toplevel = tk.Toplevel(root_Tk, width=20, height=15)
    about_Toplevel.geometry("320x160")
    about_Toplevel.title("About")
    frame = ttk.Frame(about_Toplevel)
    frame.pack(expand=True)
    ttk.Label(frame, text=f"{PROGRAM_NAME}  v{VERSION}\n\n").pack()

    hyperlink_Label = ttk.Label(
        frame, text=HOME_LINK, cursor="hand2", foreground="blue", font=underline_font
    )
    hyperlink_Label.bind("<Button-1>", lambda _: hyperlink_jump(HOME_LINK))
    hyperlink_Label.pack()


menu_help_Menu = tk.Menu(menu_Menu, tearoff=0)
menu_help_Menu.add_command(label="关于", command=create_help_about_Toplevel)
menu_Menu.add_cascade(label="帮助", menu=menu_help_Menu)


root_Tk.config(menu=menu_Menu)

# 左侧控件
left_Frame = ttk.Frame(root_Tk, cursor="tcross")
left_Frame.grid(row=0, column=0, padx=5, pady=5)
# 右侧控件
right_Frame = ttk.Frame(root_Tk)
right_Frame.grid(row=0, column=1, padx=5, pady=5)


# 左侧控件

# 视频预览控件
video_review_Label = ttk.Label(left_Frame)

# 进度条控件
video_Progressbar = ttk.Progressbar(left_Frame)


def draw_video_frame_Label_frameColor(frame_num: int, color: tuple[int, int, int]):
    global video_frame_img
    x = round(new_frame_width * frame_num / (frame_count - 1)) - 1
    if x < 0:
        x = 0
    video_frame_img[: VIDEO_FRAME_IMG_HEIGHT - 1, x] = color


def flush_video_frame_Label():
    photo = ImageTk.PhotoImage(Image.fromarray(video_frame_img))
    video_frame_Label.config(image=photo)
    video_frame_Label.image = photo  # type: ignore


def draw_video_frame_Label_range(
    start_frame: int, end_frame: int, color: tuple[int, int, int]
):
    global video_frame_img
    video_frame_img[-1, :, :] = 0
    video_frame_img[
        -1,
        max(round(new_frame_width * start_frame / (frame_count - 1)) - 1, 0) : max(
            round(new_frame_width * end_frame / (frame_count - 1)) - 1, 0
        )
        + 1,
    ] = color


video_frame_Label = ttk.Label(left_Frame)

frame_count = 0
frame_now = 0

cut_scale_x = 0
cut_scale_y = 0


# 跳转当前帧
def jump_to_frame():
    global scale, frame_now, frame_count
    main_rendering_Cap.set(cv2.CAP_PROP_POS_FRAMES, frame_now)
    frame = main_rendering_Cap.read()[1]
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except cv2.error:
        log.warning(f"[{frame_now}]该帧无法读取(应检查视频封装)")
    else:
        if scale:
            if cut_scale:
                frame = frame[
                    cut_scale_y : cut_scale_y + new_frame_height,
                    cut_scale_x : cut_scale_x + new_frame_width,
                ]
            else:
                frame = cv2.resize(frame, (new_frame_width, new_frame_height))

        video_Progressbar["value"] = frame_now / (frame_count - 1) * 100

        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        video_review_Label.config(image=photo)
        video_review_Label.image = photo  # type: ignore

        # set frame_now
        frame_now_Tkint.set(frame_now)


# 进度条的滚轮事件
def video_progressbar_mousewheel(event):
    global frame_now, frame_count
    frame_now += 1 if event.delta < 0 else -1
    if frame_now < 0:
        frame_now = 0
    if frame_now >= frame_count:
        frame_now = frame_count - 1

    jump_to_frame()


video_review_Label.bind("<MouseWheel>", video_progressbar_mousewheel)
video_Progressbar.bind("<MouseWheel>", video_progressbar_mousewheel)
video_frame_Label.bind("<MouseWheel>", video_progressbar_mousewheel)


# 进度条鼠标点击事件
def video_progressbar_leftDrag(event):
    ratio = event.x / video_Progressbar.winfo_width()
    if ratio > 1:
        ratio = 1
    if ratio < 0:
        ratio = 0
    # video_Progressbar["value"] = ratio*100
    global frame_now, frame_count
    frame_now = int((frame_count - 1) * ratio)
    jump_to_frame()


video_Progressbar.bind("<B1-Motion>", video_progressbar_leftDrag)
video_Progressbar.bind("<Button-1>", video_progressbar_leftDrag)
video_frame_Label.bind("<B1-Motion>", video_progressbar_leftDrag)
video_frame_Label.bind("<Button-1>", video_progressbar_leftDrag)


# 输入路径 初始化
def submit_path(_):
    global \
        input_video_path_1, \
        input_video_path_2, \
        input_video_path_now, \
        scale, \
        main_rendering_Cap, \
        frame_count, \
        frame_height, \
        frame_width, \
        new_frame_height, \
        new_frame_width, \
        fps, \
        frame_now
    input_video_path_1 = input_video_1_Entry.get()
    input_video_path_2 = input_video_2_Entry.get()
    input_video_path_now = input_video_path_1
    # 渲染控件
    frame_num_Frame.grid(row=2, column=0)
    change_video_preview_Button.grid(row=5, column=0, pady=8)
    change_video_scale_Button.grid(row=6, column=0)
    cmp_set_Frame.grid(row=7, column=0, pady=5)
    log_Frame.grid(row=8, column=0, pady=8)

    # sec_rendering_Cap = cv2.VideoCapture(input_video_path_1)
    main_rendering_Cap = cv2.VideoCapture(input_video_path_1)
    main_rendering_Cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = main_rendering_Cap.read()
    if ret:
        # 获取尺寸 判断缩放
        frame_height, frame_width, _ = frame.shape
        video_size_Label.config(text=str(frame_width) + " x " + str(frame_height))
        fps = main_rendering_Cap.get(cv2.CAP_PROP_FPS)
        video_fps_Label.config(text=str(fps) + " FPS")
        video_size_Label.grid(row=0, column=0)
        video_fps_Label.grid(row=0, column=1, padx=8)
        if (
            frame_height > root_Tk.winfo_screenheight() * 5 / 6
            or frame_width > root_Tk.winfo_screenwidth() * 4 / 5
        ):
            scale = max(
                (1.2 * frame_height + 100) / root_Tk.winfo_screenheight(),
                (1.2 * frame_width + 500) / root_Tk.winfo_screenwidth(),
                1.5,
            )
            new_frame_width, new_frame_height = (
                int(frame_width / scale),
                int(frame_height / scale),
            )
            frame = cv2.resize(frame, (new_frame_width, new_frame_height))
            log.info(
                f"视频画幅过大 预览画面已缩小(1/{scale:.2f}-->{new_frame_width}x{new_frame_height})"
            )
            change_video_scale_Button.config(state=tk.NORMAL)
        else:
            new_frame_width, new_frame_height = frame_width, frame_height
            scale = False

        # 重写进度条
        video_review_Label.grid(row=0, column=0, pady=5)

        video_Progressbar.config(length=new_frame_width)
        video_Progressbar.grid(row=2, column=0)

        # 渲染进度
        frame_now = 0
        jump_to_frame()

        # 初始化右侧控件
        frame_count = int(main_rendering_Cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count_Label.config(text=f" / {frame_count - 1:9d}")
        frame_now_Tkint.set(0)

        def _long_text_cut(text):
            if len(text) > 49:
                text = text[0:49] + "\n" + text[49:]
            if len(text) > 99:
                text = text[0:99] + "\n" + text[99:]
            if len(text) > 149:
                text = text[0:149] + "\n" + text[149:]
            return text

        video_path_review_Label.config(
            text=_long_text_cut(input_video_1_Entry.get())
            + "\n"
            + _long_text_cut(input_video_2_Entry.get())
        )

        # 绘制进度条的帧提示
        global video_frame_img
        video_frame_img = (
            np.ones((VIDEO_FRAME_IMG_HEIGHT, new_frame_width, 3), np.uint8) * 224
        )
        video_frame_img[-1, :, :] = 1
        for frame_num in range(0, frame_count):
            draw_video_frame_Label_frameColor(frame_num, (0, 0, 0))
        flush_video_frame_Label()

        video_frame_Label.grid(row=3, column=0)

    else:
        log.error("无法打开" + input_video_path_1)

    root_Tk.focus_set()


# 右侧控件

# 路径输入
input_video_Frame = ttk.Frame(right_Frame)
input_video_Frame.grid(row=1, column=0, pady=15)

video_path_review_Label = ttk.Label(input_video_Frame, text="输入视频路径名")
video_path_review_Label.grid(row=1, column=0, columnspan=2, pady=8)

input_video_1_entry_Style = ttk.Style()
input_video_1_entry_Style.configure("Default.TEntry", foreground="black")
input_video_2_entry_Style = ttk.Style()
input_video_2_entry_Style.configure("Select.TEntry", foreground="red")

input_video_1_Entry = ttk.Entry(input_video_Frame, width=40, style="Select.TEntry")
input_video_1_Entry.grid(row=2, column=0)
input_video_1_Entry.focus_set()

input_video_1_Entry.bind("<Return>", lambda _: input_video_2_Entry.focus_set())

input_video_2_Entry = ttk.Entry(input_video_Frame, width=40, style="Default.TEntry")
input_video_2_Entry.grid(row=3, column=0, pady=8)

input_video_2_Entry.bind("<Return>", submit_path)

input_video_Button = ttk.Button(
    input_video_Frame, text="提交", width=4, command=lambda: submit_path(None)
)
input_video_Button.grid(row=4, column=0)


def change_video_preview():
    global input_video_path_now, main_rendering_Cap
    if input_video_path_now == input_video_path_2:
        input_video_path_now = input_video_path_1
        input_video_1_Entry.config(style="Select.TEntry")
        input_video_2_Entry.config(style="Default.TEntry")
    else:
        input_video_path_now = input_video_path_2
        input_video_1_Entry.config(style="Default.TEntry")
        input_video_2_Entry.config(style="Select.TEntry")
    main_rendering_Cap = cv2.VideoCapture(input_video_path_now)
    jump_to_frame()


change_video_preview_Button = ttk.Button(
    input_video_Frame, text="切换预览视频", width=12, command=change_video_preview
)

video_review_on_drag_start_x: int
video_review_on_drag_start_y: int
cut_scale_x_start: int
cut_scale_y_start: int


def video_review_on_drag_start(event):
    global \
        video_review_on_drag_start_x, \
        video_review_on_drag_start_y, \
        cut_scale_x_start, \
        cut_scale_y_start
    video_review_on_drag_start_x = event.x
    video_review_on_drag_start_y = event.y
    cut_scale_x_start = cut_scale_x
    cut_scale_y_start = cut_scale_y


def video_review_on_drag(event):
    global cut_scale_x, cut_scale_y
    cut_scale_x = cut_scale_x_start - event.x + video_review_on_drag_start_x
    cut_scale_y = cut_scale_y_start - event.y + video_review_on_drag_start_y
    if cut_scale_x + new_frame_width > frame_width:
        cut_scale_x = frame_width - new_frame_width
    if cut_scale_x < 0:
        cut_scale_x = 0
    if cut_scale_y + new_frame_height > frame_height:
        cut_scale_y = frame_height - new_frame_height
    if cut_scale_y < 0:
        cut_scale_y = 0
    jump_to_frame()


def change_video_scale():
    global cut_scale
    if cut_scale:
        video_review_Label.unbind("<Button-1>")
        video_review_Label.unbind("<B1-Motion>")
        cut_scale = False
    else:
        video_review_Label.bind("<Button-1>", video_review_on_drag_start)
        video_review_Label.bind("<B1-Motion>", video_review_on_drag)
        cut_scale = True
    jump_to_frame()


change_video_scale_Button = ttk.Button(
    input_video_Frame, text="切换缩放", width=8, command=change_video_scale
)
change_video_scale_Button.config(state=tk.DISABLED)

# 帧数
frame_num_Frame = ttk.Frame(right_Frame)


def enter_to_change_frame_now(_):
    global frame_now, frame_count

    frame_now = int(frame_now_Entry.get())
    if frame_now < 0:
        frame_now = 0
    if frame_now >= frame_count:
        frame_now = frame_count - 1

    jump_to_frame()
    root_Tk.focus_set()


frame_now_Frame = ttk.Frame(frame_num_Frame)
frame_now_Frame.grid(row=1, column=0)

frame_now_Tkint = tk.IntVar()
frame_now_Entry = ttk.Entry(frame_now_Frame, textvariable=frame_now_Tkint, width=5)
frame_now_Entry.bind("<Return>", enter_to_change_frame_now)
frame_now_Entry.grid(row=0, column=0)

frame_count_Label = ttk.Label(frame_now_Frame, text=" / NULL")
frame_count_Label.grid(row=0, column=1)


# 视频信息
video_info_Frame = ttk.Frame(right_Frame)
video_info_Frame.grid(row=3, column=0, pady=10)

video_size_Label = ttk.Label(video_info_Frame)
video_fps_Label = ttk.Label(video_info_Frame)


cmp_set_Frame = ttk.Frame(right_Frame)


cmp_combobox_Frame = ttk.Frame(cmp_set_Frame)
cmp_combobox_Frame.grid(row=0, column=0)

cmp_select_Combobox = ttk.Combobox(
    cmp_combobox_Frame, width=4, values=["ssim", "psnr", "vmaf", "mse"]
)
cmp_select_Combobox.state(["readonly"])
cmp_select_Combobox.set(config.get("DEFAULT", "cmp_type"))
cmp_select_Combobox.grid(row=0, column=0, padx=5)


cmp_hwaccel_vals = [
    "none",
    "cuda",
    "vaapi",
    "dxva2",
    "qsv",
    "d3d11va",
    "opencl",
    "vulkan",
    "d3d12va",
]

cmp_hwaccel_1_Combobox = ttk.Combobox(
    cmp_combobox_Frame,
    width=7,
    values=cmp_hwaccel_vals,
)
cmp_hwaccel_1_Combobox.set(config.get("DEFAULT", "cmp_hwaccel_1"))
cmp_hwaccel_1_Combobox.grid(row=0, column=1, padx=5)

cmp_hwaccel_2_Combobox = ttk.Combobox(
    cmp_combobox_Frame,
    width=7,
    values=cmp_hwaccel_vals,
)
cmp_hwaccel_2_Combobox.set(config.get("DEFAULT", "cmp_hwaccel_2"))
cmp_hwaccel_2_Combobox.grid(row=0, column=2)


start_cmp_Frame = tk.Frame(cmp_set_Frame)
start_cmp_Frame.grid(row=1, column=0, pady=15)

extreme_value_Frame = ttk.Frame(start_cmp_Frame)
extreme_value_Frame.grid(row=0, column=0)

max_data_Label = ttk.Label(
    extreme_value_Frame, width=14, anchor="center", foreground="#00BB00"
)
max_data_Label.grid(row=0, column=0)
max_data_Label.config(state=tk.DISABLED)

avg_data_Label = ttk.Label(
    extreme_value_Frame, width=14, anchor="center", foreground="#BBBB00"
)
avg_data_Label.grid(row=0, column=1)
avg_data_Label.config(state=tk.DISABLED)

min_data_Label = ttk.Label(
    extreme_value_Frame, width=14, anchor="center", foreground="#BB0000"
)
min_data_Label.grid(row=0, column=2)
min_data_Label.config(state=tk.DISABLED)

show_data_Label = ttk.Label(start_cmp_Frame, width=42, anchor="center")
show_data_Label.grid(row=1, column=0, pady=5)

start_cmp_Button = ttk.Button(start_cmp_Frame, text="开始")
start_cmp_Button.grid(row=2, column=0)


def show_plt():
    plt.figure(figsize=(16, 9), dpi=72)

    algorithm.get_plt()

    plt.xlabel("frame num")
    plt.ylabel(cmp_select_Combobox.get())
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=20))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=25))
    plt.legend()
    plt.grid(True)

    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # 文本样式
        x, y = sel.target
        sel.annotation.set_text(
            f"{'frame num:':<12}{str(round(x, 6)):>9}\n{cmp_select_Combobox.get() + ' ' + sel.artist.get_label() + ':':<12}{str(round(y, 6)):>9}"
        )
        sel.annotation.set_fontsize(12)
        sel.annotation.set_color((0.49, 0.1216, 0.682))
        sel.annotation.get_bbox_patch().set_facecolor((0.8, 0.8, 0.8))
        sel.annotation.get_bbox_patch().set_alpha(0.8)

    plt.show()


show_plt_Button = ttk.Button(start_cmp_Frame, text="显示统计图", command=show_plt)
show_plt_Button.grid(row=3, column=0)

data_file_button_Frame = ttk.Frame(start_cmp_Frame)
data_file_button_Frame.grid(row=4, column=0)


def input_data_file():
    file_path = filedialog.askopenfilename(
        title="选择导入文件", filetypes=[("Text file", "*.*")]
    )
    if file_path:
        config_output_log_path = os.path.join(config_dir, "output.log")
        shutil.copyfile(file_path, config_output_log_path)
        algorithm.change_algorithm()
        with open(config_output_log_path, "r", encoding="utf-8") as file:
            algorithm.read_file(file)
        algorithm.get_data()
        max_data_Label.config(state=tk.NORMAL, text=f"Max:{algorithm.max_data}")
        avg_data_Label.config(state=tk.NORMAL, text=f"Avg:{algorithm.avg_data}")
        min_data_Label.config(state=tk.NORMAL, text=f"Min:{algorithm.min_data}")
        log.info("已导入数据文件: " + file_path)


input_data_file_Button = ttk.Button(
    data_file_button_Frame, text="导入数据", command=input_data_file
)
input_data_file_Button.grid(row=0, column=0)


def output_data_file():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".log",
        filetypes=[("Text file", "*.*")],
    )
    if file_path:
        shutil.copyfile(config_dir + "\\output.log", file_path)
        log.info("已导出数据文件: " + file_path)


output_data_file_Button = ttk.Button(
    data_file_button_Frame, text="导出数据", command=output_data_file
)
output_data_file_Button.grid(row=0, column=1)


log_Frame = ttk.Frame(right_Frame)

log_vScrollbar = ttk.Scrollbar(log_Frame)
log_vScrollbar.grid(row=0, column=1, sticky="ns")

log_Text = tk.Text(log_Frame, width=45, height=10, yscrollcommand=log_vScrollbar.set)
log_Text.grid(row=0, column=0, sticky="nsew")


log_vScrollbar.config(command=log_Text.yview)


class algorithm:
    algorithm: str
    filter_option: str
    data: list = []
    max_data: float
    min_data: float
    avg_data: float
    is_encoding: bool = False

    @staticmethod
    def find_line_ssim(line) -> list[str]:
        match_obj = re.search(r"Y(.*)\(", line)
        if match_obj is None:
            log.error("Search ssim error")
            return []
        return re.findall(r"[\d\.]+", match_obj.group(1))

    @staticmethod
    def find_line_psnr(line) -> list[str]:
        match_obj = re.search(r"psnr_avg(.+)$", line)
        if match_obj is None:
            log.error("Search psnr error")
            return []
        return re.findall(r"[\d\.]+", match_obj.group(1))

    @staticmethod
    def find_line_mse(line) -> list[str]:
        match_obj = re.search(r"mse_avg(.+)psnr_avg", line)
        if match_obj is None:
            log.error("Search mse error")
            return []
        return re.findall(r"[\d\.]+", match_obj.group(1))

    find_line = find_line_ssim

    @staticmethod
    def read_file_ssim_psnr_mse(file):
        for line in file:
            algorithm.data.append([float(v) for v in algorithm.find_line(line)])

    @staticmethod
    def read_file_vmaf(file):
        for line in file:
            text = re.search(r'vmaf="([\d\.]+)"', line)
            if text:
                algorithm.data.append(float(text.group(1)))
            else:
                text = re.search(
                    r'<metric name="vmaf" min="([\d\.]+)" max="([\d\.]+)" mean="([\d\.]+)"',
                    line,
                )
                if text:
                    algorithm.min_data = float(text.group(1))
                    algorithm.max_data = float(text.group(2))
                    algorithm.avg_data = float(text.group(3))

    read_file = read_file_ssim_psnr_mse

    @staticmethod
    def get_data_ssim():
        algorithm.min_data = algorithm.data[0][3]
        algorithm.max_data = algorithm.data[0][3]
        algorithm.avg_data = 0
        for v in algorithm.data:
            algorithm.avg_data += v[3]
            if v[3] > algorithm.max_data:
                algorithm.max_data = v[3]
            if v[3] < algorithm.min_data:
                algorithm.min_data = v[3]
        algorithm.avg_data = round(algorithm.avg_data / frame_count, 6)

    @staticmethod
    def get_data_psnr_mse():
        algorithm.min_data = algorithm.data[0][0]
        algorithm.max_data = algorithm.data[0][0]
        algorithm.avg_data = 0
        for v in algorithm.data:
            algorithm.avg_data += v[0]
            if v[0] > algorithm.max_data:
                algorithm.max_data = v[0]
            if v[0] < algorithm.min_data:
                algorithm.min_data = v[0]
        algorithm.avg_data = round(algorithm.avg_data / frame_count, 2)

    @staticmethod
    def get_data_vmaf():
        pass

    get_data = get_data_ssim

    @staticmethod
    def listen_data_ssim():
        try:
            v = algorithm.data[frame_now]
        except Exception:
            return
        if v:
            dif = algorithm.max_data - algorithm.min_data
            if dif == 0:
                color_percent = 1
            else:
                color_percent = (float(v[3]) - algorithm.min_data) / dif
            show_data_Label.config(
                text=f"Y:{v[0]}  U:{v[1]}  V:{v[2]}  All:{v[3]}",
                foreground=f"#{round(187 if color_percent <= 0.5 else 374 * (1 - color_percent)):02X}{round(187 if color_percent >= 0.5 else 374 * color_percent):02X}00",
            )

    @staticmethod
    def listen_data_psnr_mse():
        try:
            v = algorithm.data[frame_now]
        except Exception:
            return
        if v:
            dif = algorithm.max_data - algorithm.min_data
            if dif == 0:
                color_percent = 1
            else:
                color_percent = (float(v[0]) - algorithm.min_data) / dif
            show_data_Label.config(
                text=f"Y:{v[1]}  U:{v[2]}  V:{v[3]}  All:{v[0]}",
                foreground=f"#{round(187 if color_percent <= 0.5 else 374 * (1 - color_percent)):02X}{round(187 if color_percent >= 0.5 else 374 * color_percent):02X}00",
            )

    @staticmethod
    def listen_data_vmaf():
        try:
            v = algorithm.data[frame_now]
        except Exception:
            return
        if v:
            dif = algorithm.max_data - algorithm.min_data
            if dif == 0:
                color_percent = 1
            else:
                color_percent = (float(v) - algorithm.min_data) / dif
            show_data_Label.config(
                text=f"Score:{v}",
                foreground=f"#{round(187 if color_percent <= 0.5 else 374 * (1 - color_percent)):02X}{round(187 if color_percent >= 0.5 else 374 * color_percent):02X}00",
            )

    listen_data = listen_data_ssim

    @staticmethod
    def get_plt_ssim():
        x = [i for i in range(0, len(algorithm.data))]

        y = [i[0] for i in algorithm.data]
        plt.plot(x, y, color="red", label="Y")

        y = [i[1] for i in algorithm.data]
        plt.plot(x, y, color="green", label="U")

        y = [i[2] for i in algorithm.data]
        plt.plot(x, y, color="blue", label="V")

        y = [i[3] for i in algorithm.data]
        plt.plot(x, y, color="black", label="All")

        plt.axhline(y=algorithm.max_data, linestyle="--", label="Max", color="#00BB00")
        plt.axhline(y=algorithm.avg_data, linestyle="--", label="Avg", color="#BBBB00")
        plt.axhline(y=algorithm.min_data, linestyle="--", label="Min", color="#BB0000")

    @staticmethod
    def get_plt_psnr_mse():
        x = [i for i in range(0, len(algorithm.data))]

        y = [i[1] for i in algorithm.data]
        plt.plot(x, y, color="red", label="Y")

        y = [i[2] for i in algorithm.data]
        plt.plot(x, y, color="green", label="U")

        y = [i[3] for i in algorithm.data]
        plt.plot(x, y, color="blue", label="V")

        y = [i[0] for i in algorithm.data]
        plt.plot(x, y, color="black", label="All")

        plt.axhline(y=algorithm.max_data, linestyle="--", label="Max", color="#00BB00")
        plt.axhline(y=algorithm.avg_data, linestyle="--", label="Avg", color="#BBBB00")
        plt.axhline(y=algorithm.min_data, linestyle="--", label="Min", color="#BB0000")

    @staticmethod
    def get_plt_vmaf():
        x = [i for i in range(0, len(algorithm.data))]

        y = [i for i in algorithm.data]
        plt.plot(x, y, color="black", label="Score")

        plt.axhline(y=algorithm.max_data, linestyle="--", label="Max", color="#00BB00")
        plt.axhline(y=algorithm.avg_data, linestyle="--", label="Avg", color="#BBBB00")
        plt.axhline(y=algorithm.min_data, linestyle="--", label="Min", color="#BB0000")

    get_plt = get_plt_ssim

    @staticmethod
    def change_algorithm():
        algorithm.algorithm = cmp_select_Combobox.get()
        match algorithm.algorithm:
            case "ssim":
                algorithm.find_line = algorithm.find_line_ssim
                algorithm.filter_option = "ssim=f="
                algorithm.read_file = algorithm.read_file_ssim_psnr_mse
                algorithm.get_data = algorithm.get_data_ssim
                algorithm.listen_data = algorithm.listen_data_ssim
                algorithm.get_plt = algorithm.get_plt_ssim
            case "psnr":
                algorithm.find_line = algorithm.find_line_psnr
                algorithm.filter_option = "psnr=f="
                algorithm.read_file = algorithm.read_file_ssim_psnr_mse
                algorithm.get_data = algorithm.get_data_psnr_mse
                algorithm.listen_data = algorithm.listen_data_psnr_mse
                algorithm.get_plt = algorithm.get_plt_psnr_mse
            case "vmaf":
                algorithm.filter_option = "libvmaf=log_fmt=xml:log_path="
                algorithm.read_file = algorithm.read_file_vmaf
                algorithm.get_data = algorithm.get_data_vmaf
                algorithm.listen_data = algorithm.listen_data_vmaf
                algorithm.get_plt = algorithm.get_plt_vmaf
            case "mse":
                algorithm.find_line = algorithm.find_line_mse
                algorithm.filter_option = "psnr=f="
                algorithm.read_file = algorithm.read_file_ssim_psnr_mse
                algorithm.get_data = algorithm.get_data_psnr_mse
                algorithm.listen_data = algorithm.listen_data_psnr_mse
                algorithm.get_plt = algorithm.get_plt_psnr_mse


def start_to_ready():
    global end_all_thread
    end_all_thread = False

    save_config()

    input_video_1_Entry.config(state=tk.DISABLED)
    input_video_2_Entry.config(state=tk.DISABLED)
    input_video_Button.config(state=tk.DISABLED)
    change_video_preview_Button.config(state=tk.DISABLED)
    cmp_select_Combobox.config(state=tk.DISABLED)
    frame_now_Entry.config(state=tk.DISABLED)

    start_cmp_Button.config(text="终止", command=cancel_encoding)

    video_review_Label.unbind("<MouseWheel>")

    video_Progressbar.unbind("<MouseWheel>")
    video_Progressbar.unbind("<B1-Motion>")
    video_Progressbar.unbind("<Button-1>")

    video_frame_Label.unbind("<MouseWheel>")
    video_frame_Label.unbind("<B1-Motion>")
    video_frame_Label.unbind("<Button-1>")

    start_encoding()


def end_to_ready():
    global frame_now, end_all_thread

    end_all_thread = True

    input_video_1_Entry.config(state=tk.NORMAL)
    input_video_2_Entry.config(state=tk.NORMAL)
    input_video_Button.config(state=tk.NORMAL)
    change_video_preview_Button.config(state=tk.NORMAL)
    cmp_select_Combobox.config(state=tk.NORMAL)
    frame_now_Entry.config(state=tk.NORMAL)

    video_review_Label.bind("<MouseWheel>", video_progressbar_mousewheel)

    video_Progressbar.bind("<MouseWheel>", video_progressbar_mousewheel)
    video_Progressbar.bind("<B1-Motion>", video_progressbar_leftDrag)
    video_Progressbar.bind("<Button-1>", video_progressbar_leftDrag)

    video_frame_Label.bind("<B1-Motion>", video_progressbar_leftDrag)
    video_frame_Label.bind("<Button-1>", video_progressbar_leftDrag)
    video_frame_Label.bind("<MouseWheel>", video_progressbar_mousewheel)

    start_cmp_Button.config(text="开始", command=start_to_ready, state=tk.NORMAL)


start_cmp_Button.config(command=start_to_ready)


def Listener_data():
    global is_listener_data
    while is_listener_data:
        if not algorithm.is_encoding:
            algorithm.listen_data()
        sleep(0.2)


Thread(target=Listener_data).start()


def flush_ffmpeg_speed_progress(frame):
    global frame_now
    frame -= 1
    if frame >= frame_count:
        frame_now = frame_count - 1
        log.warning(
            f"ffmpeg 解码的视频帧数({frame + 1})与 cv2 解码的帧数({frame_count})不一致"
        )
    else:
        frame_now = frame
    jump_to_frame()
    sleep(0.2)


def Thread_encoding():
    global process_ffmpeg
    algorithm.is_encoding = True
    log.info("开始")

    algorithm.change_algorithm()

    ff_cmd = [
        v
        for v in [
            "ffmpeg",
            None
            if cmp_hwaccel_1_Combobox.get() == "none"
            else f"-hwaccel {cmp_hwaccel_1_Combobox.get()}",
            "-i",
            str(input_video_path_1),
            None
            if cmp_hwaccel_2_Combobox.get() == "none"
            else f"-hwaccel {cmp_hwaccel_2_Combobox.get()}",
            "-i",
            str(input_video_path_2),
            "-lavfi",
            algorithm.filter_option
            + config_dir.replace("C:", "", 1).replace("\\", "/")
            + "/output.log",
            "-f",
            "null",
            "-",
        ]
        if v is not None
    ]

    log.info("ffmpeg 命令: " + " ".join(ff_cmd))

    process_ffmpeg = subprocess.Popen(
        ff_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    while True:
        if process_ffmpeg.stdout is None:
            sleep(0.1)
            continue
        line = process_ffmpeg.stdout.readline()
        if not line and process_ffmpeg.poll() is not None:
            break
        if line:
            print(f"ffmpeg info: {line}", end="")
            frame = re.search(r"frame=\s*(\d+)\s", line)
            if frame:
                flush_ffmpeg_speed_progress(int(frame.group(1)))
    process_ffmpeg.wait()

    if algorithm.is_encoding:
        algorithm.data = []
        with open(config_dir + "\\output.log", "r", encoding="utf-8") as file:
            algorithm.read_file(file)

        algorithm.get_data()

        max_data_Label.config(state=tk.NORMAL, text=f"Max:{algorithm.max_data}")
        avg_data_Label.config(state=tk.NORMAL, text=f"Avg:{algorithm.avg_data}")
        min_data_Label.config(state=tk.NORMAL, text=f"Min:{algorithm.min_data}")

        log.info("完成")
        algorithm.is_encoding = False
    end_to_ready()


def start_encoding():
    global frame_count, frame_now

    draw_video_frame_Label_range(0, frame_count, (27, 241, 255))
    flush_video_frame_Label()

    frame_now = 0

    Thread(target=Thread_encoding).start()


def cancel_encoding():
    log.warning("手动强制终止")
    algorithm.is_encoding = False
    try:
        process_ffmpeg.kill()
    except Exception as e:
        log.error("强制终止失败: 终止线程失败: " + repr(e))
    end_to_ready()


root_Tk.mainloop()
