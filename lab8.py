import sys
import math
import time
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QGridLayout, QMessageBox,
    QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


# Методы интерполяции
NEAREST = "nearest"
BILINEAR = "bilinear"
BICUBIC = "bicubic"
METHODS = (NEAREST, BILINEAR, BICUBIC)


def numpy_to_qpixmap(arr):
    arr = np.ascontiguousarray(np.clip(np.round(arr), 0, 255).astype(np.uint8))
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def load_image(path):
    qimg = QImage(path)
    if qimg.isNull():
        raise ValueError(f"Не удалось загрузить: {path}")
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w = qimg.width()
    h = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.sizeInBytes())
    # строки QImage выровнены по 4 байта, поэтому читаем bytesPerLine байт на строку
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, qimg.bytesPerLine()))
    return arr[:, :w * 3].reshape((h, w, 3)).copy()


# ---------------------------------------------------------------- матрицы преобразований
def scale_matrix(kx, ky):
    return np.array([[kx, 0.0],
                     [0.0, ky]], dtype=np.float64)


def rotation_matrix(angle_deg):
    # ось Y направлена вниз, поэтому положительный угол - поворот по часовой стрелке
    phi = math.radians(angle_deg)
    c = math.cos(phi)
    s = math.sin(phi)
    return np.array([[c, -s],
                     [s, c]], dtype=np.float64)


def shear_matrix(sx, sy):
    # x' = x + sx * y,  y' = y + sy * x
    return np.array([[1.0, sx],
                     [sy, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------- интерполяция
def clamp(idx, n):
    # пиксели за границей изображения заменяются крайними
    return np.clip(idx, 0, n - 1)


def sample_nearest(f, x, y):
    # g(x, y) = f(round(x), round(y))
    h, w = f.shape[:2]
    xi = clamp(np.rint(x).astype(np.int64), w)
    yi = clamp(np.rint(y).astype(np.int64), h)
    return f[yi, xi]


def sample_bilinear(f, x, y):
    # g(x, y) = (1-a)(1-b) f(l, k) + a(1-b) f(l+1, k)
    #         + b(1-a) f(l, k+1) + a b f(l+1, k+1),
    # l = floor(x), k = floor(y), a = x - l, b = y - k
    h, w = f.shape[:2]
    l = np.floor(x).astype(np.int64)
    k = np.floor(y).astype(np.int64)
    a = (x - l)[:, None]
    b = (y - k)[:, None]

    l0 = clamp(l, w)
    l1 = clamp(l + 1, w)
    k0 = clamp(k, h)
    k1 = clamp(k + 1, h)

    return ((1 - a) * (1 - b) * f[k0, l0] +
            a * (1 - b) * f[k0, l1] +
            b * (1 - a) * f[k1, l0] +
            a * b * f[k1, l1])


def cubic_weights(t):
    # Одномерные кубические полиномы Лагранжа для узлов -1, 0, 1, 2.
    # Произведение двух таких полиномов (по a и по b) даёт коэффициенты c1..c16.
    return (-t * (t - 1) * (t - 2) / 6.0,
            (t + 1) * (t - 1) * (t - 2) / 2.0,
            -(t + 1) * t * (t - 2) / 2.0,
            (t + 1) * t * (t - 1) / 6.0)


def sample_bicubic(f, x, y):
    # g(x, y) = sum_{m=-1..2} sum_{n=-1..2} wx[m] * wy[n] * f(l + m, k + n)
    h, w = f.shape[:2]
    l = np.floor(x).astype(np.int64)
    k = np.floor(y).astype(np.int64)
    a = (x - l)[:, None]
    b = (y - k)[:, None]

    wx = cubic_weights(a)
    wy = cubic_weights(b)

    res = np.zeros((x.shape[0], f.shape[2]), dtype=np.float64)
    for n in range(4):
        kk = clamp(k + n - 1, h)
        for m in range(4):
            ll = clamp(l + m - 1, w)
            res += wx[m] * wy[n] * f[kk, ll]
    return res


SAMPLERS = {
    NEAREST: sample_nearest,
    BILINEAR: sample_bilinear,
    BICUBIC: sample_bicubic,
}


# ---------------------------------------------------------------- аффинное преобразование
def affine_transform(img, M, method):
    # Прямое преобразование относительно центра изображения: dst = M * (src - c) + c.
    # Чтобы в результате не было "дыр", оно выполняется в обратном виде: для каждого
    # пикселя результата находится прообраз src = M^-1 * (dst - c) + c (дробные
    # координаты), а значение в нём вычисляется выбранным методом интерполяции.
    h, w = img.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    inv = np.linalg.inv(M)

    jj, ii = np.mgrid[0:h, 0:w]
    dx = ii - cx
    dy = jj - cy
    x = inv[0, 0] * dx + inv[0, 1] * dy + cx
    y = inv[1, 0] * dx + inv[1, 1] * dy + cy

    # прообраз должен лежать внутри изображения, иначе пиксель остаётся чёрным
    mask = (x >= -0.5) & (x <= w - 0.5) & (y >= -0.5) & (y <= h - 0.5)

    out = np.zeros(img.shape, dtype=np.uint8)
    f = img.astype(np.float64)
    vals = SAMPLERS[method](f, x[mask], y[mask])
    out[mask] = np.clip(np.round(vals), 0, 255).astype(np.uint8)
    return out


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ЛР №10 — Интерполяция изображений (вариант 5)")
        self.resize(1400, 900)

        self.image = None
        self.res = {NEAREST: None, BILINEAR: None, BICUBIC: None}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        params = QGridLayout()
        params.setHorizontalSpacing(6)
        params.setVerticalSpacing(8)

        self.btn_open = QPushButton("Открыть")
        params.addWidget(self.btn_open, 0, 0)

        params.addWidget(QLabel("Масштаб X:"), 0, 1)
        self.edit_kx = QLineEdit("2.5")
        self.edit_kx.setFixedWidth(50)
        params.addWidget(self.edit_kx, 0, 2)

        params.addWidget(QLabel("Масштаб Y:"), 0, 3)
        self.edit_ky = QLineEdit("2.5")
        self.edit_ky.setFixedWidth(50)
        params.addWidget(self.edit_ky, 0, 4)

        self.btn_scale = QPushButton("Масштабирование")
        params.addWidget(self.btn_scale, 0, 5)

        params.addWidget(QLabel("Угол, °:"), 1, 1)
        self.edit_angle = QLineEdit("30")
        self.edit_angle.setFixedWidth(50)
        params.addWidget(self.edit_angle, 1, 2)

        self.btn_rotate = QPushButton("Поворот")
        params.addWidget(self.btn_rotate, 1, 5)

        params.addWidget(QLabel("Скос X:"), 2, 1)
        self.edit_sx = QLineEdit("0.5")
        self.edit_sx.setFixedWidth(50)
        params.addWidget(self.edit_sx, 2, 2)

        params.addWidget(QLabel("Скос Y:"), 2, 3)
        self.edit_sy = QLineEdit("0")
        self.edit_sy.setFixedWidth(50)
        params.addWidget(self.edit_sy, 2, 4)

        self.btn_shear = QPushButton("Скос")
        params.addWidget(self.btn_shear, 2, 5)

        self.btn_save_nearest = QPushButton("Сохранить (ближайший сосед)")
        params.addWidget(self.btn_save_nearest, 0, 6)

        self.btn_save_bilinear = QPushButton("Сохранить (билинейная)")
        params.addWidget(self.btn_save_bilinear, 1, 6)

        self.btn_save_bicubic = QPushButton("Сохранить (бикубическая)")
        params.addWidget(self.btn_save_bicubic, 2, 6)

        self.lbl_size = QLabel("Размер: -")
        params.addWidget(self.lbl_size, 0, 7)

        self.lbl_info = QLabel("Преобразование: -")
        params.addWidget(self.lbl_info, 1, 7)

        self.lbl_time = QLabel("Время: -")
        params.addWidget(self.lbl_time, 2, 7)

        root.addLayout(params)

        grid = QGridLayout()

        self.lb_src = self.make_label("Исходное изображение")
        self.lb_nearest = self.make_label("Ближайший сосед")
        self.lb_bilinear = self.make_label("Билинейная интерполяция")
        self.lb_bicubic = self.make_label("Бикубическая интерполяция")

        grid.addWidget(self.lb_src, 0, 0)
        grid.addWidget(self.lb_nearest, 0, 1)
        grid.addWidget(self.lb_bilinear, 1, 0)
        grid.addWidget(self.lb_bicubic, 1, 1)
        root.addLayout(grid, 1)

        self.btn_open.clicked.connect(self.on_open)
        self.btn_scale.clicked.connect(self.on_scale)
        self.btn_rotate.clicked.connect(self.on_rotate)
        self.btn_shear.clicked.connect(self.on_shear)
        self.btn_save_nearest.clicked.connect(self.on_save_nearest)
        self.btn_save_bilinear.clicked.connect(self.on_save_bilinear)
        self.btn_save_bicubic.clicked.connect(self.on_save_bicubic)

    def make_label(self, text):
        lb = QLabel(text)
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lb.setMinimumSize(300, 300)
        lb.setStyleSheet("QLabel { border: 1px solid #888; }")
        return lb

    def show_image(self, label, arr):
        if arr is None:
            return
        pix = numpy_to_qpixmap(arr)
        label.setPixmap(
            pix.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def read_float(self, edit):
        return float(edit.text().replace(",", "."))

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть изображение", "",
            "Изображение (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm *.tif *.tiff)"
        )
        if not path:
            return
        try:
            self.image = load_image(path)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            return

        for key in METHODS:
            self.res[key] = None
        for lb, text in ((self.lb_nearest, "Ближайший сосед"),
                         (self.lb_bilinear, "Билинейная интерполяция"),
                         (self.lb_bicubic, "Бикубическая интерполяция")):
            lb.clear()
            lb.setText(text)

        h, w = self.image.shape[:2]
        self.lbl_size.setText(f"Размер: {w} x {h}")
        self.lbl_info.setText("Преобразование: -")
        self.lbl_time.setText("Время: -")
        self.show_image(self.lb_src, self.image)

    def apply(self, M, info):
        # Применяет матрицу M тремя методами интерполяции и выводит результаты.
        times = []
        for key in METHODS:
            t0 = time.perf_counter()
            self.res[key] = affine_transform(self.image, M, key)
            times.append((time.perf_counter() - t0) * 1000.0)

        self.show_image(self.lb_nearest, self.res[NEAREST])
        self.show_image(self.lb_bilinear, self.res[BILINEAR])
        self.show_image(self.lb_bicubic, self.res[BICUBIC])

        self.lbl_info.setText(f"Преобразование: {info}")
        self.lbl_time.setText(
            f"Время: ближайший {times[0]:.0f} мс, "
            f"билинейная {times[1]:.0f} мс, бикубическая {times[2]:.0f} мс"
        )

    def on_scale(self):
        if self.image is None:
            QMessageBox.warning(self, "Внимание", "Сначала откройте изображение")
            return
        try:
            kx = self.read_float(self.edit_kx)
            ky = self.read_float(self.edit_ky)
        except ValueError:
            QMessageBox.warning(self, "Ошибка",
                                "Масштаб X и Масштаб Y должны быть числами")
            return
        if not (0.05 <= kx <= 10 and 0.05 <= ky <= 10):
            QMessageBox.warning(self, "Ошибка",
                                "Масштаб должен быть в диапазоне от 0.05 до 10")
            return
        self.apply(scale_matrix(kx, ky), f"масштабирование ({kx:g}; {ky:g})")

    def on_rotate(self):
        if self.image is None:
            QMessageBox.warning(self, "Внимание", "Сначала откройте изображение")
            return
        try:
            angle = self.read_float(self.edit_angle)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Угол должен быть числом")
            return
        self.apply(rotation_matrix(angle), f"поворот на {angle:g}°")

    def on_shear(self):
        if self.image is None:
            QMessageBox.warning(self, "Внимание", "Сначала откройте изображение")
            return
        try:
            sx = self.read_float(self.edit_sx)
            sy = self.read_float(self.edit_sy)
        except ValueError:
            QMessageBox.warning(self, "Ошибка",
                                "Скос X и Скос Y должны быть числами")
            return
        if abs(sx) > 3 or abs(sy) > 3:
            QMessageBox.warning(self, "Ошибка",
                                "Коэффициенты скоса должны быть от -3 до 3")
            return
        if abs(1.0 - sx * sy) < 1e-6:
            QMessageBox.warning(self, "Ошибка",
                                "Преобразование вырождено (1 - Скос X * Скос Y = 0)")
            return
        self.apply(shear_matrix(sx, sy), f"скос ({sx:g}; {sy:g})")

    def save_result(self, arr, default_name):
        if arr is None:
            QMessageBox.warning(self, "Внимание", "Нечего сохранять")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", default_name,
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)"
        )
        if path:
            numpy_to_qpixmap(arr).save(path)

    def on_save_nearest(self):
        self.save_result(self.res[NEAREST], "result_nearest.png")

    def on_save_bilinear(self):
        self.save_result(self.res[BILINEAR], "result_bilinear.png")

    def on_save_bicubic(self):
        self.save_result(self.res[BICUBIC], "result_bicubic.png")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
