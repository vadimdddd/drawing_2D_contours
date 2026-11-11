#!/usr/bin/env python3
"""Выделение толстых основных линий на растровых чертежах.

Запуск: .env/bin/python extract_contours.py
Маска: белые пиксели — линии, чёрные — фон. Области не заливаются.
Координаты и размер исходного изображения сохраняются.
"""

import re
import sys
import csv
import cv2
import html
import json
import math
import time
import fitz
import shutil
import hashlib
import argparse
import numpy as np
from pathlib import Path
from functools import lru_cache
from __future__ import annotations
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".pdf"}
PDF_DPI = 400

def components(mask):
    """Обёртка над connectedComponentsWithStats с приведением к uint8.

    OpenCV требует uint8; вызывающий код обычно передаёт bool-маски, и
    без явного cast'а часть версий OpenCV молча возвращает мусор.
    Возвращает (n, labels, stats, centroids), как оригинал.
    """
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)

def line_boxes(binary, length, horizontal=True):
    """Найти боксы длинных тонких горизонтальных (или вертикальных) штрихов.

    Штрих выделяется морфологическим открытием 1×length (или length×1):
    всё короче length исчезает, всё длиннее — остаётся как компонента.

    Фильтр «тонкости»: перпендикулярный размер штриха должен быть не больше
    max(25, min(h, w) * .012). Это отсекает крупные залитые пятна (стрелки,
    штриховку), но пропускает длинные линии — рамки, оси, линии таблиц.

    Возвращает список (x, y, w, h) без учёта смещения нулевой компоненты.
    """
    kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    _, _, stats, _ = components(opened)
    return [tuple(map(int, row[:4])) for row in stats[1:]
            if (row[2] if horizontal else row[3]) >= length
            and (row[3] if horizontal else row[2]) <= max(25, min(binary.shape) * .012)]

def service_regions(binary):
    """Рамка, верхнее обозначение, штамп, сетки таблиц и общий знак Ra.

    Геометрия определяется по самим линиям, а не по размеру листа A4/A3.
    Удаление штампа ограничено его прямоугольником, не всей нижней полосой.

    Возвращает (excluded, boxes, field):
      * excluded — uint8-маска 0/255: 255 там, где служебные зоны (их надо
        вычистить из рабочей копии перед основным анализом);
      * boxes   — список {'kind': str, 'box': [x1,y1,x2,y2]} для отчёта;
      * field   — [left, top, right, bottom] — внутренняя граница поля
        чертежа; используется только для диагностики в report.json.

    Правила:
      * рамка — самые длинные вертикальные/горизонтальные линии у краёв;
      * верхнее перевёрнутое обозначение — короткая горизонталь в верхней
        полосе, касающаяся левой границы поля;
      * таблицы — группы из ≥3 горизонталей с одинаковыми левой/правой
        границами и вертикальными бордюрами по краям и внутри;
      * штамп — ≥3 горизонтали длиной > .18w, заканчивающиеся у правой
        рамки; удаляется только их объединение, не вся нижняя полоса;
      * знак шероховатости — связные компоненты в верхней правой четверти,
        не пересекающиеся с уже исключёнными зонами.
    """
    h, w = binary.shape
    short = min(h, w)
    pad = max(3, round(short * .0015))
    horizontal = line_boxes(binary, max(45, round(short * .025)))
    vertical = line_boxes(binary, max(45, round(short * .025)), False)
    # Внешняя рамка: вертикали, проходящие почти всю высоту, и горизонтали,
    # проходящие почти всю ширину. left/right/top/bottom — координаты
    # внутренней границы поля (то, что остаётся после вырезания рамки).
    lefts = [x + bw for x, y, bw, bh in vertical if x < .16*w and bh > .65*h]
    rights = [x for x, y, bw, bh in vertical if x > .85*w and bh > .65*h]
    tops = [y + bh for x, y, bw, bh in horizontal if y < .055*h and bw > .7*w]
    bottoms = [y for x, y, bw, bh in horizontal if y > .94*h and bw > .7*w]
    left = max(lefts, default=0)
    right = min(rights, default=w)
    top = max(tops, default=0)
    bottom = min(bottoms, default=h)
    excluded = np.zeros_like(binary)
    boxes = []

    def exclude(x1, y1, x2, y2, kind, padding=pad):
        """Пометить область к исключению и записать её в boxes."""
        x1, y1 = max(0, int(x1-padding)), max(0, int(y1-padding))
        x2, y2 = min(w, int(x2+padding)), min(h, int(y2+padding))
        if x2 > x1 and y2 > y1:
            excluded[y1:y2, x1:x2] = 255
            boxes.append({"kind": kind, "box": [x1, y1, x2, y2]})

    if left:
        exclude(0, 0, left, h, "left_frame")
    if right < w:
        exclude(right, 0, w, h, "right_frame")
    if top:
        exclude(0, 0, w, top, "top_frame")
    if bottom < h:
        exclude(0, bottom, w, h, "bottom_frame")

    # Верхнее перевёрнутое обозначение соединено с левой границей поля.
    for x, y, bw, bh in horizontal:
        if top + pad < y < .10*h and x <= left + 2*pad and .08*w < bw < .8*w:
            exclude(left, top, x+bw, y+bh, "drawing_number")

    # Строки таблицы имеют совпадающие левую и правую границы.
    # Собираем горизонтали в группы, у которых x1 и x2 совпадают с
    # точностью до tolerance. Таблицу от набора размерных линий отличаем
    # по наличию вертикальных бордюров слева, справа и внутри.
    tolerance = max(12, round(short * .004))
    groups = []
    for box in sorted(horizontal, key=lambda b: b[1]):
        x, y, bw, bh = box
        if bw < .065*short or y <= top + pad or y >= bottom-pad:
            continue
        for group in groups:
            gx, _, gw, _ = group[0]
            if abs(x-gx) <= tolerance and abs(x+bw-gx-gw) <= tolerance:
                group.append(box)
                break
        else:
            groups.append([box])
    for group in groups:
        if len(group) < 3:
            continue
        x1 = min(b[0] for b in group)
        x2 = max(b[0]+b[2] for b in group)
        y1 = min(b[1] for b in group)
        y2 = max(b[1]+b[3] for b in group)
        # Границы по вертикали отличают таблицу от набора размерных линий.
        borders = [x for x, y, bw, bh in vertical
                   if y <= y1+tolerance and y+bh >= y2-tolerance
                   and x1-tolerance <= x <= x2+tolerance]
        left_border = any(abs(x-x1) <= tolerance for x in borders)
        right_border = any(abs(x-x2) <= tolerance for x in borders)
        internal = any(x1+tolerance < x < x2-tolerance for x in borders)
        if left_border and right_border and internal:
            exclude(x1, y1, x2, y2, "table")

    # Штамп: повторяющиеся горизонтали заканчиваются у правой рамки.
    # Требуем минимум три такие линии — иначе одиночная горизонталь у
    # правого края может быть частью детали.
    stamp_lines = [b for b in horizontal
                   if .70*h < b[1] < bottom-pad and b[2] > .18*w
                   and abs(b[0]+b[2]-right) < 3*tolerance]
    if len(stamp_lines) >= 3:
        for x, y, bw, bh in stamp_lines:
            exclude(x, y, right, bottom, "title_block")

    # Общий знак шероховатости в верхнем правом углу поля.
    # Выделяем его связанные компоненты, не обрезаем весь верх чертежа.
    roi = binary[:round(.14*h), round(.50*w):].copy()
    roi[excluded[:roi.shape[0], round(.50*w):] > 0] = 0
    n, labels, stats, _ = components(roi)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if y < .10*h and y+bh < .14*h:
            exclude(x+round(.50*w), y, x+bw+round(.50*w), y+bh, "top_annotation")
    return excluded, boxes, [left, top, right, bottom]

def estimate_width(binary):
    """Толстая мода толщин протяжённых горизонтальных/вертикальных штрихов.

    Идея: гистограмма «толщина → суммарная длина штрихов такой толщины».
    Тонкая мода — оси/размерные линии; толстая мода — основные линии
    чертежа. Возвращаем (width, histogram_dict):
      * width — взвешенное среднее в окне ±1 вокруг выбранной моды;
      * histogram_dict — {str(толщина): длина} для диагностики в отчёте.

    Почему «толстая», а не просто первая мода: первая мода почти всегда
    тонкая (оси/выноски), и если брать её, фильтр толстых линий не сработает.
    Штрихи секущих плоскостей могут быть толще самого контура, поэтому
    берём самую массивную моду среди «толще thin*1.65», а не максимум
    по гистограмме.
    """
    short = min(binary.shape)
    histogram = np.zeros(max(32, round(short*.012)), np.float64)
    for horizontal in (True, False):
        for x, y, w, h in line_boxes(binary, max(50, round(short*.025)), horizontal):
            width, length = (h, w) if horizontal else (w, h)
            if 1 <= width < len(histogram) and length > 12*width:
                histogram[width] += length
    smoothed = np.convolve(histogram, [.2, .6, .2], mode="same")
    peaks = [i for i in range(2, len(smoothed)-1)
             if smoothed[i] >= smoothed[i-1] and smoothed[i] > smoothed[i+1]
             and smoothed[i] >= max(100, smoothed.max()*.07)]
    if peaks:
        # Первая мода обычно тонкая (оси/размеры); более толстые короткие
        # штрихи секущих плоскостей могут быть толще самого контура.
        thin = min(peaks)
        thick = [p for p in peaks if p >= max(3, thin*1.65)]
        selected = max(thick or peaks, key=lambda p: smoothed[p])
        indices = np.arange(max(1, selected-1), min(len(histogram), selected+2))
        width = float(np.average(indices, weights=histogram[indices]))
    else:
        width = max(3., short/420)
    return width, {str(i): round(float(v)) for i, v in enumerate(histogram) if v}

def prune_branches(skeleton, distances, width):
    """Короткие ответвления от толстых линий: остатки стрелок и засечки.

    Изолированные открытые линии не обрезаются. Ветвь удаляется только
    от свободного конца до узла и лишь при признаке утолщения/короткой длине.

    distances — карта расстояний до фона (для оценки, где штрих «толстый»).
    Возвращает урезанный скелет (bool ndarray той же формы).
    """
    result = skeleton.copy()
    degree = cv2.filter2D(result.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    # degree == 2 означает «ровно два соседа» — это свободный конец линии
    # (сам пиксель в фильтр не попадает, поэтому 2 = 1 сосед + сам).
    endpoints = np.argwhere(result & (degree == 2))
    hh, ww = result.shape
    for ey, ex in endpoints:
        current, previous = (int(ey), int(ex)), None
        path = []
        # Идём от свободного конца до узла (len(neighbors) != 1).
        for _ in range(max(10, round(width*10))):
            y, x = current
            path.append(current)
            neighbors = [(yy, xx) for yy in range(max(0, y-1), min(hh, y+2))
                         for xx in range(max(0, x-1), min(ww, x+2))
                         if (yy, xx) != current and (yy, xx) != previous and result[yy, xx]]
            if len(neighbors) != 1:
                if len(neighbors) >= 2:
                    # Дошли до узла: оцениваем, стоит ли резать всю ветвь.
                    # Утолщение в радиусах (>0.7*width) — признак стрелки.
                    radii = [distances[yy, xx] for yy, xx in path[:-1]]
                    is_arrow = radii and max(radii) > width*.70
                    # Короткая ветвь (<1.4w) — засечка; «стрелочная» до 9w.
                    if len(path) < width*1.4 or (is_arrow and len(path) < width*9):
                        for yy, xx in path[:-1]:
                            result[yy, xx] = False
                break
            previous, current = current, neighbors[0]
    return result

@lru_cache(maxsize=1)
def annotation_templates():
    """Образцы шрифта из предоставленных чертежей; без OCR и загрузки моделей.

    Читает annotation_templates.npz (ключи — имена, значения — uint8
    0/255) и кэширует разобранный результат на всё время работы процесса.
    Кэш общий для всех вызовов glyph_match — это заметно ускоряет
    обработку, когда в папке много файлов.

    Возвращает список (name, aspect, normalized, dilated):
      * aspect — w/h исходной обрезки (для быстрого отсева по пропорциям);
      * normalized — bool 48×80, приведённая к единому размеру маска;
      * dilated    — normalized с дилатацией 3×3 (для coverage).
    """
    path = ROOT/'annotation_templates.npz'
    if not path.exists():
        raise ValueError(f'Не найден файл шаблонов символов: {path}')
    result = []
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            crop = data[name]
            # Обрезаем по bounding box — в NPZ могут быть лишние поля.
            xx,yy,ww,hh = cv2.boundingRect(cv2.findNonZero(crop))
            crop = crop[yy:yy+hh,xx:xx+ww]
            normalized = cv2.resize(crop, (48, 80), interpolation=cv2.INTER_AREA) >= 127
            result.append((name, crop.shape[1]/crop.shape[0], normalized,
                           cv2.dilate(normalized.astype(np.uint8), np.ones((3,3),np.uint8)) > 0))
    return result

def glyph_match(local, width):
    """Попытаться распознать local как один из образцов annotation_templates.

    Возвращает имя образца или None. Используется как фильтр: если
    компонента похожа на букву/цифру, её удаляют из маски.

    Ограничения (быстрый отсев):
      * высота в диапазоне (6*width, 25*width) — размер глифа относительно
        толщины линии;
      * пропорции 0.25 < w/h < 1.1 — как у add_annotation_template.py;
      * аспект совпадает с образцом с точностью до log 0.12.

    Метрики:
      * coverage — взаимное покрытие (нормализованная маска покрывает
        dilated-шаблон и наоборот), минимум из двух направлений;
      * IoU — пересечение/объединение без дилатации.
    Оба должны пройти порог: coverage > .94 и IoU > .69.
    """
    points = cv2.findNonZero(local.astype(np.uint8))
    if points is None:
        return None
    xx, yy, ww, hh = cv2.boundingRect(points)
    local = local[yy:yy+hh,xx:xx+ww]
    h, w = local.shape
    if not (6*width < h < 25*width and .25 < w/h < 1.1):
        return None
    normalized = cv2.resize(local.astype(np.uint8)*255, (48,80), interpolation=cv2.INTER_AREA) >= 127
    dilated = cv2.dilate(normalized.astype(np.uint8), np.ones((3,3),np.uint8)) > 0
    best = (0., None)
    for name, aspect, template, template_dilated in annotation_templates():
        if abs(math.log((w/h)/aspect)) > .12:
            continue
        coverage = min(np.count_nonzero(normalized & template_dilated)/max(1, normalized.sum()),
                       np.count_nonzero(template & dilated)/max(1, template.sum()))
        iou = np.count_nonzero(template & normalized)/max(1, np.count_nonzero(template | normalized))
        if coverage > .94 and iou > .69 and coverage > best[0]:
            best = (coverage, name)
    return best[1]

def parenthesis_side(local, width):
    """Форма узкой скобки — только кандидат, без текстового контекста не удаляем.

    Проверяем, что local похожа на «(» или «)»:
      * высота (6*width, 26*width), ширина/высота в (.06, .32);
      * в каждой строке есть пиксели (≥90% высоты заполнено);
      * ширина строки ≤1.8*width (тонкая, не буква);
      * центры строк ложатся на параболу с коэффициентом в
        (max(.5w, .035h), .25h) — выпуклость влево/вправо;
      * остаточная ошибка подгонки мала.

    Возвращает 'left' (выпуклая влево, «(»), 'right' («)») или None.
    """
    h, w = local.shape
    if not (6*width < h < 26*width and .06 < w/h < .32):
        return None
    ys = np.flatnonzero(local.any(axis=1))
    if len(ys) < .9*h:
        return None
    centers = []
    spans = []
    for y in ys:
        xs = np.flatnonzero(local[y])
        centers.append((xs[0]+xs[-1])/2)
        spans.append(xs[-1]-xs[0]+1)
    if np.percentile(spans, 90) > 1.8*width:
        return None
    # У скобки верхний и нижний концы находятся по одну сторону от середины.
    normalized_y = 2*ys/max(1,h-1)-1
    curve = np.polyfit(normalized_y, centers, 2)
    residual = np.sqrt(np.mean((np.polyval(curve,normalized_y)-centers)**2))
    if not (max(.5*width,.035*h) < abs(curve[0]) < .25*h):
        return None
    if abs(curve[1]) > .04*h or residual > max(1., .25*width):
        return None
    return 'left' if curve[0] > 0 else 'right'

def has_scale_text(binary):
    """Цифроподобные штрихи по обе стороны двоеточия внутри пары скобок.

    Ищем две «точки» (маленькие квадратные компоненты) на одной вертикали
    с зазором (.15h, .6h) и цифроподобные компоненты по обе стороны от
    их x. Это эвристика «между скобками написано 1:1 / 2:1 / …».

    Возвращает True/False. Не пытается распознать конкретные цифры —
    только сам факт «есть что-то похожее на текст с двоеточием».
    """
    h, w = binary.shape
    if not h or not w:
        return False
    _, _, stats, _ = components(binary)
    characters = [s for s in stats[1:] if .45*h < s[3] < h and .08*h < s[2] < .9*h]
    dots = [s for s in stats[1:] if .02*h < s[2] < .18*h and .02*h < s[3] < .18*h
            and s[4] > .35*s[2]*s[3]]
    for upper in dots:
        ux, uy = upper[0]+upper[2]/2, upper[1]+upper[3]/2
        for lower in dots:
            lx, ly = lower[0]+lower[2]/2, lower[1]+lower[3]/2
            if abs(ux-lx) > .06*h or not .15*h < ly-uy < .6*h:
                continue
            if not .1*h < uy < .65*h or not .35*h < ly < .9*h:
                continue
            if (any(s[0]+s[2] < ux for s in characters)
                    and any(s[0] > ux for s in characters)):
                return True
    return False

def scale_parentheses(selected, binary, segments, width, factor):
    """Находим обе скобки записи (1:1), даже если буква вида не распознана.

    selected — рабочая маска в масштабе factor, binary — исходная
    бинаризация (её читаем, чтобы увидеть цифры, которые фильтры уже
    могли удалить). segments — текущие фрагменты с 'box' в исходном
    масштабе; width — толщина в исходном масштабе; factor — апскейл.

    Возвращает {index: 'scale_parenthesis_left'|'scale_parenthesis_right'}
    для тех фрагментов, которые надо удалить из selected.
    """
    candidates = []
    for index, segment in enumerate(segments):
        x,y,x2,y2 = [round(v*factor) for v in segment['box']]
        side = parenthesis_side(selected[y:y2,x:x2] > 0, width*factor)
        if side:
            candidates.append((index,side,x,y,x2,y2))
    removed = {}
    for li,side,lx,ly,lx2,ly2 in candidates:
        if side != 'left':
            continue
        h = ly2-ly
        for ri,rside,rx,ry,rx2,ry2 in candidates:
            if rside != 'right' or not .5*h < rx-lx2 < 8*h:
                continue
            if abs((ry2-ry)-h) > .12*h or abs((ry+ry2-ly-ly2)/2) > .08*h:
                continue
            # Контекст читаем из исходной бинаризации: предыдущие фильтры
            # могли уже удалить/разбить цифры, особенно узкий штрих «1».
            text = binary[round(min(ly,ry)/factor):round(max(ly2,ry2)/factor),
                          round(lx2/factor):round(rx/factor)]
            if has_scale_text(text):
                removed[li] = 'scale_parenthesis_left'
                removed[ri] = 'scale_parenthesis_right'
    return removed

def paired_section_strokes(selected, distances, segments, glyphs, width, factor):
    """Пара соосных толстых штрихов с одинаковыми буквами обозначения разреза.

    Один короткий отрезок неотличим от контура по форме. Поэтому требуем
    одновременно повышенную толщину, пару сходных прямых и две буквы
    одного обозначения на одной стороне общей оси. Ориентация произвольная.

    selected/distances — рабочая маска и distanceTransform в масштабе factor;
    segments — текущие фрагменты с 'box'/'endpoints' в исходном масштабе;
    glyphs — уже удалённые буквы-обозначения (нужны как контекст);
    width — толщина в исходном масштабе; factor — апскейл.

    Возвращает {index: family} для удаляемых штрихов.
    """
    # Семейства: буквы, которые в разных размерах означают одну и ту же
    # плоскость разреза. Asmall = A, Ve_large = Ve и т.д.
    families = {'A': 'A', 'Asmall': 'A', 'Be': 'Be', 'Besmall': 'Be',
                'Ve': 'Ve', 'Ve_large': 'Ve', 'Ge': 'Ge', 'De': 'De', 'Ee': 'Ee'}
    letters = []
    for index, glyph in enumerate(glyphs):
        if glyph['glyph'] in families:
            x,y,x2,y2 = np.array(glyph['box'], dtype=float)*factor
            letters.append((index, families[glyph['glyph']],
                            np.array([(x+x2)/2, (y+y2)/2]), y2-y))
    if len(letters) < 2:
        return {}
    stroke_width = width*factor
    candidates = []
    for index, segment in enumerate(segments):
        x,y,x2,y2 = [round(v*factor) for v in segment['box']]
        # Только одиночные прямые подходящей длины (8w..40w).
        if segment['endpoints'] != 2 or not 8*stroke_width < max(x2-x,y2-y) < 40*stroke_width:
            continue
        local = (selected[y:y2,x:x2] > 0).astype(np.uint8)
        count, _ = cv2.connectedComponents(local, connectivity=8)
        if count != 2:
            continue
        points = cv2.findNonZero(local).reshape(-1,2).astype(float)
        center = points.mean(axis=0)
        centered = points-center
        # Главная ось штриха через SVD; нормаль — перпендикуляр к ней.
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        axis = axes[0]
        if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
            axis = -axis
        normal = np.array([-axis[1], axis[0]])
        along, across = centered@axis, centered@normal
        length = float(np.ptp(along)+1)
        if np.ptp(across)+1 > 1.8*stroke_width or length < 10*stroke_width:
            continue
        # Толщину измеряем на исходных штрихах, до восстановления скелета,
        # в средней части: присоединённая стрелка у конца не должна решать.
        middle = ((np.abs(across) < .15*stroke_width)
                  & (along > along.min()+.15*length)
                  & (along < along.max()-.15*length))
        samples = points[middle].astype(int)
        if not len(samples):
            continue
        radii = distances[y:y2,x:x2][samples[:,1], samples[:,0]]
        # Требуем, чтобы нижний квартиль радиусов был > .68*width —
        # штрих заметно толще основной линии.
        if np.percentile(radii,25) <= .68*stroke_width:
            continue
        center += (x,y)
        labels = []
        for letter_index, family, letter_center, height in letters:
            offset = letter_center-center
            side = float(offset@normal)
            # Буква должна быть сбоку от штриха (3w..22w), не слишком
            # далеко вдоль оси и размером .3..1 от длины штриха.
            if (abs(offset@axis) < 1.2*length and 3*stroke_width < abs(side) < 22*stroke_width
                    and .3*length < height < length):
                labels.append((letter_index, family, side, height))
        if labels:
            candidates.append((index,center,axis,normal,length,labels))
    removed = {}
    for position, (index,center,axis,normal,length,labels) in enumerate(candidates):
        for other,other_center,other_axis,_,other_length,other_labels in candidates[position+1:]:
            # Параллельность, соосность, зазор .5..6 длины, разброс длин <15%.
            if abs(axis@other_axis) < math.cos(math.radians(3)):
                continue
            if min(length,other_length)/max(length,other_length) < .85:
                continue
            offset = other_center-center
            mean_length = (length+other_length)/2
            gap = abs(offset@axis)-mean_length
            if abs(offset@normal) > .6*stroke_width or not .5*mean_length < gap < 6*mean_length:
                continue
            # Буквы должны быть «одинаковыми по семейству», с одной стороны
            # от оси и на близком расстоянии — иначе это два разных разреза.
            for letter,family,side,height in labels:
                for other_letter,other_family,other_side,other_height in other_labels:
                    if (letter != other_letter and family == other_family and side*other_side > 0
                            and min(height,other_height)/max(height,other_height) > .85
                            and abs(side-other_side) < .35*min(height,other_height)):
                        removed[index] = family
                        removed[other] = family
    return removed

def remove_section_marks(clean, distances, width):
    """Штрихи секущей плоскости толще основной линии и имеют длинное ядро.

    clean — рабочая маска в масштабе factor; distances — карта расстояний;
    width — толщина в том же масштабе.

    Возвращает (result, boxes):
      * result — clean без найденных штрихов;
      * boxes  — список удалённых прямоугольников для отчёта.
    """
    thick_core = (distances > width*.68).astype(np.uint8)
    n, labels, stats, _ = components(thick_core)
    remove = np.zeros_like(clean)
    boxes = []
    normal_axes = []
    # Собираем «обычные» оси — штрихи толщиной около width, они помогут
    # понять, соединяет ли найденный кандидат две основные линии.
    for horizontal in (True,False):
        for x,y,w,h in line_boxes(clean, max(40, round(12*width)), horizontal):
            thickness = h if horizontal else w
            if width*.7 <= thickness <= width*1.2:
                normal_axes.append((horizontal,x,y,w,h))
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        # Кандидат: длина 8w..45w, толщина < 4w.
        if width*8 < max(w,h) < width*45 and min(w,h) < width*4:
            # Сближенные торцевые грани могут выглядеть как один толстый
            # штрих. Если он соединяет две основные линии, это часть детали.
            anchors = [False,False]
            for horizontal,ax,ay,aw,ah in normal_axes:
                if h > w and horizontal and ax <= x+w/2 <= ax+aw:
                    anchors[0] |= abs(ay+ah/2-y) < 2.5*width
                    anchors[1] |= abs(ay+ah/2-y-h) < 2.5*width
                elif w > h and not horizontal and ay <= y+h/2 <= ay+ah:
                    anchors[0] |= abs(ax+aw/2-x) < 2.5*width
                    anchors[1] |= abs(ax+aw/2-x-w) < 2.5*width
            if all(anchors):
                continue
            remove[y:y+h,x:x+w][labels[y:y+h,x:x+w] == i] = 255
            boxes.append([x,y,x+w,y+h])
    if not boxes:
        return clean, boxes
    # Расширяем удаление на половину толщины — штрих не бывает «ядром»
    # без периферии.
    radius = math.ceil(width*.70)
    remove = cv2.dilate(remove, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    # Сохраняем основные прямые в местах, где отметка касается детали.
    for horizontal,x,y,w,h in normal_axes:
        remove[y:y+h,x:x+w] = 0
    result = clean.copy()
    result[remove > 0] = 0
    return result, boxes

def extract(gray, settings):
    """Главная функция: из grayscale-чертежа получить маску толстых линий.

    settings — словарь с необязательными ключами:
      * binarization : 'otsu' (по умолчанию) | 'adaptive';
      * line_width   : принудительная толщина (иначе estimate_width);
      * core_ratio   : порог ядра относительно толщины (по умолчанию .43);
      * min_length   : минимальная длина фрагмента;
      * exclude      : [[x1,y1,x2,y2], …] — ручные зоны к исключению.

    Возвращает (mask, details):
      * mask    — uint8 0/255, размер совпадает с gray;
      * details — словарь для report.json: threshold, line_width,
                  estimated_width, width_histogram, core_radius,
                  processing_scale, field, excluded_regions,
                  removed_glyphs, section_marks_removed,
                  paired_section_marks_removed, components, segments,
                  contour_pixels, fraction, warnings.

    Конвейер:
      1) бинаризация (Otsu или adaptive) — линии 255, фон 0;
      2) service_regions + ручные exclude — вычитаем служебные зоны;
      3) estimate_width — оцениваем толщину основной линии;
      4) при width < 5.5 апскейлим ×2 (distanceTransform иначе путает
         3 и 4 px — оба радиуса 2);
      5) remove_section_marks — убираем штрихи секущих плоскостей;
      6) ядро = distanceTransform >= core_ratio*width, дилатация диском
         восстановления — тонкие линии отсекаются, присоединённая
         штриховка не возвращается (реконструкция ограничена радиусом);
      7) для каждой компоненты: glyph_match, prune_branches, порог по
         min_length, фильтр «компактная с >2 концами», повторный glyph_match;
      8) scale_parentheses и paired_section_strokes — контекстные удаления;
      9) возврат маски к исходному размеру.
    """
    # Otsu хорошо подходит к этим чистым CAD-растрированиям; adaptive доступен
    # для фотографий/сканов с неравномерным фоном. Размытие не размывает грани.
    if settings.get("binarization", "otsu") == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 15)
        threshold = None
    else:
        threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    excluded, boxes, field = service_regions(binary)
    for box in settings.get("exclude", []):
        x1, y1, x2, y2 = map(int, box)
        h, w = gray.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        excluded[y1:y2, x1:x2] = 255
        boxes.append({"kind": "manual", "box": [x1, y1, x2, y2]})
    clean = binary.copy()
    clean[excluded > 0] = 0
    estimated, histogram = estimate_width(clean)
    width = float(settings.get("line_width", estimated))
    # У линий толщиной 3 и 4 px дискретный distanceTransform может дать
    # одинаковый радиус 2 px. Удвоение сетки разделяет эти два случая.
    factor = 2 if width < 5.5 else 1
    if factor > 1:
        clean = cv2.resize(clean, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
    work_width = width*factor
    radius = max(1.1, work_width*float(settings.get("core_ratio", .43)))
    distances = cv2.distanceTransform(clean, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    clean, section_boxes = remove_section_marks(clean, distances, work_width)
    if section_boxes:
        distances = cv2.distanceTransform(clean, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    core = (distances >= radius).astype(np.uint8)
    # Открытие диском удаляет тонкие линии. Восстановление ограничено радиусом:
    # полная геодезическая реконструкция вернула бы присоединённую штриховку.
    recovery = max(1, math.ceil(radius)-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (recovery*2+1, recovery*2+1))
    candidates = cv2.dilate(core, kernel) & (clean > 0).astype(np.uint8)
    n, labels, stats, _ = components(candidates)
    selected = np.zeros_like(candidates)
    kept = []
    removed_glyphs = []
    min_length = float(settings.get("min_length", max(width*12, min(gray.shape)*.018)))*factor
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area < min_length*work_width*.45 or max(w, h) < work_width*4:
            continue
        local = labels[y:y+h, x:x+w] == i
        glyph = glyph_match(local, work_width)
        if glyph:
            removed_glyphs.append({'box': [round(x/factor),round(y/factor),round((x+w)/factor),round((y+h)/factor)], 'glyph': glyph})
            continue
        skeleton = skeletonize(local)
        if skeleton.sum() < min_length:
            continue
        d = distances[y:y+h, x:x+w]
        skeleton = prune_branches(skeleton, d, work_width)
        if skeleton.sum() < min_length:
            continue
        # Толстые буквы/указатели обычно компактны. Круглые отверстия при этом
        # сохраняются: их скелет не имеет свободных концов.
        degrees = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3,3),np.uint8))
        endpoints = int(np.count_nonzero(skeleton & (degrees == 2)))
        if max(w,h) < work_width*13 and endpoints > 2:
            continue
        # Восстанавливаем толщину из скелета и пересекаем с локальной
        # маской — так границы не «расползаются» на соседние штрихи.
        rebuild_radius = max(1, math.ceil(work_width/2))
        rebuild_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2*rebuild_radius+1,2*rebuild_radius+1))
        reconstructed = cv2.dilate(skeleton.astype(np.uint8), rebuild_kernel)
        local_mask = reconstructed & local.astype(np.uint8)
        glyph = glyph_match(local_mask, work_width)
        if glyph:
            removed_glyphs.append({'box': [round(x/factor),round(y/factor),round((x+w)/factor),round((y+h)/factor)], 'glyph': glyph})
            continue
        selected[y:y+h, x:x+w] |= local_mask
        kept.append({"box": [round(x/factor), round(y/factor), round((x+w)/factor), round((y+h)/factor)],
                     "pixels": round(float(local_mask.sum())/(factor*factor)),
                     "length": round(float(skeleton.sum())/factor), "endpoints": endpoints})
    # Запись масштаба узнаётся также по цифрам и двоеточию между скобками,
    # независимо от наличия буквы вида в словаре образцов.
    paired_parentheses = scale_parentheses(selected, binary, kept, width, factor)
    paired_sections = paired_section_strokes(selected, distances, kept, removed_glyphs, width, factor)
    removed_section_pairs = []
    removed_parentheses = []
    # Скобки и знаки масштаба рядом с распознанным текстом. Узкие прямые
    # отдельно от текстовой строки этим правилом не удаляются.
    filtered = []
    for index, segment in enumerate(kept):
        x,y,x2,y2 = segment['box']
        if index in paired_sections:
            selected[round(y*factor):round(y2*factor),round(x*factor):round(x2*factor)] = 0
            removed_section_pairs.append({'box': segment['box'], 'label': paired_sections[index]})
            continue
        if index in paired_parentheses:
            selected[round(y*factor):round(y2*factor),round(x*factor):round(x2*factor)] = 0
            removed_parentheses.append({'box': segment['box'], 'glyph': paired_parentheses[index]})
            continue
        is_punctuation = False
        if x2-x < 5*width and 6*width < y2-y < 26*width:
            for glyph in removed_glyphs:
                gx,gy,gx2,gy2 = glyph['box']
                if abs((gy+gy2-y-y2)/2) < width*3 and min(abs(x2-gx),abs(gx2-x)) < width*25:
                    is_punctuation = True
                    break
        if is_punctuation:
            selected[round(y*factor):round(y2*factor),round(x*factor):round(x2*factor)] = 0
        else:
            filtered.append(segment)
    kept = filtered
    removed_glyphs.extend(removed_parentheses)
    mask = selected * 255
    if factor > 1:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    fraction = float(np.count_nonzero(mask) / mask.size)
    warnings = []
    if not kept:
        warnings.append("Контуры не найдены: проверьте толщину/качество исходника")
    if width < 3:
        warnings.append("Толстая линия имеет менее 3 px: разделение ненадёжно")
    if len(kept) > 80:
        warnings.append("Много фрагментов: возможны остатки текста или штриховки")
    return mask, {"threshold": threshold, "line_width": round(width, 2),
                  "estimated_width": round(estimated, 2), "width_histogram": histogram,
                  "core_radius": round(radius/factor, 2), "processing_scale": factor,
                  "field": field, "excluded_regions": boxes,
                  "removed_glyphs": removed_glyphs,
                  "section_marks_removed": len(section_boxes)+len(removed_section_pairs),
                  "paired_section_marks_removed": removed_section_pairs,
                  "components": len(kept), "segments": kept,
                  "contour_pixels": int(np.count_nonzero(mask)),
                  "fraction": round(fraction, 6), "warnings": warnings}

def read_pdf(path, dpi=PDF_DPI):
    """Первая страница PDF → BGR-массив (как cv2.imdecode).

    dpi=PDF_DPI=400 — компромисс: на 300 px тонкие линии могут стать 1-2 px
    и estimate_width сработает неуверенно; на 400 линии уже разделимы,
    а память ещё приемлема для A1.
    Рендерим только первую страницу: если в PDF несколько листов, обработать
    их все — задача вызывающего кода (сейчас main берёт одну запись на файл).
    """
    with fitz.open(path) as doc:
        if doc.page_count == 0:
            raise ValueError(f"PDF без страниц: {path}")
        zoom = dpi / 72.0
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                colorspace=fitz.csRGB, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def read_image(path):
    """Прочитать изображение (PDF или растровый формат) в BGR-массив.

    Для .pdf вызывает read_pdf; для остальных — cv2.imdecode через
    np.fromfile (это нужно, чтобы работали не-ASCII пути, которые
    cv2.imread на Windows не понимает).
    """
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return image

def write_image(path, image):
    """Записать изображение по расширению path (JPEG — quality 88, PNG — 6).

    Использует imencode + tofile, чтобы корректно обрабатывать не-ASCII
    пути. Бросает OSError, если кодирование не удалось.
    """
    options = [cv2.IMWRITE_JPEG_QUALITY, 88] if path.suffix.lower() == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, encoded = cv2.imencode(path.suffix, image, options)
    if not ok:
        raise OSError(f"Не удалось закодировать {path}")
    encoded.tofile(path)

def output_stem(relative):
    """Имя-основа для выходных файлов: путь + короткий хеш.

    Хеш sha256 от относительного пути (8 hex-символов) предотвращает
    коллизии одинаковых имён в разных подпапках. Слэши и прочие
    недопустимые в имени файла символы заменяются на «_». База
    обрезается до 150 символов, чтобы итоговый путь не превышал лимиты ФС.
    """
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', relative.with_suffix('').as_posix())
    digest = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:8]
    return base[:150] + '__' + digest

def make_preview(original, mask, max_side=1800):
    """Сделать превью: исходник приглушён, найденные линии — красным.

    Уменьшение с сохранением видимости тонких линий: маску перед ресайзом
    дилатируем на ceil(1/scale), иначе 1-2 px линии пропадут при
    интерполяции.
    """
    h, w = mask.shape
    scale = min(1., max_side / max(h, w))
    size = (max(1, round(w*scale)), max(1, round(h*scale)))
    base = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
    # Максимум перед уменьшением: тонкие линии остаются заметными на превью.
    r = max(1, math.ceil(1/scale))
    visible = cv2.dilate(mask, np.ones((r,r), np.uint8))
    visible = cv2.resize(visible, size, interpolation=cv2.INTER_NEAREST) > 0
    faded = np.clip(base.astype(np.float32)*.6 + 102, 0, 255).astype(np.uint8)
    faded[visible] = (40, 45, 230)
    return faded

def write_gallery(output, records):
    """Записать index.html — локальную галерею результатов.

    Только локальные файлы: галерея работает без сервера и без сети.
    Карточки ссылаются на *__contours.png, *__mask.png, *__overlay.jpg
    рядом с index.html. При наличии ошибки в record карточка показывает
    текст ошибки, а не картинки.
    """
    cards = []
    for rec in records:
        if rec.get("error"):
            cards.append(f'<article><h2>{html.escape(rec["source"])}</h2><p>{html.escape(rec["error"])}</p></article>')
            continue
        title = html.escape(rec['source'])
        stem = rec['output_stem']
        links = ' · '.join(f'<a href="{html.escape(stem+suffix, quote=True)}">{label}</a>' for suffix, label in
                           [('__contours.png', 'Контуры'), ('__mask.png', 'Маска'), ('__overlay.jpg', 'Наложение')])
        warning = html.escape('; '.join(rec['warnings']))
        cards.append(f'<article><h2>{title}</h2><p>{links}</p><p>Толщина: {rec["line_width"]} px; фрагментов: {rec["components"]}</p>'
                     f'<p class="warning">{warning}</p><a href="{html.escape(stem+"__overlay.jpg", quote=True)}">'
                     f'<img loading="lazy" src="{html.escape(stem+"__overlay.jpg", quote=True)}"></a></article>')
    page = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка контуров</title>'
    page += '<style>body{font:16px system-ui;margin:24px;background:#eee;color:#222}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:18px}article{background:white;padding:16px}h2{font-size:15px;overflow-wrap:anywhere}img{width:100%;height:580px;object-fit:contain}.warning{color:#a30}a{color:#1459a0}</style>'
    page += '<h1>Проверка выделения контуров</h1><p>Красным показаны выбранные линии. Контуры: чёрное на белом. Маска: белое на чёрном. Результат эвристический; сравните с исходником.</p><main>'
    page += ''.join(cards) + '</main></html>'
    (output/'index.html').write_text(page, encoding='utf-8')

def main(argv=None):
    """CLI: разобрать аргументы, обойти файлы, вызвать extract, записать отчёты.

    Возвращает 0, если ошибок не было, 1 — если хотя бы один файл упал
    (тогда в report.json у соответствующей записи есть поле "error").
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=ROOT/'исходники_растр', help='Файл или каталог; поиск рекурсивный')
    parser.add_argument('--output', type=Path, default=ROOT/'результаты_контуры', help='Плоский каталог результатов')
    parser.add_argument('--config', type=Path, help='JSON: defaults и files[относительный/путь.jpg] с поправками')
    parser.add_argument('--line-width', type=float, help='Принудительная толщина основной линии в пикселях')
    parser.add_argument('--core-ratio', type=float, default=.43, help='Порог ядра относительно толщины (по умолчанию 0.43)')
    parser.add_argument('--min-length', type=float, help='Минимальная длина фрагмента в пикселях')
    parser.add_argument('--binarization', choices=['otsu','adaptive'], default='otsu')
    parser.add_argument('--limit', type=int, help='Обработать первые N изображений')
    parser.add_argument('--match', default='', help='Подстрока в относительном пути для выборочной обработки')
    args = parser.parse_args(argv)
    if args.line_width is not None and args.line_width <= 0:
        parser.error('--line-width должен быть положительным')
    if not .15 <= args.core_ratio <= .60:
        parser.error('--core-ratio должен быть между 0.15 и 0.60')
    if args.min_length is not None and args.min_length <= 0:
        parser.error('--min-length должен быть положительным')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit должен быть положительным')
    source, output = args.input.resolve(), args.output.resolve()
    if not source.exists():
        parser.error(f'Исходный путь отсутствует: {source}')
    if output == source or (source.is_file() and output == source.parent):
        parser.error('Выберите отдельную папку результатов')
    # base — корень для relative: у файла это его папка, у каталога — он сам.
    base = source.parent if source.is_file() else source
    candidates = [source] if source.is_file() else source.rglob('*')
    # Отсекаем файлы не из EXTENSIONS, файлы внутри output (чтобы повторный
    # запуск не съел собственные результаты) и не подходящие под --match.
    files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS
                   and output not in p.parents and args.match.casefold() in p.relative_to(base).as_posix().casefold())
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error('Изображения не найдены')
    config = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    defaults = {'core_ratio': args.core_ratio, 'binarization': args.binarization}
    if args.line_width is not None:
        defaults['line_width'] = args.line_width
    if args.min_length is not None:
        defaults['min_length'] = args.min_length
    # CLI-значения перекрываются config['defaults'], а они — per-file
    # настройками из config['files'][relative]. Это стандартный приоритет
    # «файл важнее общих».
    defaults.update(config.get('defaults', {}))
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    records = []
    started = time.monotonic()
    for number, path in enumerate(files, 1):
        relative = path.relative_to(base)
        rec = {'source': relative.as_posix()}
        try:
            original = read_image(path)
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            settings = {**defaults, **config.get('files', {}).get(relative.as_posix(), {})}
            mask, details = extract(gray, settings)
            stem = output_stem(relative)
            write_image(output/(stem+'__mask.png'), mask)
            write_image(output/(stem+'__contours.png'), 255-mask)
            write_image(output/(stem+'__overlay.jpg'), make_preview(original, mask))
            rec.update(details, output_stem=stem, width=original.shape[1], height=original.shape[0])
            print(f'[{number:3}/{len(files)}] {relative}: {details["line_width"]:.1f} px, {details["components"]} фрагм.', flush=True)
        except (OSError, ValueError, cv2.error) as exc:
            rec['error'] = str(exc)
            print(f'[{number:3}/{len(files)}] ОШИБКА {relative}: {exc}', file=sys.stderr, flush=True)
        records.append(rec)

        overlays = output / "overlays"
        overlays.mkdir(parents=True, exist_ok=True)
        overlay_copies = 0
        for src in output.glob("*__overlay.jpg"):
            try:
                shutil.copy2(src, overlays / src.name)
                overlay_copies += 1
            except OSError as exc:
                print(f'Не удалось скопировать {src.name}: {exc}', file=sys.stderr, flush=True)

    report = {'input': str(source), 'output': str(output), 'processed': len(records),
              'errors': sum('error' in r for r in records), 'seconds': round(time.monotonic()-started, 1),
              'algorithm': 'adaptive stroke width + service regions + connected strokes', 'files': records}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output/'report.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['source','line_width','components','contour_pixels','warnings','error']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        for rec in records:
            writer.writerow({**rec, 'warnings': '; '.join(rec.get('warnings', []))})
    write_gallery(output, records)
    print(f'Готово: {len(records)-report["errors"]}/{len(records)}, {report["seconds"]} с. {output}', flush=True)
    return 1 if report['errors'] else 0

if __name__ == '__main__':
    raise SystemExit(main())