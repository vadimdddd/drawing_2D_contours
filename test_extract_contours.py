"""Проверки важных свойств результата, без требования замкнутого контура."""
import cv2
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from extract_contours import ROOT, extract, output_stem, read_image, paired_section_strokes

# Реальные чертежи, на которых проверяются конкретные правила. Часть тестов
# помечена @unittest.skipUnless(...): на машине без этих файлов они
# пропускаются, а не падают. Это позволяет запускать набор и без полного
# корпуса исходников.
DRAWING_290 = ROOT/'исходники_растр/290_stage/К7-ФГС-200.00.007 - Палец шнека.jpg'
DRAWING_254 = ROOT/'исходники_растр/254_stage/ГЕУС.387527.003.20_02 - Вал.jpg'
DRAWING_255 = ROOT/'исходники_растр/255_stage/ГЕУС.387527.003.20_03 - Корпус подшипника.jpg'

class ContourTests(unittest.TestCase):
    """Тесты extract_contours: свойства результата и отдельные правила.

    Часть тестов синтетическая (рисуем картинку и прогоняем extract
    с line_width=7), часть — на реальных чертежах из ROOT/исходники_растр.
    Реальные чертежи жёстко привязаны к координатам: при смене PDF_DPI,
    line_width по умолчанию или обновлении исходников координаты могут
    «поплыть» — это ожидаемая плата за проверку на боевых данных.
    """

    @unittest.skipUnless(DRAWING_255.is_file(), 'Нет исходного чертежа 255_stage')
    def test_only_two_section_strokes_removed_on_drawing_255(self):
        """На 255_stage правило парных штрихов удаляет ровно два штриха «Ge».

        Сравниваем два прогона extract на одном и том же изображении:
          * before — с отключённым paired_section_strokes (patch → {});
          * after  — с правилом.
        Разница должна быть только в двух боксах — и внутри них before
        содержит >1000 пикселей (то есть штрихи реально были), after — 0.
        Вне боксов результаты должны совпадать побайтово.
        """
        gray = cv2.cvtColor(read_image(DRAWING_255), cv2.COLOR_BGR2GRAY)
        with patch('extract_contours.paired_section_strokes', return_value={}):
            before, _ = extract(gray, {})
        after, details = extract(gray, {})
        boxes = [[2126,2718,2269,2784], [1758,2871,1901,2937]]
        allowed = np.zeros(gray.shape, bool)
        for x,y,x2,y2 in boxes:
            allowed[y:y2,x:x2] = True
            self.assertGreater(np.count_nonzero(before[y:y2,x:x2]), 1000)
            self.assertEqual(np.count_nonzero(after[y:y2,x:x2]), 0)
        np.testing.assert_array_equal(after[~allowed], before[~allowed])
        self.assertEqual(details['paired_section_marks_removed'],
                         [{'box': box, 'label': 'Ge'} for box in boxes])

    @staticmethod
    def section_pair_scene(angle=-22.5, factor=1, thickness=13, count=2, shift=0):
        """Синтетическая сцена «пара штрихов + буквы Ge» для paired_section_strokes.

        Рисуем count коротких толстых штрихов вдоль оси angle (градусы),
        каждый с буквой «Ge» сбоку. Возвращает ровно тот кортеж, который
        ждёт paired_section_strokes:
            (selected, distances, segments, glyphs, width, factor)

        Параметры:
          * angle    — угол наклона общей оси штрихов, градусы;
          * factor   — масштаб сетки (factor=2 имитирует апскейл ×2 из
                       extract при width < 5.5);
          * thickness — толщина штрихов в binary (влияет на distances);
          * count    — сколько штрихов нарисовать (1 — одиночный);
          * shift    — сдвиг второго штриха по нормали (ненулевой делает
                       пару несоосной — правило не должно срабатывать).

        Внутренние толщины (7*factor для selected, thickness*factor для
        binary, width=8) подобраны так, чтобы сцена проходила пороги
        paired_section_strokes (.68*width по радиусам, 8..40*width по
        длине, .5..6*mean_length по зазору между штрихами). Конкретные
        числа важны для прохождения теста — это не «просто константы».
        """
        axis = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        normal = np.array([-axis[1], axis[0]])
        selected = np.zeros((900*factor,900*factor), np.uint8)
        binary = np.zeros_like(selected)
        segments, glyphs = [], []
        for index in range(count):
            center = np.array([400.,450.])+index*(350*axis+shift*normal)
            a, b = [tuple(np.rint((center+sign*70*axis)*factor).astype(int)) for sign in (-1,1)]
            # selected — то, что extract уже «выбрал» (толщина 7*factor);
            # binary — исходный чертёж (толщина thickness*factor), из него
            # считается distanceTransform.
            stroke = np.zeros_like(selected)
            cv2.line(stroke, a, b, 1, 7*factor)
            cv2.line(binary, a, b, 255, thickness*factor)
            selected |= stroke
            x,y,w,h = cv2.boundingRect(cv2.findNonZero(stroke))
            segments.append({'box': [x/factor,y/factor,(x+w)/factor,(y+h)/factor], 'endpoints': 2})
            # Буква «Ge» сбоку от штриха: с одной стороны у первого,
            # с другой у второго — так они «на одной стороне от оси»,
            # как требует правило.
            letter = center+(1 if index else -1)*90*axis+110*normal
            gx,gy = letter
            glyphs.append({'box': [gx-20,gy-40,gx+20,gy+40], 'glyph': 'Ge'})
        distances = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return selected, distances, segments, glyphs, 8, factor

    def test_section_pair_recognized_at_different_angles_and_scale(self):
        """Правило парных штрихов работает при любых углах и factor=2.

        Прогоняем синтетическую сцену под несколькими углами (включая
        горизонталь, вертикаль и наклон) и с factor=2 (имитация апскейла).
        Во всех случаях ожидаем {0: 'Ge', 1: 'Ge'} — оба штриха удаляются.
        """
        for angle, factor in [(-22.5,1), (0,1), (35,1), (90,1), (-22.5,2)]:
            with self.subTest(angle=angle, factor=factor):
                scene = self.section_pair_scene(angle=angle, factor=factor)
                self.assertEqual(paired_section_strokes(*scene), {0:'Ge', 1:'Ge'})

    def test_open_strokes_preserved_without_full_section_context(self):
        """Одиночные/несоосные/безбуквенные штрихи правило не удаляет.

        Пять негативных сцен:
          * обычная толщина (7) — даже рядом с буквами штрих не «толстый»;
          * одиночный толстый штрих — пары нет;
          * параллельные несоосные (shift=20) — сдвиг по нормали слишком велик;
          * нет букв — нет контекста;
          * разные буквы (Ge + A) — семейства не совпадают.
        Для каждой ожидаем {}, то есть ничего не удалено.
        """
        scenes = {
            'обычная толщина даже рядом с буквами': self.section_pair_scene(thickness=7),
            'одинокий толстый штрих': self.section_pair_scene(count=1),
            'параллельные несоосные штрихи': self.section_pair_scene(shift=20),
            'нет букв': self.section_pair_scene(),
            'разные буквы': self.section_pair_scene(),
        }
        # Каждый self.section_pair_scene() возвращает независимые объекты,
        # так что мутации ниже не задевают соседние сцены.
        scenes['нет букв'][3].clear()
        scenes['разные буквы'][3][1]['glyph'] = 'A'
        for description, scene in scenes.items():
            with self.subTest(description=description):
                self.assertEqual(paired_section_strokes(*scene), {})

    @unittest.skipUnless(DRAWING_254.is_file(), 'Нет исходного чертежа 254_stage')
    def test_scale_parentheses_removed_on_drawing_254(self):
        """На 254_stage скобки «(1:1)» вычищены, но основная деталь на месте.

        Проверяем две узкие скобки по координатам — обе должны быть пусты
        (mask=0). И проверяем два контрольных блока с большим числом
        пикселей: удаление скобок не должно вычищать главный вид и
        нижний блок разрезов.
        """
        image = read_image(DRAWING_254)
        mask, _ = extract(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), {})
        for x,y,x2,y2 in [(1878,2732,1893,2829), (1996,2732,2011,2829)]:
            with self.subTest(box=(x,y,x2,y2)):
                self.assertEqual(int(mask[y:y2,x:x2].sum()), 0)
        self.assertGreater(np.count_nonzero(mask[2850:3330,1610:2100]), 25000)
        self.assertGreater(np.count_nonzero(mask[600:1080,530:3900]), 50000)

    def test_parenthesis_shaped_arcs_without_scale_are_kept(self):
        """Дуги-«скобки» без цифр/двоеточия рядом не удаляются.

        Рисуем две дуги эллипса, по форме похожие на «(» и «)», но без
        текста между ними. scale_parentheses не должна их трогать
        (has_scale_text вернёт False), поэтому в контрольных точках
        маска остаётся 255.
        """
        image = np.full((900,1100),255,np.uint8)
        cv2.ellipse(image,(400,380),(80,100),0,150,210,0,7)
        cv2.ellipse(image,(490,380),(80,100),0,330,390,0,7)
        mask, _ = extract(image, {'line_width':7})
        self.assertEqual(mask[380,320],255)
        self.assertEqual(mask[380,570],255)

    @unittest.skipUnless(DRAWING_290.is_file(), 'Нет исходного чертежа 290_stage')
    def test_large_view_letters_removed_on_drawing_290(self):
        """На 290_stage буквы видов удаляются, но главный вид и разрезы целы.

        Верхняя «В» — это образец из annotation_templates.npz; нижняя —
        другая буква на том же листе, должна ловиться тем же правилом
        (glyph_match). Проверяем, что обе пусты, и что при этом не
        вычищены контрольные блоки с деталью.
        """
        image = read_image(DRAWING_290)
        mask, _ = extract(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), {})
        # Верхняя «В» служит образцом; нижняя — другая буква на том же листе.
        for x,y,x2,y2 in [(1569,359,1629,477), (812,1981,872,2099)]:
            with self.subTest(box=(x,y,x2,y2)):
                self.assertEqual(int(mask[y:y2,x:x2].sum()), 0)
        # Удаление текста не должно превращать результат в пустую маску
        # или убирать главный вид и нижние разрезы детали.
        self.assertGreater(np.count_nonzero(mask[690:1365,460:2085]), 50000)
        self.assertGreater(np.count_nonzero(mask[2170:2610,900:2170]), 25000)

    def test_open_stroke_kept_thin_crossing_removed(self):
        """Толстые открытые линии сохранены, тонкое пересечение удалено.

        Сцена: Г-образная толстая линия (7 px) + тонкая (1 px) горизонталь
        поперёк. Ожидаем:
          * горизонтальный и вертикальный сегменты Г — в маске;
          * тонкая горизонталь — вычищена;
          * ниже Г-конца (mask[520]) — пусто (то есть контур не замкнут);
          * внутри угла — пусто (нет заливки).
        """
        image = np.full((800, 1000), 255, np.uint8)
        cv2.line(image, (250, 300), (650, 300), 0, 7)
        cv2.line(image, (650, 300), (650, 520), 0, 7)
        cv2.line(image, (180, 410), (760, 410), 0, 1)
        mask, _ = extract(image, {'line_width': 7})
        self.assertEqual(mask.shape, image.shape)
        self.assertTrue(np.all(mask[300, 260:640] == 255))
        self.assertTrue(np.all(mask[310:510, 650] == 255))
        self.assertEqual(int(mask[410, 250:600].sum()), 0)
        self.assertEqual(int(mask[520, 250:600].sum()), 0)  # разрыв не замкнут
        self.assertEqual(int(mask[350:390, 300:500].sum()), 0)  # заливки нет

    def test_hole_and_partial_arc_survive(self):
        """Круглое отверстие (замкнутый скелет) выживает, открытая дуга — нет.

        Круг: его скелет — кольцо без свободных концов, поэтому фильтр
        «компактное с >2 концами» его не режет. Дуга 30°..280° — открытая,
        её скелет имеет два свободных конца, и она отбрасывается.
        Проверяем по точкам: слева/справа от круга — 255, по краям
        дуги — 0.
        """
        image = np.full((800, 1000), 255, np.uint8)
        cv2.circle(image, (300, 350), 24, 0, 7)
        cv2.ellipse(image, (600, 350), (90,90), 0, 30, 280, 0, 7)
        mask, _ = extract(image, {'line_width': 7})
        self.assertEqual(mask[350, 324], 255)
        self.assertEqual(mask[350, 300], 0)
        self.assertEqual(mask[350, 510], 255)
        self.assertEqual(mask[350, 690], 0)  # открытая дуга

    def test_small_table_removed_but_part_kept(self):
        """Таблица с тонкими внутренними линиями удаляется целиком, деталь — нет.

        Таблица: рамка (200,200)–(400,330) толщиной 7 + сетка 4×3:
          * горизонтали на y=450..600, крайние толщиной 7, средние 1;
          * вертикали на x=200,400,600, крайние 7, средняя 1.
        service_regions должна распознать это как table (≥3 горизонталей
        с одинаковыми границами + вертикальные бордюры слева/справа/внутри)
        и вычистить. Прямоугольник-«деталь» выше таблицы — остаться.
        """
        image = np.full((800, 1000), 255, np.uint8)
        cv2.rectangle(image, (200,200), (400,330), 0, 7)
        for y in (450,500,550,600):
            cv2.line(image, (200,y), (600,y), 0, 7 if y in (450,600) else 1)
        for x in (200,400,600):
            cv2.line(image, (x,450), (x,600), 0, 7 if x in (200,600) else 1)
        mask, _ = extract(image, {'line_width': 7})
        self.assertEqual(mask[200, 300], 255)
        self.assertEqual(int(mask[440:610,190:610].sum()), 0)

    def test_same_filename_different_directories(self):
        """output_stem различает одинаковые имена в разных папках и расширениях.

        Хеш от относительного пути должен давать разные stem для:
          * a/деталь.jpg и b/деталь.jpg (разные папки);
          * a/деталь.jpg и a/деталь.png (разные расширения).
        И не должен содержать «/» — stem используется как имя файла.
        """
        self.assertNotEqual(output_stem(Path('a/деталь.jpg')), output_stem(Path('b/деталь.jpg')))
        self.assertNotEqual(output_stem(Path('a/деталь.jpg')), output_stem(Path('a/деталь.png')))
        self.assertNotIn('/', output_stem(Path('a/деталь.jpg')))

    def test_close_end_faces_not_section_marks(self):
        """Сближенные торцевые грани не принимаются за штрихи разрезов.

        Сцена: прямоугольник (250,300)–(650,400) толщиной 7 и две
        вертикали в 5 px от его левой/правой граней. Из-за наложения
        «вертикаль + край» дают толстый блок ~12 px — remove_section_marks
        мог бы принять его за штрих разреза. Но блок соединяет две
        нормальные горизонтали (верх и низ прямоугольника), поэтому
        anchors[0] и anchors[1] в remove_section_marks обе True →
        штрих сохраняется. Проверяем, что левый и правый края на
        y=350 остались в маске.
        """
        image = np.full((800, 1000), 255, np.uint8)
        cv2.rectangle(image, (250,300), (650,400), 0, 7)
        cv2.line(image, (255,300), (255,400), 0, 7)
        cv2.line(image, (645,300), (645,400), 0, 7)
        mask, _ = extract(image, {'line_width': 7})
        self.assertEqual(mask[350,250], 255)
        self.assertEqual(mask[350,650], 255)

if __name__ == '__main__':
    unittest.main()