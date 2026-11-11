"""Проверки обновления пары NPZ/JSON; рабочие шаблоны не изменяются."""
import io
import cv2
import sys
import json
import tempfile
import unittest
import subprocess
import numpy as np
import extract_contours
from pathlib import Path
from unittest.mock import patch
import add_annotation_template as updater

class AddTemplateTests(unittest.TestCase):
    """Тесты add_annotation_template.add_template и load_collection.

    Каждый тест работает в собственном временном каталоге (tempfile).
    Боевые annotation_templates.npz / annotation_templates.sources.json
    не читаются и не модифицируются — только в одном тесте
    (test_updated_project_copy_is_usable_by_glyph_match) они читаются
    как эталон для проверки совместимости.
    """

    def setUp(self):
        """Подготовить временную коллекцию, маску-источник и снимок originals.

        * self.collection — каталог с пробелом в имени (проверяет
          корректность работы с не-ASCII/пробельными путями);
        * self.mask — F-образный глиф 46×90 px, гарантированно
          связный, пропорции 0.51 — в допустимом для glyph_match
          диапазоне (0.25, 1.1);
        * в NPZ — один старый образец "old", в JSON — согласованная
          с ним запись;
        * self.originals — снимок байтов обоих файлов, чтобы тесты
          могли проверять «ничего не изменилось».
        """
        temporary = tempfile.TemporaryDirectory(prefix="annotation_test_")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.collection = self.directory / "набор с пробелами"
        self.collection.mkdir()
        self.mask_path = self.directory / "чертёж__mask.png"
        self.box = [100, 200, 160, 300]
        self.mask = np.zeros((350, 220), np.uint8)
        # Связный F-образный символ: 46×90 px, без сглаживания.
        self.mask[205:295, 105:112] = 255
        self.mask[205:212, 105:151] = 255
        self.mask[245:252, 105:140] = 255
        self.write_mask(self.mask)
        # Старый образец — эллипс с толстой обводкой; главное, чтобы
        # это был валидный uint8 0/255 и он сохранялся в тестах.
        old = np.zeros((80, 50), np.uint8)
        cv2.ellipse(old, (25, 40), (20, 35), 0, 0, 360, 255, 5)
        np.savez_compressed(self.collection / updater.NPZ_NAME, old=old)
        self.write_sources([{"name": "old", "source": "старый.jpg", "box": [0, 0, 50, 80]}])
        self.originals = self.snapshot()

    def write_mask(self, mask):
        """Записать mask в self.mask_path как PNG (через imencode).

        imencode + tofile, а не cv2.imwrite — чтобы не-ASCII имена
        (в пути есть «чертёж») работали одинаково на всех ОС.
        """
        ok, encoded = cv2.imencode(".png", mask)
        self.assertTrue(ok)
        encoded.tofile(self.mask_path)

    def write_sources(self, sources):
        """Записать sources как JSON в self.collection/JSON_NAME."""
        (self.collection / updater.JSON_NAME).write_text(
            json.dumps(sources, ensure_ascii=False), encoding="utf-8")

    def snapshot(self):
        """Снять побайтовый снимок текущих NPZ и JSON."""
        return {name: (self.collection / name).read_bytes()
                for name in (updater.NPZ_NAME, updater.JSON_NAME)}

    def add(self, name="новый_символ", **kwargs):
        """Вызвать updater.add_template с путями по умолчанию этого теста."""
        return updater.add_template(self.mask_path, self.box, name,
                                    templates_dir=self.collection, **kwargs)

    def assert_unchanged(self):
        """Проверить, что NPZ и JSON не изменились с момента setUp."""
        self.assertEqual(self.snapshot(), self.originals)

    def assert_backup(self, result):
        """Проверить содержимое каталога резервной копии.

        Ожидаем:
          * побайтовые копии старых NPZ и JSON;
          * new_template_preview.png = инверсия вырезки (чёрный глиф
            на белом фоне) — чтобы было видно, что именно добавили;
          * change.json с тем же record, что вернул add_template.
        """
        backup = Path(result["backup"])
        for name, data in self.originals.items():
            self.assertEqual((backup / name).read_bytes(), data)
        preview = cv2.imdecode(np.fromfile(backup / "new_template_preview.png", np.uint8),
                               cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(preview, 255-self.mask[200:300, 100:160])
        manifest = json.loads((backup / "change.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["record"], result["record"])

    def test_add_preserves_old_templates_and_byte_exact_backup(self):
        """Добавление сохраняет старый образец и делает точный бэкап."""
        result = self.add()
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["operation"], "add")
        current = self.snapshot()
        templates, sources = updater.load_collection(current[updater.NPZ_NAME], current[updater.JSON_NAME])
        with np.load(io.BytesIO(self.originals[updater.NPZ_NAME]), allow_pickle=False) as old:
            np.testing.assert_array_equal(templates["old"], old["old"])
        np.testing.assert_array_equal(templates["новый_символ"], self.mask[200:300, 100:160])
        self.assertEqual(sources[1]["box"], self.box)
        # Без --source и без report.json источником считается сама маска.
        self.assertEqual(sources[1]["source_kind"], "mask")
        self.assertEqual(sources[1]["source"], str(self.mask_path))
        self.assert_backup(result)

    def test_report_supplies_drawing_and_width(self):
        """report.json рядом с маской даёт source и line_width (source_kind='drawing')."""
        report = {"files": [{"source": "подпапка/чертёж.jpg", "output_stem": "чертёж",
                             "width": 220, "height": 350, "line_width": 7}]}
        (self.directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
        result = self.add()
        self.assertEqual(result["record"]["source"], "подпапка/чертёж.jpg")
        self.assertEqual(result["record"]["source_kind"], "drawing")
        self.assertEqual(result["record"]["line_width"], 7)
        # line_width=7, h=90 → 6*7=42 < 90 < 175=25*7 — попадает в диапазон,
        # никаких «толщина неизвестна» быть не должно.
        self.assertEqual(result["notes"], [])

    def test_duplicate_requires_explicit_replace(self):
        """Существующее имя без --replace — ValueError; с --replace — замена."""
        with self.assertRaisesRegex(ValueError, "--replace"):
            self.add(name="old")
        self.assert_unchanged()
        self.assertFalse((self.collection / "резервные_шаблоны").exists())
        # Заменяем старую запись "old"; count остаётся 1, operation=replace.
        result = self.add(name="old", replace=True, source="новый.jpg", line_width=7)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["operation"], "replace")
        self.assertEqual(result["record"]["source"], "новый.jpg")
        self.assert_backup(result)

    def test_dry_run_does_not_change_collection_or_create_backup(self):
        """--dry-run не пишет ни NPZ/JSON, ни бэкап, но возвращает count."""
        result = self.add(dry_run=True)
        self.assertEqual(result["count"], 2)
        self.assertIsNone(result["backup"])
        self.assert_unchanged()
        self.assertFalse((self.collection / "резервные_шаблоны").exists())

    def test_invert_stores_white_symbol(self):
        """--invert превращает чёрный символ на белом в белый на чёрном в NPZ."""
        self.write_mask(255-self.mask)
        result = self.add(invert=True)
        with np.load(self.collection / updater.NPZ_NAME, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["новый_символ"], self.mask[200:300, 100:160])
        self.assertEqual(result["record"]["input_foreground"], "black")

    def test_invalid_inputs_leave_collection_unchanged(self):
        """Все негативные входные данные → ValueError, набор не изменён.

        Кейсы:
          0..3 — некорректный box (вне маски / x1>=x2 / y1>=y2 / пустая
                 вырезка в верхнем левом углу);
          4..5 — некорректное имя (выход из папки, зарезервированное
                 для numpy);
          6..7 — некорректный line_width (несовместим с высотой,
                 NaN);
          8    — разрывной компонент в вырезке;
          9    — небинарная маска (пиксель 127).
        """
        # disconnected содержит «оторванный» пиксель — компонент не один.
        disconnected = self.mask.copy()
        disconnected[280:283, 150:153] = 255
        # nonbinary содержит пиксель 127 — нарушает контракт 0/255.
        nonbinary = self.mask.copy()
        nonbinary[250, 125] = 127
        variants = [
            (self.mask, [-1, 200, 160, 300], "glyph", {}),
            (self.mask, [100, 200, 160, 400], "glyph", {}),
            (self.mask, [100, 200, 100, 300], "glyph", {}),
            (self.mask, [0, 0, 50, 50], "glyph", {}),
            (self.mask, self.box, "../glyph", {}),
            (self.mask, self.box, "allow_pickle", {}),
            (self.mask, self.box, "glyph", {"line_width": 20}),
            (self.mask, self.box, "glyph", {"line_width": float("nan")}),
            (disconnected, self.box, "glyph", {}),
            (nonbinary, self.box, "glyph", {}),
        ]
        for index, (mask, box, name, kwargs) in enumerate(variants):
            with self.subTest(case=index):
                self.write_mask(mask)
                with self.assertRaises(ValueError):
                    updater.add_template(self.mask_path, box, name,
                                         templates_dir=self.collection, **kwargs)
                self.assert_unchanged()

    def test_mismatched_collection_is_rejected(self):
        """NPZ и JSON с разными ключами → ValueError, ничего не изменено.

        В NPZ есть "old", а JSON пустой. load_collection ловит это
        через set(names) != set(templates).
        """
        self.write_sources([])
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "Ключи NPZ и JSON"):
            self.add()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.collection / "резервные_шаблоны").exists())

    def test_npy_instead_of_npz_is_rejected(self):
        """Одиночный .npy вместо .npz → ValueError «архивом NPZ»."""
        buffer = io.BytesIO()
        np.save(buffer, np.zeros((2, 2), np.uint8))
        with self.assertRaisesRegex(ValueError, "архивом NPZ"):
            updater.load_collection(buffer.getvalue(), self.originals[updater.JSON_NAME])

    def test_second_file_failure_rolls_back_first_file(self):
        """Сбой os.replace на JSON откатывает уже установленный NPZ.

        Подменяем updater.os.replace так, чтобы первый вызов (для NPZ)
        прошёл, а второй (для JSON) упал. install_pair должен
        восстановить NPZ из originals и не оставить .tmp-файлов.
        Бэкап при этом создан и содержит старые файлы.
        """
        actual_replace = updater.os.replace
        failed = False

        def fail_once(source, target):
            nonlocal failed
            if Path(target).name == updater.JSON_NAME and not failed:
                failed = True
                raise OSError("Смоделированная ошибка записи JSON")
            return actual_replace(source, target)

        with patch.object(updater.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "прежний набор сохранён"):
                self.add()
        self.assertTrue(failed)
        self.assert_unchanged()
        self.assertEqual(list(self.collection.glob("*.tmp")), [])
        backups = list((self.collection / "резервные_шаблоны").iterdir())
        self.assertEqual(len(backups), 1)
        for name, data in self.originals.items():
            self.assertEqual((backups[0] / name).read_bytes(), data)

    def test_backup_failure_leaves_collection_unchanged(self):
        """Если backup_dir — это файл, а не папка, набор не меняется."""
        blocked = self.directory / "это_файл"
        blocked.write_text("не папка", encoding="utf-8")
        with self.assertRaises(OSError):
            self.add(backup_dir=blocked)
        self.assert_unchanged()

    def test_concurrent_update_is_rejected(self):
        """Второй процесс с flock видит занятый lock → ValueError."""
        with updater.collection_lock(self.collection):
            with self.assertRaisesRegex(ValueError, "Другой процесс"):
                self.add()
        self.assert_unchanged()

    def test_cli_from_other_directory(self):
        """CLI работает из другой cwd и печатает ожидаемые строки."""
        completed = subprocess.run(
            [sys.executable, str(updater.ROOT / "add_annotation_template.py"),
             "--mask", str(self.mask_path), "--box", *map(str, self.box),
             "--name", "F_variant1", "--templates-dir", str(self.collection)],
            cwd=self.directory, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Добавлен образец: F_variant1", completed.stdout)
        self.assertIn("Резервная копия обоих файлов", completed.stdout)
        with np.load(self.collection / updater.NPZ_NAME, allow_pickle=False) as archive:
            self.assertIn("F_variant1", archive.files)

    def test_updated_project_copy_is_usable_by_glyph_match(self):
        """Сквозной тест: добавленный образец ловится extract_contours.glyph_match.

        Требует наличия боевых annotation_templates.npz и
        annotation_templates.sources.json рядом с updater.ROOT — они
        служат эталоном совместимости. Если их нет, тест упадёт: это
        намеренно, потому что проверяется именно совместимость с
        реальным набором, а не с синтетическим.
        """
        for filename in (updater.NPZ_NAME, updater.JSON_NAME):
            (self.collection / filename).write_bytes((updater.ROOT / filename).read_bytes())
        with np.load(self.collection / updater.NPZ_NAME, allow_pickle=False) as archive:
            old_count = len(archive.files)
        result = self.add(name="F_test_variant", line_width=7)
        self.assertEqual(result["count"], old_count+1)
        # annotation_templates кэширован через lru_cache — сбрасываем,
        # иначе glyph_match увидит старый набор без F_test_variant.
        extract_contours.annotation_templates.cache_clear()
        try:
            # Подменяем ROOT extract_contours на наш временный каталог,
            # чтобы annotation_templates() читал NPZ из self.collection.
            with patch.object(extract_contours, "ROOT", self.collection):
                matched = extract_contours.glyph_match(self.mask[200:300, 100:160] > 0, 7)
                self.assertEqual(matched, "F_test_variant")
        finally:
            # Возвращаем кэш в исходное состояние — иначе другие тесты
            # (в том же процессе) увидят чужой ROOT.
            extract_contours.annotation_templates.cache_clear()

if __name__ == "__main__":
    unittest.main()