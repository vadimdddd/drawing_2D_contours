#!/usr/bin/env python3
"""Добавить образец в annotation_templates.npz и JSON с резервной копией.

Пример: .env/bin/python add_annotation_template.py \
    --mask рисунок__mask.png --box 100 200 160 300 --name Zhe_variant1

Координаты относятся к маске исходного размера; x2 и y2 не включаются.
По умолчанию символ белый (255), фон чёрный (0). Для обратной полярности: --invert.
"""

import os
import re
import io
import cv2
import sys
import json
import math
import fcntl
import zipfile
import argparse
import tempfile
import numpy as np
from pathlib import Path
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
NPZ_NAME = "annotation_templates.npz"
JSON_NAME = "annotation_templates.sources.json"

def validate_array(array, label):
    """Проверить, что array — непустой 2D uint8 только из 0 и 255.

    label попадает в текст ошибки, чтобы было понятно, какой именно
    образец (или вырезка) не прошёл проверку. Функция используется и при
    чтении существующего набора, и при подготовке новой вырезки — так
    гарантируется, что «то, что пишем» совпадает с «тем, что умеем читать».
    """
    if (array.dtype != np.uint8 or array.ndim != 2 or not array.size
            or not np.all((array == 0) | (array == 255)) or not np.any(array)):
        raise ValueError(f"{label}: ожидается непустой двумерный uint8 с пикселями 0/255")

def load_collection(npz_bytes, json_bytes):
    """Разобрать и провалидировать пару NPZ+JSON из байтов.

    Возвращает (templates, sources):
      * templates — {name: uint8 ndarray}, копии массивов из NPZ;
      * sources   — список записей из JSON (порядок сохраняется).

    Проверки:
      * NPZ — именно архив (np.lib.npyio.NpzFile), без повторяющихся имён;
      * каждый массив проходит validate_array;
      * JSON — список словарей с непустым строковым name и корректным box
        (четыре int, 0 <= x1 < x2, 0 <= y1 < y2);
      * множества имён в NPZ и в JSON совпадают, дубликатов в JSON нет.

    Функция вызывается дважды в add_template: на текущих файлах и на
    подготовленных replacements до записи. Так «формат записи» и
    «формат чтения» не могут разойтись.
    """
    archive = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Файл шаблонов должен быть архивом NPZ")
    with archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("В NPZ есть повторяющиеся имена")
        templates = {name: archive[name].copy() for name in archive.files}
    for name, array in templates.items():
        validate_array(array, f"Образец {name}")
    sources = json.loads(json_bytes.decode("utf-8"))
    if not isinstance(sources, list):
        raise ValueError("JSON должен содержать список записей об источниках")
    names = []
    for record in sources:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("В JSON есть запись без строкового name")
        box = record.get("box")
        if (not isinstance(record.get("source"), str) or not record["source"]
                or not isinstance(box, list) or len(box) != 4
                or any(type(v) is not int for v in box)
                or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])):
            raise ValueError(f"Некорректные source/box у записи {record['name']}")
        names.append(record["name"])
    if len(names) != len(set(names)) or set(names) != set(templates):
        raise ValueError("Ключи NPZ и JSON не совпадают или в JSON есть повторяющиеся имена")
    return templates, sources

def report_context(mask_path, shape, invert):
    """Попытаться вытащить исходную запись о чертеже из соседнего report.json.

    Отчёт extract_contours.py лежит рядом с масками и содержит для каждого
    обработанного файла поле source и line_width. Если маска называется
    <stem>__mask.png (или <stem>__contours.png при invert), находим
    соответствующую запись и возвращаем (record, []).

    Возвращает (None, [пояснение]) во всех «мягких» случаях: файла нет,
    маска не упомянута в отчёте, размер не совпадает, обработка в отчёте
    завершилась ошибкой, поле source пустое. Такие случаи не фатальны —
    пользователь всё равно может указать --source/--line-width вручную.
    """
    path = mask_path.parent / "report.json"
    if not path.is_file():
        return None, []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        records = report["files"]
        if not isinstance(records, list):
            raise ValueError("files не является списком")
        # При invert входной файл — это __contours.png (инверсия маски),
        # поэтому ищем маску по обоим возможным суффиксам.
        suffixes = ["__mask.png"] + (["__contours.png"] if invert else [])
        matches = [r for r in records if isinstance(r, dict)
                   and any(mask_path.name == str(r.get("output_stem", ""))+s for s in suffixes)]
        if not matches:
            return None, []
        if len(matches) != 1:
            raise ValueError("для маски найдено несколько записей")
        record = matches[0]
        if "error" in record or (record.get("height"), record.get("width")) != shape:
            raise ValueError("размер маски не совпадает с отчётом или обработка завершилась ошибкой")
        if not isinstance(record.get("source"), str) or not record["source"]:
            raise ValueError("нет source")
        return record, []
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return None, [f"Не использован {path}: {exc}"]

def prepare_sample(mask_path, box, name, *, invert=False, source=None, line_width=None):
    """Прочитать вырезку-образец и построить запись для JSON.

    Возвращает (sample, record, notes):
      * sample — uint8 ndarray 0/255, готовая вырезка (после возможного invert);
      * record — словарь для annotation_templates.sources.json;
      * notes  — список нефатальных предупреждений (например, «толщина
                 неизвестна, проверка высоты не выполнена»).

    Все проверки, при которых продолжать нельзя, бросают ValueError:
      * имя: \\w+ и не зарезервировано под ключи np.save/np.load;
      * line_width: конечное положительное число;
      * маска: читается, 2D uint8 0/255 (серый RGB-склейка допускается);
      * box: четыре int, 0 <= x1 < x2, 0 <= y1 < y2, в пределах маски;
      * вырезка: непустая, после validate_array — один связный компонент,
        пропорции 0.25 < w/h < 1.1 (как в glyph_match);
      * если известна толщина линии, высота глифа должна попадать
        в (6*width, 25*width) — ровно тот же диапазон, что в glyph_match.
    """
    if not re.fullmatch(r"\w+", name) or name in {"file", "allow_pickle"}:
        # "file" и "allow_pickle" — зарезервированные ключи np.save/np.load,
        # np.savez_compressed на них откажется писать архив.
        raise ValueError("Имя: буквы, цифры и подчёркивания; file и allow_pickle зарезервированы")
    if line_width is not None and (not math.isfinite(line_width) or line_width <= 0):
        raise ValueError("--line-width должен быть конечным положительным числом")
    mask_path = mask_path.resolve()
    raw = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Не удалось прочитать маску: {mask_path}")
    # Grayscale, сохранённый как RGB (все три канала одинаковы), сводим к 2D.
    if raw.ndim == 3 and raw.shape[2] == 3 and np.all(raw == raw[:, :, :1]):
        raw = raw[:, :, 0]
    if raw.dtype != np.uint8 or raw.ndim != 2 or not np.all((raw == 0) | (raw == 255)):
        raise ValueError("Нужна бинарная 8-битная маска 0/255, без цвета и прозрачности; используйте PNG")
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("--box: четыре целых числа x1 y1 x2 y2")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= raw.shape[1] and 0 <= y1 < y2 <= raw.shape[0]):
        raise ValueError(f"Координаты вне маски {raw.shape[1]}×{raw.shape[0]} или задают пустую область")
    sample = raw[y1:y2, x1:x2].copy()
    if invert:
        sample = 255-sample
    validate_array(sample, "Вырезка")
    if not np.any(sample == 0):
        # validate_array гарантирует, что в вырезке есть хоть один пиксель;
        # здесь проверяем, что есть и фон (иначе вырезка целиком белая).
        raise ValueError("Вырезка полностью белая: проверьте координаты и полярность (--invert)")
    # connectedComponentsWithStats считает и фон; нам нужен ровно один глиф,
    # т.е. count == 2 (0 — фон, 1 — единственный объект).
    count, _ = cv2.connectedComponents((sample > 0).astype(np.uint8), connectivity=8)
    if count != 2:
        raise ValueError(f"Нужен один связный фрагмент; в вырезке найдено {count-1}")
    _, _, w, h = cv2.boundingRect(cv2.findNonZero(sample))
    if not .25 < w/h < 1.1:
        raise ValueError(f"Пропорции {w}/{h} не подходят glyph_match: требуется 0.25 < ширина/высота < 1.1")

    context, notes = report_context(mask_path, raw.shape, invert)
    if line_width is None and context is not None:
        value = context.get("line_width")
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            line_width = float(value)
    if line_width is not None and not 6*line_width < h < 25*line_width:
        raise ValueError(f"Высота {h} px не подходит glyph_match при толщине {line_width:g} px: требуется от 6 до 25 толщин, строго")
    if line_width is None:
        # Толщину можно не знать: тогда высоту относительно неё проверить
        # нельзя, но образец всё равно годится — glyph_match отфильтрует
        # его по пропорциям и IoU.
        notes.append("Толщина основной линии неизвестна: высота относительно неё не проверена. Для этой проверки используйте --line-width.")
    if source is not None and not source.strip():
        raise ValueError("--source не должен быть пустым")
    if source is not None:
        source_value, source_kind = source, "drawing"
    elif context is not None:
        source_value, source_kind = context["source"], "drawing"
    else:
        # Нет ни --source, ни report.json — единственное, что известно
        # о происхождении, это путь к самой маске.
        source_value, source_kind = str(mask_path), "mask"
    record = {"name": name, "source": source_value, "box": list(box),
              "source_kind": source_kind, "mask": str(mask_path),
              "input_foreground": "black" if invert else "white"}
    if line_width is not None:
        record["line_width"] = line_width
    return sample, record, notes

@contextmanager
def collection_lock(directory):
    """Эксклюзивная блокировка набора на время обновления (flock, non-blocking).

    Постоянный файл блокировки не удаляется: иначе возможна гонка двух
    inode (процесс A удалил, процесс B создал заново и взял другой lock
    на другом inode). flock автоматически освобождается и при аварийном
    завершении процесса.

    Если блокировку уже держит кто-то другой, поднимается ValueError с
    понятным сообщением, а не таймаут/зависание.
    """
    with (directory / ".annotation_templates.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Другой процесс обновляет этот набор; дождитесь завершения") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

def stage_file(target, content):
    """Подготовить файл на той же файловой системе для os.replace.

    Возвращает путь к временному файлу рядом с target. Содержимое уже
    записано и fsync'нуто, права скопированы с target (если он есть).
    os.replace затем может атомарно подменить target на этот файл.

    При любой ошибке (включая KeyboardInterrupt) временный файл удаляется.
    """
    fd, filename = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    path = Path(filename)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(target.stat().st_mode & 0o777)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise

def install_pair(directory, originals, replacements, backup):
    """Установить пару файлов (NPZ+JSON) согласованно, с откатом при сбое.

    Алгоритм:
      1. Записать оба новых содержимых в staged-файлы на той же ФС.
      2. Перечитать текущие файлы и сверить с originals. Если кто-то
         успел их изменить между нашим чтением и установкой — отменить
         обновление (чужие изменения не затираем).
      3. Заменить оба файла через os.replace (атомарно на POSIX).

    Если на шаге 3 второй replace падает, оба уже установленных файла
    восстанавливаются из originals. Если восстановление тоже падает —
    поднимается OSError с указанием backup, откуда можно взять копию.

    KeyboardInterrupt/SystemExit пробрасываются после отката как есть,
    чтобы Ctrl-C вёл себя как Ctrl-C, а не как ошибка обновления.
    """
    staged, installed = {}, []
    try:
        for filename, content in replacements.items():
            staged[filename] = stage_file(directory / filename, content)
        for filename, content in originals.items():
            if (directory / filename).read_bytes() != content:
                raise ValueError("Рабочий набор изменён другим способом во время подготовки; обновление отменено")
        for filename, path in staged.items():
            os.replace(path, directory / filename)
            installed.append(filename)
    except BaseException as exc:
        errors = []
        for filename in reversed(installed):
            rollback = None
            try:
                rollback = stage_file(directory / filename, originals[filename])
                os.replace(rollback, directory / filename)
            except BaseException as restore_error:
                errors.append(f"{filename}: {restore_error}")
            finally:
                if rollback is not None:
                    rollback.unlink(missing_ok=True)
        if errors:
            raise OSError(f"Обновление не завершено. Восстановите оба файла из {backup}. Ошибки отката: {'; '.join(errors)}") from exc
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise OSError(f"Обновление отменено; прежний набор сохранён. Резервная копия: {backup}. Причина: {exc}") from exc
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)

def add_template(mask_path, box, name, *, templates_dir=ROOT, backup_dir=None,
                 invert=False, source=None, line_width=None, replace=False, dry_run=False):
    """Добавить (или заменить) образец в наборе NPZ+JSON.

    Возвращает словарь-результат:
      * name, operation ("add" | "replace"), count — состояние набора;
      * record — запись, которая записана (или была бы записана) в JSON;
      * notes — нефатальные предупреждения (см. prepare_sample);
      * dry_run — флаг режима;
      * templates_dir — куда пишем;
      * backup — путь к каталогу резервной копии или None (при dry_run).

    Порядок работы:
      1. prepare_sample — все проверки вырезки до каких-либо изменений.
      2. Проверить, что templates_dir существует, а NPZ/JSON — обычные
         файлы (не симлинки).
      3. Под flock прочитать текущие байты, разобрать load_collection,
         проверить дубликат name (без --replace — ValueError).
      4. Сформировать новые байты NPZ и JSON в памяти.
      5. При dry_run — вернуться сразу, ничего не записав.
      6. Валидировать новые байты тем же load_collection (защита от
         расхождения форматов чтения/записи).
      7. Создать каталог резервной копии со снимком старых файлов,
         превью и change.json.
      8. install_pair — атомарно подменить пару с откатом при сбое.
    """
    sample, record, notes = prepare_sample(Path(mask_path), box, name, invert=invert,
                                         source=source, line_width=line_width)
    directory = Path(templates_dir).resolve()
    if not directory.is_dir():
        raise ValueError(f"Папка набора не найдена: {directory}")
    for filename in (NPZ_NAME, JSON_NAME):
        path = directory / filename
        # Симлинк опасен: os.replace заменит цель симлинка, а не сам
        # симлинк, и «набор» может уехать в неожиданное место.
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Ожидается обычный существующий файл: {path}")
    with collection_lock(directory):
        originals = {filename: (directory / filename).read_bytes() for filename in (NPZ_NAME, JSON_NAME)}
        templates, sources = load_collection(originals[NPZ_NAME], originals[JSON_NAME])
        exists = name in templates
        if exists and not replace:
            raise ValueError(f"Имя {name!r} уже существует. Выберите другое имя или явно укажите --replace")
        templates[name] = sample
        if exists:
            # При замене сохраняем позицию записи в JSON, чтобы diff был
            # минимальным и не переставлял записи.
            sources = [record if item["name"] == name else item for item in sources]
        else:
            sources.append(record)
        result = {"name": name, "operation": "replace" if exists else "add",
                  "count": len(templates), "record": record, "notes": notes,
                  "dry_run": dry_run, "templates_dir": str(directory), "backup": None}
        if dry_run:
            return result

        buffer = io.BytesIO()
        np.savez_compressed(buffer, **templates)
        replacements = {NPZ_NAME: buffer.getvalue(),
                        JSON_NAME: (json.dumps(sources, ensure_ascii=False, indent=2)+"\n").encode("utf-8")}
        # Проверяем согласованность подготовленных файлов до изменения рабочих.
        load_collection(replacements[NPZ_NAME], replacements[JSON_NAME])
        backup_root = Path(backup_dir).resolve() if backup_dir is not None else directory / "резервные_шаблоны"
        backup_root.mkdir(parents=True, exist_ok=True)
        # Каждый запуск создаёт новый подкаталог с таймстампом в UTC —
        # история резервных копий не перезаписывает предыдущие.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ_")
        backup = Path(tempfile.mkdtemp(prefix=stamp, dir=backup_root))
        for filename, content in originals.items():
            (backup / filename).write_bytes(content)
        # Превью нового образца — инверсия, чтобы символ был чёрным на белом.
        ok, preview = cv2.imencode(".png", 255-sample)
        if not ok:
            raise ValueError("Не удалось создать превью образца; рабочий набор не изменён")
        preview.tofile(backup / "new_template_preview.png")
        result["backup"] = str(backup)
        (backup / "change.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        install_pair(directory, originals, replacements, backup)
        return result

def main(argv=None):
    """CLI-обёртка: разобрать аргументы, вызвать add_template, напечатать итог.

    Коды возврата: 0 — успех (в т.ч. dry_run), 1 — ValueError/OSError/
    битый архив, 130 — Ctrl-C.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask", type=Path, required=True, help="Бинарная маска исходного размера (обычно __mask.png)")
    parser.add_argument("--box", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--name", required=True, help="Уникальное имя: буквы, цифры, подчёркивания")
    parser.add_argument("--source", help="Исходный чертёж для JSON; иначе определяется по соседнему report.json")
    parser.add_argument("--line-width", type=float, help="Толщина основной линии для проверки высоты; иначе берётся из отчёта")
    parser.add_argument("--invert", action="store_true", help="Входная маска: чёрный символ на белом фоне")
    parser.add_argument("--replace", action="store_true", help="Разрешить замену существующего имени с резервным копированием")
    parser.add_argument("--dry-run", action="store_true", help="Только проверка; NPZ/JSON не изменяются, резервная копия не создаётся")
    parser.add_argument("--templates-dir", type=Path, default=ROOT, help="Папка NPZ и JSON; по умолчанию рядом со скриптом")
    parser.add_argument("--backup-dir", type=Path, help="Папка резервных копий; по умолчанию резервные_шаблоны в папке набора")
    args = parser.parse_args(argv)
    try:
        result = add_template(args.mask, args.box, args.name, templates_dir=args.templates_dir,
                              backup_dir=args.backup_dir, invert=args.invert, source=args.source,
                              line_width=args.line_width, replace=args.replace, dry_run=args.dry_run)
    except (OSError, ValueError, cv2.error, zipfile.BadZipFile) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Обновление прервано.", file=sys.stderr)
        return 130
    for note in result["notes"]:
        print(f"Примечание: {note}")
    if result["dry_run"]:
        print(f"Проверка пройдена: {args.name}. После установки в наборе будет {result['count']} образцов. NPZ/JSON не изменены.")
    else:
        print(f"{'Заменён' if result['operation']=='replace' else 'Добавлен'} образец: {args.name}. Всего: {result['count']}.")
        print(f"Обновлены NPZ и JSON в: {result['templates_dir']}")
        print(f"Резервная копия обоих файлов и превью: {result['backup']}")
    print(f"Источник: {result['record']['source']} ({result['record']['source_kind']}). Область: {args.box}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())