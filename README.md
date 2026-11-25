# Tekti

**Tekti** — CLI ИИ-агент для статического ревью документов с требованиями до начала разработки (поддерживаются `.md`, `.markdown`, `.txt` и текстовые `.pdf` без OCR).

Агент выделяет требования из документа, проверяет каждое требование и весь набор целиком по закрытому перечню критериев качества **ISO/IEC/IEEE 29148:2018** (п. 5.2.5, 5.2.6) и **IEEE Std 830-1998** (п. 4.3), а также выявляет встроенные инструкции, адресованные модели или проверяющему (`embedded_instruction`). По итогам прогона формируется отчёт на русском языке с точными дословными цитатами из документа, адресными уточняющими вопросами и детерминированным вердиктом о готовности к разработке и тестированию.

---

## Быстрый старт

### Требования к окружению

- Python **3.12+**
- Менеджер пакетов [`uv`](https://docs.astral.sh/uv/)

Установите зависимости проекта:

```bash
uv sync
```

### Вариант А — бесплатный ключ Gemini API (Google AI Studio)

Не требует платной подписки или аккаунта AWS.

1. Получите бесплатный API-ключ в Google AI Studio: [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Скопируйте шаблон конфигурации и вставьте свой ключ в `TEKTI_API_KEY`:
   ```bash
   cp .env.example .env
   ```
   В `.env.example` уже преднастроен профиль для Google AI Studio через OpenAI-совместимый эндпоинт:
   - `TEKTI_PROVIDER=openai_compatible`
   - `TEKTI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`
   - `TEKTI_PANEL=google:gemini-2.5-flash` (при необходимости можно указать, например, `google:gemini-2.5-pro`)
3. Запустите проверку одной командой:
   ```bash
   uv run tekti path/to/requirements.md path/to/report.md --mode fast
   ```

### Вариант Б — полный консилиум моделей на Amazon Bedrock (режим по умолчанию `board`)

Использует стандартную цепочку учётных данных AWS (`AWS_PROFILE` или `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, регион по умолчанию `us-east-1`) при `TEKTI_PROVIDER=bedrock` (значение по умолчанию).

По умолчанию в `src/tekti/config.py` настроены независимые семейства моделей:

- **Нарезка требований** (`DEFAULT_SEGMENTER` / `TEKTI_SEGMENTER`):
  `us.anthropic.claude-haiku-4-5-20251001-v1:0` (Anthropic)
- **Коллегия рецензентов** (`DEFAULT_BENCH` / `TEKTI_PANEL`, в режиме `board` берутся первые 3 семейства):
  1. `anthropic:us.anthropic.claude-haiku-4-5-20251001-v1:0` (Anthropic)
  2. `openai:openai.gpt-oss-120b-1:0` (OpenAI)
  3. `moonshot:us.moonshotai.kimi-k3.1` (Moonshot)
  - Дополнительные семейства в `DEFAULT_BENCH` (для перекрёстной проверки и расширения списка): `mistral:mistral.mistral-large-3-675b-instruct` (Mistral), `qwen:qwen.qwen3-vl-235b-a22b` (Qwen).
- **Формулировка уточняющих вопросов** (`DEFAULT_QUESTION_WRITER` / `TEKTI_QUESTION_WRITER`):
  `us.anthropic.claude-haiku-4-5-20251001-v1:0` (Anthropic)

> Любую из моделей по умолчанию можно переопределить через `.env` или переменные окружения (см. закомментированный блок для Bedrock в `.env.example`).

Запуск проверки (Markdown или PDF):

```bash
uv run tekti path/to/requirements.md path/to/report.md
uv run tekti path/to/requirements.pdf path/to/report.md
```

---

## Куда сохраняется результат, параметры CLI и конфигурация

### Выходные файлы

- **Отчёт в Markdown на русском языке** — сохраняется по пути второго аргумента (`path/to/report.md`).
- **Полный JSON-слепок прогона (`Run`)** — автоматически сохраняется рядом с отчётом с расширением `.run.json` (`path/to/report.run.json`) или по пути, указанному в `--run-json`.

### Параметры командной строки (`tekti`)

```text
uv run tekti <input> <report> [--mode {fast,standard,board}] [--run-json <path>] [--no-rules]
```

| Параметр | Описание |
| :--- | :--- |
| `input` | Входной файл с требованиями (`.md`, `.markdown`, `.txt`, `.pdf`). |
| `report` | Путь для сохранения итогового отчёта в Markdown (`.md`). |
| `--mode {fast,standard,board}` | Режим коллегии: `fast` — 1-е семейство из `TEKTI_PANEL`; `standard` — первые 2; `board` — первые 3 (по умолчанию). |
| `--run-json <path>` | Явный путь для сохранения JSON-данных прогона (по умолчанию `<report>.run.json`). |
| `--no-rules` | Отключить детерминированный слой проверки правилами (`tekti.rules`). |

### Переменные окружения

Настройки загружаются из файла `.env` в текущей рабочей директории; системные переменные окружения имеют приоритет над `.env`.

| Переменная | Назначение и значение по умолчанию |
| :--- | :--- |
| `TEKTI_PROVIDER` | Провайдер моделей: `bedrock` (по умолчанию) или `openai_compatible`. |
| `TEKTI_BACKEND` | Бэкенд вызова: `strands` (по умолчанию) или `converse` (`converse` только для `bedrock`). |
| `TEKTI_REGION` | AWS-регион для Bedrock (по умолчанию `AWS_REGION`, затем `AWS_DEFAULT_REGION`, затем `us-east-1`). |
| `TEKTI_PANEL` | Список моделей через запятую в формате `family:model_id` (по одному на семейство, сильнейшие первыми). Обязательна при `TEKTI_PROVIDER=openai_compatible`. |
| `TEKTI_SEGMENTER` | Модель для нарезки требований (`family:model_id` или `model_id`). |
| `TEKTI_QUESTION_WRITER` | Модель для формулировки уточняющих вопросов (`family:model_id` или `model_id`). |
| `TEKTI_LENSES` | Список углов проверки через запятую (по умолчанию все 5: `functional,api_contract,data_integration,tester,developer`). |
| `TEKTI_MAX_PARALLEL` | Максимум параллельных вызовов коллегии (по умолчанию `6`). |
| `TEKTI_MAX_OUTPUT_TOKENS` | Лимит выходных токенов на один вызов модели (по умолчанию `12000`). |
| `TEKTI_BASE_URL` | URL OpenAI-совместимого эндпоинта (обязателен при `TEKTI_PROVIDER=openai_compatible`, например `https://generativelanguage.googleapis.com/v1beta/openai/` или `http://localhost:11434/v1`). |
| `TEKTI_API_KEY` | API-ключ для `TEKTI_BASE_URL` (никогда не выводится и не логируется). |

---

## Как устроен конвейер проверки

```text
ingest → segment → rules → panel → adjudicate → questions → verdict → report
 код       LLM      код     LLM     код + LLM      LLM        код      код
```

1. **`ingest`** *(код)* — чтение `.md`/`.markdown`/`.txt` и текстовых `.pdf` (через `pymupdf`) в пронумерованные блоки с сохранением путей разделов и строк таблиц (`TABLE_ROW`).
2. **`segment`** *(LLM)* — выделение требований из документа, разбивка составных требований на атомарные подпункты и проверка дословности цитат.
3. **`rules`** *(код, отключается `--no-rules`)* — детерминированный слой (`tekti.rules`): поиск пустых разделов и заглушек, незаполненных атрибутов и чек-листов, размытых формулировок, сверка единиц измерения, типов полей запроса/ответа, HTTP-методов и статусов, а также защита от встроенных инструкций (`embedded_instruction`).
4. **`panel`** *(LLM, параллельно)* — проверка документа и выделенных требований коллегией из 5 ролевых углов (`functional`, `api_contract`, `data_integration`, `tester`, `developer`) на выбранных семействах моделей.
5. **`adjudicate`** *(код + LLM)* — проверка точного вхождения каждой цитаты в текст документа (после нормализации пробелов), дедупликация замечаний и перекрёстная проверка (`challenge`) моделью другого семейства для замечаний, поднятых только одним семейством.
6. **`questions`** *(LLM)* — группировка подтверждённых замечаний в приоритизированные уточняющие вопросы с указанием адресата (роли), вариантов ответа и того, что вопрос блокирует.
7. **`verdict`** *(код)* — детерминированный расчёт готовности к разработке (отсутствие подтверждённых критичных дефектов) и к тестированию (готовность к разработке и отсутствие блокирующих дефектов по критериям проверяемости и валидируемости).
8. **`report`** *(код)* — генерация 7-секционного отчёта на русском языке строго из данных `Run`:
   1. Паспорт прогона
   2. Как агент нарезал требования
   3. Проверка каждого требования
   4. Проверка всего набора требований
   5. Уточняющие вопросы
   6. Сводка
   7. Итог

---

## Запуск через Docker

Сборка образа:

```bash
docker build -t tekti .
```

Запуск проверки на локальных файлах из каталога `./data` с конфигурацией из `.env`:

```bash
docker run --rm --env-file .env -v "$PWD/data:/data" tekti /data/requirements.md /data/report.md
```

Дополнительные флаги CLI (например, `--mode fast` или `--no-rules`) передаются в конец команды:

```bash
docker run --rm --env-file .env -v "$PWD/data:/data" tekti /data/requirements.md /data/report.md --mode fast
```

---

## Тесты и оценка по эталонному набору (scorer)

### Офлайн-тесты и линтер

Тесты полностью детерминированы, работают без сети и вызовов LLM:

```bash
uv run pytest
uv run ruff check src tests
```

### Оценка качества по эталону (`evals/gold.yaml`)

Замер полноты (recall) по всем и критичным дефектам, доли непокрытых замечаний, а также сравнение нескольких прогонов бок о бок с расчётом времени и стоимости по `Run.calls`:

```bash
uv run python -m tekti.evals runs/board.run.json
uv run python -m tekti.evals runs/fast.run.json runs/board-no-rules.run.json runs/board.run.json
```

Поддерживаются флаги `--gold <path>`, `--format {markdown,json}`, `--out <path>`, `--json-out <path>`, `--label <name>`.
