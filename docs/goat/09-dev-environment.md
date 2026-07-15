# 开发环境与依赖

## 推荐 Python 版本

开发和发布环境统一使用 Python 3.12。

原因是当前 OCR 路线依赖 `paddlepaddle==2.6.2`、`paddleocr==2.7.3` 和 `numpy>=1.26,<2` 这一组稳定组合。Python 3.13 会迫使 `numpy 1.26` 走源码构建或直接失败，不适合作为当前开发环境。

项目的 `pyproject.toml` 已限制为：

```toml
requires-python = ">=3.10,<3.13"
```

## 基础依赖

发布基线使用仓库内的 `requirements-py312.lock`。它由 Python 3.12 从
`pyproject.toml` 的 `dev` 依赖解析得到，不包含 editable 工作区路径、用户目录或敏感配置。
CI 必须先安装该锁文件，再用 `--no-deps` 安装当前源码。

```powershell
python -m pip install -r requirements-py312.lock
python -m pip install --no-deps --no-build-isolation .
```

基础依赖只包含桌面壳、截图预处理和 AI 请求所需内容：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

基础依赖适合运行 GUI、悬浮层、配置窗口、翻译 API 客户端和大部分测试。

## PaddleOCR 依赖

如果要启用 PaddleOCR，使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ocr-paddle]"
```

推荐组合：

```text
Python 3.12
numpy>=1.26,<2
paddlepaddle==2.6.2
paddleocr==2.7.3
```

## Tesseract 兜底依赖

如果要启用 Tesseract 兜底，需要先安装系统级 Tesseract OCR，再安装 Python 包：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ocr-tesseract]"
```

## 当前注意事项

- 打包成 exe 后，用户不需要自己安装 Python。
- 开发环境先求稳定，不建议在 Python 3.13 上硬凑 PaddleOCR。
- OCR 依赖体积较大，后续打包时需要单独优化体积和模型路径。
- Prompt 系统和 AI 翻译不依赖 PaddleOCR，但依赖 OCR 输出质量。

## 发布验收

构建完成不代表 OCR 已能在冻结环境中初始化。发布前必须执行：

```powershell
python -m pytest -q
python -m PyInstaller --noconfirm --clean instant-translate.spec
dist\instant-translate\instant-translate.exe --smoke-ocr
```

最后一个命令会真实初始化 PaddleOCR 并做一次小图推理；退出码非 `0` 时不得发布。
