
# LLM 聊天系统

一个基于 Ray 的本地大模型聊天和训练平台，支持在 CPU 环境下运行。

## 功能特性

- 加载和使用本地大语言模型
- 与模型进行交互式聊天
- 训练自定义模型（模拟实现）
- 基于 Ray 的分布式计算支持
- 美观的 Web 界面

## 系统要求

- Python 3.8 或更高版本
- Ray 已安装在服务器上
- 足够的 CPU 内存（建议 8GB 以上）

## 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：requirements.txt中不包含Ray，因为Ray已在服务器上安装。如果在其他环境中运行，请确保先安装Ray。

## 使用方法

### 1. 准备模型

将您的本地大语言模型（如 LLaMA、ChatGLM 等）放置在 `local_llm_modes` 文件夹中。确保模型包含必要的文件（如 config.json、pytorch_model.bin 等）。

例如：
```
llm_chat/
└── local_llm_modes/
    └── your_model_name/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        └── ...
```

### 2. 启动服务器

```bash
python app.py
```

服务器将在 http://localhost:5000 启动。

### 3. 使用 Web 界面

打开浏览器，访问 http://localhost:5000，您将看到以下界面：

- **左侧面板**：模型管理和训练功能
  - 选择并加载本地模型
  - 设置训练参数并开始训练
  - 查看系统状态

- **右侧面板**：聊天界面
  - 与加载的模型进行对话
  - 支持文本输入和发送

## 项目结构

```
llm_chat/
├── app.py              # 主程序文件
├── requirements.txt    # 项目依赖
├── templates/          # HTML 模板
│   └── index.html      # 主界面
└── README.md           # 项目说明
```

## API 接口

### 获取模型列表
```
GET /api/models
```
返回本地可用的模型列表。

### 加载模型
```
POST /api/load_model
Body: {"model_path": "模型路径"}
```
加载指定的本地模型。

### 聊天接口
```
POST /api/chat
Body: {"prompt": "用户提问", "max_length": 2048}
```
与加载的模型进行对话。

### 训练模型
```
POST /api/train
Body: {"data_path": "数据路径", "model_save_path": "保存路径", "epochs": 3}
```
训练自定义模型（当前为模拟实现）。

## 注意事项

1. 当前版本使用 CPU 进行模型加载和推理，速度可能较慢
2. 加载大型模型可能需要较多内存，请确保系统有足够的可用内存
3. 模型训练功能当前为模拟实现，在实际应用中需要替换为真实的训练逻辑
4. 如需使用 GPU 加速，请修改 app.py 中的相关配置

## 许可证

MIT