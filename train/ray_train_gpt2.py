import ray
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch

def train_gpt2_model(text_path, model_path, output_dir, epochs=10):
    """
    训练GPT-2模型的函数
    
    参数:
    text_path: 训练文本的路径，文件中每行一个训练样本
    model_path: 基础模型路径
    output_dir: 模型保存目录
    epochs: 训练轮数
    
    返回:
    str: 训练结果消息
    """
    try:
        # 初始化Ray
        ray.init(ignore_reinit_error=True)
        
        print(f"开始训练GPT-2模型...")
        print(f"文本数据路径: {text_path}")
        print(f"基础模型路径: {model_path}")
        print(f"输出目录: {output_dir}")
        print(f"训练轮数: {epochs}")
        
        # 检查GPU是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        
        # 加载模型和分词器
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 设置pad_token为eos_token
        tokenizer.pad_token = tokenizer.eos_token
        
        # 从文件读取训练数据
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"训练文本文件不存在: {text_path}")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        data = {"text": lines}
        
        # 创建数据集
        dataset = Dataset.from_dict(data)
        
        # 数据预处理函数
        def preprocess_function(examples):
            inputs = examples["text"]
            model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        # 处理数据
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        logs_dir = os.path.join(output_dir, "logs")
        results_dir = os.path.join(output_dir, "results")
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=results_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=logs_dir,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            report_to="tensorboard",
            no_cuda=not torch.cuda.is_available(),  # 根据设备选择
            gradient_accumulation_steps=4,
            fp16=torch.cuda.is_available(),  # GPU可用时启用半精度训练
        )
        
        # 创建Trainer对象
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        model_save_path = os.path.join(output_dir, "finetuned_gpt2")
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        print(f"模型训练完成并已保存到{model_save_path}目录。")
        
        return f"训练成功，模型已保存到{model_save_path}"
        
    except Exception as e:
        print(f"训练模型时发生错误: {e}")
        return f"训练失败: {str(e)}"
        
# 如果作为主脚本运行，可以用于测试
if __name__ == "__main__":
    # 示例用法
    base_dir = os.path.dirname(os.path.abspath(__file__))
    text_path = os.path.join(base_dir, "example_data.txt")
    model_path = "/home/ubuntu/python_works/llm-chat/local_llm_modes/gpt2"
    output_dir = os.path.join(base_dir, "output")
    
    # 如果示例数据文件不存在，创建一个简单的示例
    if not os.path.exists(text_path):
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("成都市是中国物流行业发展的中心。\n")
            f.write("成都市是中国西南地区重要的物流中心，拥有发达的交通网络和广泛的物流企业。\n")
            f.write("成都市的物流运输系统包括公路、铁路、航空和水运。\n")
    
    result = train_gpt2_model(text_path, model_path, output_dir, epochs=3)
    print(result)